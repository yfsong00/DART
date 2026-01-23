import numpy as np

import torch
import scipy.sparse as sp

import json
import dgl
from dgl.data import (
    CoraGraphDataset, 
    CiteseerGraphDataset, 
    PubmedGraphDataset
)
from ogb.nodeproppred import DglNodePropPredDataset
from sklearn.preprocessing import StandardScaler

from utils import get_missing_feature_mask
from filling_strategies import filling
from tqdm import tqdm
import copy

import random

import json
import numpy as np
import torch
import dgl
import scipy.sparse as sp


class SailingGraphDataset:
    def __init__(self, root='./dataset/sailing'):
        self.root = root
        self._load_data()
        self.num_classes = len(set(self.graph.ndata["label"].numpy()))  # 计算类的数量

    def _load_data(self):
        # Load adjacency matrix
        adj_full = sp.load_npz(f'{self.root}/adj_full.npz')
        edge_index1, edge_index2 = adj_full.nonzero()
        self.graph = dgl.graph((edge_index1, edge_index2))

        # Load features
        raw_feat = np.load(f'{self.root}/feats.npy')
        self.graph.ndata["feat"] = torch.from_numpy(raw_feat).float()

        # Load class labels
        with open(f'{self.root}/class_map.json', 'r') as f:
            class_label = json.load(f)
        node_labels = torch.tensor([class_label[str(i)] for i in range(self.graph.number_of_nodes())])
        self.graph.ndata["label"] = node_labels

        # Load roles
        with open(f'{self.root}/role.json', 'r') as f:
            role = json.load(f)
        train_indices = role['tr']
        val_indices = role['va']
        test_indices = role['te']

        # Create masks
        train_mask = torch.zeros(self.graph.num_nodes(), dtype=torch.bool).scatter_(0, torch.tensor(train_indices),
                                                                                    True)
        val_mask = torch.zeros(self.graph.num_nodes(), dtype=torch.bool).scatter_(0, torch.tensor(val_indices), True)
        test_mask = torch.zeros(self.graph.num_nodes(), dtype=torch.bool).scatter_(0, torch.tensor(test_indices), True)

        # Assign masks to the graph
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask

    def __getitem__(self, idx):
        return self.graph

    def __len__(self):
        return 1  # Since there's only one graph in this dataset


# Update the GRAPH_DICT with the new dataset
GRAPH_DICT = {
    "cora": CoraGraphDataset,
    "citeseer": CiteseerGraphDataset,
    "pubmed": PubmedGraphDataset,
    "ogbn-arxiv": DglNodePropPredDataset,
    "sailing": lambda root='./dataset/sailing': SailingGraphDataset(root=root),
}

def scale_feats(x):
    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats

import dgl
import torch
import random
from tqdm import tqdm

def label_connect(g):
    labels = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    device = g.device

    # 创建一个新的图，具有与原始图相同的节点数
    lc_g = dgl.graph(([], []), num_nodes=g.num_nodes()).to(device)

    # 复制原始图的节点数据
    lc_g.ndata["feat"] = g.ndata["feat"]
    lc_g.ndata["label"] = g.ndata["label"]
    lc_g.ndata["train_mask"] = g.ndata["train_mask"]
    lc_g.ndata["val_mask"] = g.ndata["val_mask"]
    lc_g.ndata["test_mask"] = g.ndata["test_mask"]

    # 将原始边添加到新图中
    src, dst = g.edges()
    lc_g.add_edges(src, dst)

    new_edges = []

    # 获取训练节点的索引
    train_nodes = torch.nonzero(train_mask, as_tuple=False).squeeze().tolist()

    # 遍历所有训练节点
    for node in tqdm(train_nodes):
        # 随机选择20个训练节点
        random_nodes = random.sample(train_nodes, min(20, len(train_nodes)))
        same_label_neighbors = [neighbor for neighbor in random_nodes if labels[node].item() == labels[neighbor].item() and node != neighbor]

        for neighbor in same_label_neighbors:
            if not lc_g.has_edges_between(node, neighbor):
                new_edges.append((node, neighbor))

    if new_edges:
        src, dst = zip(*new_edges)
        lc_g.add_edges(torch.tensor(src, device=device), torch.tensor(dst, device=device))

    return lc_g

def load_transductive_dataset(dataset_name, args):
    assert dataset_name in GRAPH_DICT, f"Unknow dataset: {dataset_name}."
    if dataset_name.startswith("ogbn"):
        dataset = GRAPH_DICT[dataset_name](dataset_name)
    else:
        dataset = GRAPH_DICT[dataset_name]()

    if dataset_name == "ogbn-arxiv":
        graph, node_labels = dataset[0]
        graph = dgl.add_reverse_edges(graph)
        num_nodes = graph.num_nodes()

        idx_split = dataset.get_idx_split()
        train_nids = idx_split['train']
        valid_nids = idx_split['valid']
        test_nids = idx_split['test']

        feat = graph.ndata["feat"]
        feat = scale_feats(feat)
        graph.ndata["feat"] = feat

        graph.ndata["train_mask"] = torch.full((num_nodes,), False).index_fill_(0, train_nids, True)
        graph.ndata["val_mask"] = torch.full((num_nodes,), False).index_fill_(0, valid_nids, True)
        graph.ndata["test_mask"] = torch.full((num_nodes,), False).index_fill_(0, test_nids, True)
        graph.ndata["label"] = node_labels[:, 0]
        num_classes = (node_labels.max() + 1).item()

    else:
        graph = dataset[0]
        num_classes = dataset.num_classes

    if args.use_label_connect:
        label_g = label_connect(graph)
    else:
        label_g = graph

    labels = graph.ndata['label']
    torch.save({'features': graph.ndata['feat'], 'label': labels}, 'x_raw.pt')

    graph = graph.remove_self_loop()
    graph = graph.add_self_loop()

    label_g = label_g.remove_self_loop()
    label_g = label_g.add_self_loop()

    num_features = graph.ndata["feat"].shape[1]
      
    return graph, label_g, (num_features, num_classes)

def get_inductive_dataset(dataset_name):
    adj_full = sp.load_npz(f'dataset/{dataset_name}/adj_full.npz')
    edge_index1, edge_index2 = adj_full.nonzero()
    g = dgl.graph((edge_index1, edge_index2))
    raw_feat = np.load(f'dataset/{dataset_name}/feats.npy')
    g.ndata["feat"] = torch.from_numpy(raw_feat).float()
    with open(f'dataset/{dataset_name}/class_map.json', 'r') as f:
        class_label = json.load(f)
    node_labels = torch.tensor([class_label[str(i)] for i in range(g.number_of_nodes())])
    g.ndata["label"] = node_labels

    with open(f'dataset/{dataset_name}/role.json', 'r') as f:
        role = json.load(f)
    train_indices = role['tr']
    val_indices = role['va']
    test_indices = role['te']

    train_mask = torch.zeros(g.num_nodes(), dtype=torch.bool).scatter_(0, torch.tensor(train_indices), True)
    val_mask = torch.zeros(g.num_nodes(), dtype=torch.bool).scatter_(0, torch.tensor(val_indices), True)
    test_mask = torch.zeros(g.num_nodes(), dtype=torch.bool).scatter_(0, torch.tensor(test_indices), True)

    g.ndata['train_mask'] = train_mask
    g.ndata['val_mask'] = val_mask
    g.ndata['test_mask'] = test_mask

    num_classes = node_labels.max().item() + 1
    return g, num_classes

def load_inductive_dataset(dataset_name, args):
    if dataset_name in ["flickr", "reddit", "sailing"]:
        g, num_classes = get_inductive_dataset(dataset_name)
    else:
        raise NotImplementedError

    num_features = g.ndata["feat"].shape[1]
    train_mask = g.ndata['train_mask']

    g.ndata["feat"] = scale_feats(g.ndata["feat"])

    n_nodes = len(g.ndata["feat"])
    src, dst = g.edges()

    edge_index = torch.stack([src, dst], dim=0)

    g = g.remove_self_loop()
    g = g.add_self_loop()
    
    train_nid = torch.nonzero(train_mask, as_tuple=True)[0]
    train_g = dgl.node_subgraph(g, train_nid).clone()

    ### 同标签连接改损失函数
    if args.use_label_connect:
        label_train_g = label_connect(train_g)
    else:
        label_train_g = copy.deepcopy(train_g)

    if dataset_name == "sailing":
        overall_missing_feature_mask = ~torch.isnan(g.ndata["feat"])
        train_missing_feature_mask = ~torch.isnan(train_g.ndata["feat"])
    else:
        labels = g.ndata['label']
        torch.save({'features': g.ndata['feat'], 'labels': labels}, 'x_raw.pt')
        overall_missing_feature_mask = get_missing_feature_mask(
            rate=args.missing_rate, n_nodes=n_nodes, n_features=num_features, type=args.mask_type,
        )
        g.ndata["feat"][~overall_missing_feature_mask] = float("nan")
        train_missing_feature_mask = get_missing_feature_mask(
            rate=args.missing_rate, n_nodes=train_nid.shape[0], n_features=num_features, type=args.mask_type,
        )
        train_g.ndata["feat"][~train_missing_feature_mask] = float("nan")
        label_train_g.ndata["feat"][~train_missing_feature_mask] = float("nan")

    if args.zero_fill == True:
        train_g.ndata["feat"] = torch.where(torch.isnan(train_g.ndata["feat"]), torch.zeros_like(train_g.ndata["feat"]), train_g.ndata["feat"])
        label_train_g.ndata["feat"] = torch.where(torch.isnan(label_train_g.ndata["feat"]), torch.zeros_like(label_train_g.ndata["feat"]), label_train_g.ndata["feat"])
        g.ndata["feat"] = torch.where(torch.isnan(g.ndata["feat"]), torch.zeros_like(g.ndata["feat"]), g.ndata["feat"])
    else:
        filled_features = filling(args.filling_method, edge_index, g.ndata["feat"],
                              overall_missing_feature_mask, args.num_iterations)
        g.ndata["feat"] = torch.where(overall_missing_feature_mask, g.ndata["feat"], filled_features)
         
    return g, train_g, label_train_g, num_features, num_classes, train_missing_feature_mask

def load_link_dataset(dataset_name, args):
    if dataset_name.startswith("ogbn"):
        dataset = GRAPH_DICT[dataset_name](dataset_name)
        graph, node_labels = dataset[0]
        graph = dgl.add_reverse_edges(graph)
        num_nodes = graph.num_nodes()

        idx_split = dataset.get_idx_split()
        train_nids = idx_split['train']
        valid_nids = idx_split['valid']
        test_nids = idx_split['test']

        feat = graph.ndata["feat"]
        feat = scale_feats(feat)
        graph.ndata["feat"] = feat

        graph.ndata["train_mask"] = torch.full((num_nodes,), False).index_fill_(0, train_nids, True)
        graph.ndata["val_mask"] = torch.full((num_nodes,), False).index_fill_(0, valid_nids, True)
        graph.ndata["test_mask"] = torch.full((num_nodes,), False).index_fill_(0, test_nids, True)
        graph.ndata["label"] = node_labels[:, 0]
        num_classes = (node_labels.max() + 1).item()

        if args.use_label_connect:
            label_g = label_connect(graph)
        else:
            label_g = graph

        graph = graph.remove_self_loop()
        graph = graph.add_self_loop()

        label_g = label_g.remove_self_loop()
        label_g = label_g.add_self_loop()

        num_features = graph.ndata["feat"].shape[1]
        
        return graph, label_g, (num_features, num_classes)

    elif dataset_name in ["cora", "citeseer", "pubmed"]:
        dataset = GRAPH_DICT[dataset_name]()
        graph = dataset[0]
        num_classes = dataset.num_classes

        if args.use_label_connect:
            label_g = label_connect(graph)
        else:
            label_g = graph

        graph = graph.remove_self_loop()
        graph = graph.add_self_loop()

        label_g = label_g.remove_self_loop()
        label_g = label_g.add_self_loop()

        num_features = graph.ndata["feat"].shape[1]
        
        return graph, label_g, (num_features, num_classes)
    
    elif dataset_name in ["flickr", "reddit", "sailing"]:
        g, num_classes = get_inductive_dataset(dataset_name)
        num_features = g.ndata["feat"].shape[1]

        g.ndata["feat"] = scale_feats(g.ndata["feat"])

        ### 同标签连接改损失函数
        if args.use_label_connect:
            label_train_g = label_connect(g)
        else:
            label_train_g = g
        
        g = g.remove_self_loop()
        g = g.add_self_loop()
        label_train_g = label_train_g.remove_self_loop()
        label_train_g = label_train_g.add_self_loop()
            
        return g, label_train_g, (num_features, num_classes)