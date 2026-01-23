import logging
import numpy as np
from tqdm import tqdm
import torch
import dgl
import copy

from utils import (
    build_args,
    create_optimizer,
    load_best_configs,
    get_edge_index,
    split_edges, 
    big_split_edges, 
    get_missing_feature_mask
)
from data_loading import load_link_dataset, label_connect
from models import build_model
from filling_strategies import filling
from models.gcn import GCNEncoder
from torch_geometric.nn import GAE
from evaluation import link_train, link_test
from torch_geometric.utils import to_undirected

def evaluate(args, model, graph, x, edge_index, pos_edge_index, neg_edge_index, missing_feature_mask):
    model.eval()
    with torch.no_grad():
        if args.use_rec:
            full_graph_features = model.missing_embed_mask(graph, x, missing_feature_mask)
        else:
            full_graph_features = x

    model = GAE(GCNEncoder(full_graph_features.shape[1], out_channels=16)).to(edge_index.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    epochs = args.max_epoch_f
    for epoch in range(1, epochs+1):
        loss = link_train(model, full_graph_features, edge_index, optimizer)
        auc, ap = link_test(model, full_graph_features, edge_index, pos_edge_index, neg_edge_index)
        if epoch % 50 == 0:
            print('epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))
    
    return auc, ap

def pretrain(args, model, graph, x, edge_index, label_edge_index, val_pos_edge_index, val_neg_edge_index, missing_feature_mask, optimizer, scheduler):
    logging.info("start training..")
    epoch_iter = tqdm(range(args.max_epoch))
    device = edge_index.device

    if args.zero_fill == False:
        filled_features = filling(args.filling_method, label_edge_index, x, missing_feature_mask,
                                args.num_iterations)
        x_with_label = torch.where(missing_feature_mask, x, filled_features)
    else:
        x_with_label = torch.where(missing_feature_mask, x, 0)

    num_edges = edge_index.size(1)
    num_edges_to_drop = int(args.train_edge_drop_rate * num_edges)
    all_drop_indices = [torch.randperm(num_edges)[:num_edges_to_drop] for _ in tqdm(range(len(epoch_iter)))]

    best_val_acc = 0
    best_model = None

    for epoch, drop_indices in zip(epoch_iter, all_drop_indices):
        model.train()
        drop_mask = torch.ones(num_edges, dtype=torch.bool)
        drop_mask[drop_indices] = False
        remaining_edge_index = edge_index[:, drop_mask]

        if args.zero_fill == False:
            no_label_filled_features = filling(args.filling_method, remaining_edge_index, x, missing_feature_mask,
                                            args.num_iterations)
            x_no_label = torch.where(missing_feature_mask, x, no_label_filled_features)
        else:
            x_no_label = torch.where(missing_feature_mask, x, 0)

        loss, loss_dict = model.new_forward(graph, x_no_label, x_with_label, missing_feature_mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")
        if (epoch + 1) % 100 == 0:
            auc, ap = evaluate(args, model, graph, x_with_label, edge_index, val_pos_edge_index, val_neg_edge_index, missing_feature_mask)
            if auc >= best_val_acc:
                best_val_acc = auc
                best_model = copy.deepcopy(model)

    return best_model

def main(args):
    device = args.device if args.device >= 0 else "cpu"

    raw_graph, raw_label_g, (num_features, num_classes) = load_link_dataset(args.dataset, args)
    args.num_features = num_features

    auc_list = []
    ap_list = []

    for i, seed in enumerate(args.seeds):
        print(f"####### Run {i} for seed {seed}")
        graph, label_g = raw_graph.clone(), raw_label_g.clone()

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        model = build_model(args)
        model.to(device)
        optimizer = create_optimizer(args.optimizer, model, args.lr, args.weight_decay)

        if args.scheduler:
            logging.info("Use schedular")
            scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / args.max_epoch) ) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        else:
            scheduler = None

        n_nodes = len(graph.ndata['feat'])
        graph = graph.to(device)
        label_g = label_g.to(device)
        x = graph.ndata['feat'].to(device)

        if args.dataset != "sailing":
            missing_feature_mask = get_missing_feature_mask(
                rate=args.missing_rate, n_nodes=n_nodes, n_features=num_features, type=args.mask_type,
            ).to(device)
            x[~missing_feature_mask] = float("nan")
        else:
            missing_feature_mask = ~torch.isnan(x)

        edge_index = get_edge_index(graph).to(device)
        
        if args.dataset == "sailing":
            train_pos_edge_index, val_pos_edge_index, test_pos_edge_index, val_neg_edge_index, test_neg_edge_index = split_edges(edge_index, val_ratio=0.1, test_ratio=0.3)
        elif args.dataset in ["cora", "citeseer", "pubmed"]:
            train_pos_edge_index, val_pos_edge_index, test_pos_edge_index, val_neg_edge_index, test_neg_edge_index = split_edges(edge_index)
        else:
            train_pos_edge_index, val_pos_edge_index, test_pos_edge_index, val_neg_edge_index, test_neg_edge_index = big_split_edges(edge_index)

        label_g = dgl.graph((train_pos_edge_index[0], train_pos_edge_index[1]), num_nodes=graph.num_nodes())
        label_g = label_g.to(device)
        
        label_g.ndata['label'] = graph.ndata['label']
        label_g.ndata['feat'] = graph.ndata['feat']
        label_g.ndata['train_mask'] = graph.ndata['train_mask']
        label_g.ndata['val_mask'] = graph.ndata['val_mask']
        label_g.ndata['test_mask'] = graph.ndata['test_mask']
        label_connected_graph = label_connect(label_g)

        new_src, new_dst = label_connected_graph.edges()
        new_edges = torch.stack([new_src, new_dst], dim=0)

        label_edge_index = torch.cat([train_pos_edge_index, new_edges], dim=1)
        label_edge_index = to_undirected(label_edge_index)

        model = pretrain(args, model, graph, x, train_pos_edge_index, label_edge_index, val_pos_edge_index, val_neg_edge_index, missing_feature_mask, optimizer, scheduler)
        
        model.eval()

        if args.zero_fill == False:
            filled_features = filling(args.filling_method, label_edge_index, x, missing_feature_mask,
                                    args.num_iterations)
            x = torch.where(missing_feature_mask, x, filled_features)
        else:
            x = torch.where(missing_feature_mask, x, 0)
        
        auc, ap = evaluate(
            args, model, graph, x, edge_index, test_pos_edge_index, test_neg_edge_index, missing_feature_mask
        )
        auc_list.append(auc)
        ap_list.append(ap)

    final_auc, final_auc_std = np.mean(auc_list), np.std(auc_list)
    final_ap, final_ap_std = np.mean(ap_list), np.std(ap_list)
    print(f"# Final_AUC: {final_auc:.4f}±{final_auc_std:.4f}")
    print(f"# Final_AP: {final_ap:.4f}±{final_ap_std:.4f}")

if __name__ == "__main__":
    args = build_args()
    args = load_best_configs(args, "configs/configs_link.yml")
    print(args)
    main(args)