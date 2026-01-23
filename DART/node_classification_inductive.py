import logging
import yaml
import numpy as np
from tqdm import tqdm
import torch
import copy

from utils import (
    build_args,
    create_optimizer,
    set_random_seed,
)
from data_loading import load_inductive_dataset
from models import build_model
from evaluation import linear_probing_for_inductive_node_classiifcation, LogisticRegression

from filling_strategies import filling


def evaluete(args, model, full_graph, num_classes, missing_feature_mask, mute=False):
    model.eval()
    # Perform reconstruction on full_graph
    with torch.no_grad():
        feat = full_graph.ndata["feat"]
        if args.use_rec:
            full_graph_features = model.missing_embed_mask(full_graph, feat, missing_feature_mask)
        else:
            full_graph_features = model.encoder(full_graph, feat)

    if not mute:
        labels = full_graph.ndata['label']
        torch.save({'features': model.encoder(full_graph, feat), 'labels': labels}, 'x_encode.pt')
        torch.save({'features': model.get_reconstraction(full_graph, feat), 'labels': labels}, 'x_rec.pt')
        torch.save({'features': full_graph_features, 'labels': labels}, 'x_final.pt')
        print('Dropped features!')

    # Initialize dictionaries to store features and labels for train, val, and test sets
    x_all = {"train": None, "val": None, "test": None}
    y_all = {"train": None, "val": None, "test": None}

    # Extract masks
    train_mask = full_graph.ndata["train_mask"]
    val_mask = full_graph.ndata["val_mask"]
    test_mask = full_graph.ndata["test_mask"]

    # Use masks to separate features and labels
    x_all["train"] = full_graph_features[train_mask]
    y_all["train"] = full_graph.ndata["label"][train_mask]

    x_all["val"] = full_graph_features[val_mask]
    y_all["val"] = full_graph.ndata["label"][val_mask]

    x_all["test"] = full_graph_features[test_mask]
    y_all["test"] = full_graph.ndata["label"][test_mask]

    # Get input dimension for the encoder
    in_dim = x_all["train"].shape[1]

    # Initialize the Logistic Regression encoder and optimizer
    encoder = LogisticRegression(in_dim, num_classes)
    encoder = encoder.to(full_graph.device)
    optimizer_f = create_optimizer("adam", encoder, args.lr_f, args.weight_decay_f)

    x = torch.cat(list(x_all.values()))
    y = torch.cat(list(y_all.values()))
    num_train, num_val, num_test = [x.shape[0] for x in x_all.values()]
    num_nodes = num_train + num_val + num_test
    train_mask = torch.arange(num_train, device=full_graph.device)
    val_mask = torch.arange(num_train, num_train + num_val, device=full_graph.device)
    test_mask = torch.arange(num_train + num_val, num_nodes, device=full_graph.device)

    val_acc, test_acc = linear_probing_for_inductive_node_classiifcation(encoder, x, y, (train_mask, val_mask, test_mask), optimizer_f, args.max_epoch_f, mute=mute)
    return val_acc, test_acc

def pretrain(args, model, graphs, num_classes, optimizer, scheduler, missing_feature_mask):
    logging.info("start training..")
    train_g, label_train_g, g = graphs
    max_epoch = args.max_epoch
    train_edge_drop_rate = args.train_edge_drop_rate

    epoch_iter = tqdm(range(max_epoch))
    with_label_graph = label_train_g

    lc_train_src, lc_train_dst = with_label_graph.edges()
    lc_train_edge_index = torch.stack([lc_train_src, lc_train_dst], dim=0)

    train_feat = with_label_graph.ndata["feat"]

    if args.zero_fill == False:
        filled_features_train = filling(args.filling_method, lc_train_edge_index, train_feat,
                                        missing_feature_mask,
                                        args.num_iterations)
        train_feat = torch.where(missing_feature_mask, train_feat, filled_features_train)

    no_label_graph = train_g
    train_src, train_dst = no_label_graph.edges()
    train_edge_index = torch.stack([train_src, train_dst], dim=0)
    num_edges = train_edge_index.size(1)
    num_edges_to_drop = int(train_edge_drop_rate * num_edges)
    all_drop_indices = [torch.randperm(num_edges)[:num_edges_to_drop] for _ in tqdm(range(len(epoch_iter)))]

    best_val_acc = 0
    best_model = None

    for epoch, drop_indices in zip(epoch_iter, all_drop_indices):
        model.train()
        loss_list = []

        drop_mask = torch.ones(num_edges, dtype=torch.bool)
        drop_mask[drop_indices] = False
        remaining_edge_index = train_edge_index[:, drop_mask]

        train_feat_dp = no_label_graph.ndata["feat"]
        if args.zero_fill == False:
            filled_features_dp = filling(args.filling_method, remaining_edge_index, train_feat_dp,
                                            missing_feature_mask,
                                            args.num_iterations)
            train_feat_dp = torch.where(missing_feature_mask, train_feat_dp, filled_features_dp)
        loss, loss_dict = model.new_forward(no_label_graph, train_feat_dp, train_feat, missing_feature_mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

        if scheduler is not None:
            scheduler.step()

        train_loss = np.mean(loss_list)
        epoch_iter.set_description(f"# Epoch {epoch} | train_loss: {train_loss:.4f}")
        if (epoch + 1) % 100 == 0:
            val_acc, test_acc = evaluete(args, model, g, num_classes, missing_feature_mask, mute=True)
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_model = copy.deepcopy(model)
    return best_model


def main(args):
    device = args.device if args.device >= 0 else "cpu"
    (
        g,
        train_g,
        label_train_g,
        num_features, 
        num_classes,
        missing_feature_mask
    ) = load_inductive_dataset(args.dataset, args)
    args.num_features = num_features

    acc_list = []
    estp_acc_list = []
    for i, seed in enumerate(args.seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        model = build_model(args)
        model.to(device)
        optimizer = create_optimizer(args.optimizer, model, args.lr, args.weight_decay)
        g = g.to(device)
        train_g = train_g.to(device)
        label_train_g = label_train_g.to(device)
        missing_feature_mask = missing_feature_mask.to(device)

        if args.scheduler:
            logging.info("Use schedular")
            scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / args.max_epoch) ) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        else:
            scheduler = None

        model = pretrain(args, model, (train_g, label_train_g, g), num_classes, optimizer, scheduler, missing_feature_mask)

        model = model.to(device)
        model.eval()

        val_acc, test_acc = evaluete(args, model, g, num_classes, missing_feature_mask)
        acc_list.append(test_acc)

    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    print(f"# Final_Acc: {final_acc:.4f}±{final_acc_std:.4f}")


def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    if args.dataset not in configs:
        logging.info("Best args not found")
        return args

    logging.info("Using best configs")
    configs = configs[args.dataset]

    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k:
            v = float(v)
        setattr(args, k, v)
    return args

if __name__ == "__main__":
    args = build_args()
    args = load_best_configs(args, "configs/configs_inductive.yml")
    print(args)
    main(args)
