import logging
import numpy as np
from tqdm import tqdm
import torch
import copy

from utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    load_best_configs,
    get_edge_index,
    get_missing_feature_mask
)
from data_loading import load_transductive_dataset
from models import build_model
from evaluation import linear_probing_for_transductive_node_classiifcation, LogisticRegression
from filling_strategies import filling

def evaluate(args, model, graph, x, num_classes, missing_feature_mask, mute=False):
    model.eval()
    with torch.no_grad():
        if args.use_rec:
            full_graph_features = model.missing_embed_mask(graph, x, missing_feature_mask)
        else:
            full_graph_features = model.encoder(graph, x)

    if not mute:
        labels = graph.ndata['label']
        torch.save({'features': model.encoder(graph, x), 'labels': labels}, 'x_encode.pt')
        torch.save({'features': model.get_reconstraction(graph, x), 'labels': labels}, 'x_rec.pt')
        torch.save({'features': full_graph_features, 'labels': labels}, 'x_final.pt')
        print('Dropped features!')

    in_feat = full_graph_features.shape[1]
    encoder = LogisticRegression(in_feat, num_classes)
    
    encoder.to(x.device)
    optimizer_f = create_optimizer("adam", encoder, args.lr_f, args.weight_decay_f)

    val_acc, test_acc = linear_probing_for_transductive_node_classiifcation(encoder, graph, full_graph_features, optimizer_f, args.max_epoch_f, mute)
    return val_acc, test_acc

def pretrain(args, model, graph, x, edge_index, label_edge_index, missing_feature_mask, optimizer, scheduler, num_classes):
    logging.info("start training..")
    epoch_iter = tqdm(range(args.max_epoch))

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
        if (epoch + 1) % 50 == 0:
            val_acc, test_acc = evaluate(args, model, graph, x_with_label, num_classes, missing_feature_mask, mute=True)
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_model = copy.deepcopy(model)

    return best_model

def main(args):
    device = args.device if args.device >= 0 else "cpu"

    graph, label_g, (num_features, num_classes) = load_transductive_dataset(args.dataset, args)
    args.num_features = num_features

    acc_list = []
    for i, seed in enumerate(args.seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
            torch.cuda.empty_cache()

        model = build_model(args)
        model.to(device)
        optimizer = create_optimizer(args.optimizer, model, args.lr, args.weight_decay)

        if args.scheduler:
            logging.info("Use schedular")
            scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / args.max_epoch) ) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        else:
            scheduler = None
            
        x = graph.ndata["feat"].clone().to(device)
        n_nodes = len(x)
        graph = graph.to(device)
        label_g = label_g.to(device)

        if args.dataset != "sailing":
            missing_feature_mask = get_missing_feature_mask(
                rate=args.missing_rate, n_nodes=n_nodes, n_features=num_features, type=args.mask_type,
            ).to(device)
            x[~missing_feature_mask] = float("nan")
        else:
            missing_feature_mask = ~torch.isnan(x)

        edge_index = get_edge_index(graph).to(device)
        label_edge_index = get_edge_index(label_g).to(device)

        model = pretrain(args, model, graph, x, edge_index, label_edge_index, missing_feature_mask, optimizer, scheduler, num_classes)
        
        model.eval()
        if args.zero_fill == False:
            filled_features = filling(args.filling_method, label_edge_index, x, missing_feature_mask,
                                    args.num_iterations)
            x = torch.where(missing_feature_mask, x, filled_features)
        else:
            x = torch.where(missing_feature_mask, x, 0)
      
        val_acc, test_acc = evaluate(args, model, graph, x, num_classes, missing_feature_mask)
        acc_list.append(test_acc)

    final_acc, final_acc_std = np.mean(acc_list) * 100, np.std(acc_list) * 100
    print(f"# Final_Acc: {final_acc:.2f}% ± {final_acc_std:.2f}")

if __name__ == "__main__":
    args = build_args()
    args = load_best_configs(args, "configs/configs_transductive.yml")
    print(args)
    main(args)
