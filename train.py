import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from torch_geometric.utils import negative_sampling
from tqdm import tqdm

from config import (
    DEVICE, EPOCHS, LEARNING_RATE,
    NEG_SAMPLING_METHOD, EMBED_DIM,
    HIDDEN_DIM, ROC_PATH
)
from data_utils import load_and_split_edges
from model import GraphSAGE
from evaluate import compute_metrics, log_results

def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()

    pos_edge_index = data.edge_index
    num_pos_edges = pos_edge_index.size(1)

    neg_edge_index = negative_sampling(
        edge_index=pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=num_pos_edges,
        method=NEG_SAMPLING_METHOD
    )

    edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    labels = torch.cat([
        torch.ones(num_pos_edges),
        torch.zeros(num_pos_edges)
    ]).to(DEVICE)

    predictions = model(data.edge_index, edge_index)
    loss = criterion(predictions, labels)
    loss.backward()
    optimizer.step()

    return loss.item()

def main():

    data, test_edges, num_nodes = load_and_split_edges()
    data = data.to(DEVICE)
    train_losses = []

    model = GraphSAGE(num_nodes=num_nodes, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = BCEWithLogitsLoss()

    for epoch in tqdm(range(1, EPOCHS + 1), desc="Epoch"):
        loss = train(model, data, optimizer, criterion)
        train_losses.append(loss)

        tqdm.write(f"► Epoch {epoch}/{EPOCHS} — Loss: {loss:.4f}")

        if epoch % 5 == 0 or epoch == EPOCHS:
            metrics = compute_metrics(model, data, test_edges, NEG_SAMPLING_METHOD, DEVICE)
            print(f"[{epoch:03d}] Loss: {loss:.4f} | AUC: {metrics['auc_roc']:.4f} | AP: {metrics['avg_prec']:.4f} | F1: {metrics['f1']:.4f}")

    fpr, tpr = metrics['fpr'], metrics['tpr']
    log_results(
        model_name="GraphSAGE",
        metrics=metrics,
        fpr=fpr,
        tpr=tpr,
        roc_save_filename=ROC_PATH,
        train_loss_history=train_losses
    )

if __name__ == "__main__":
    main()
