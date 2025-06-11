import torch
from torch_geometric.loader import LinkNeighborLoader
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm

from config import (
    DEVICE, EPOCHS, LEARNING_RATE,
    NEG_SAMPLING_RATIO, EMBED_DIM,
    HIDDEN_DIM, BATCH_SIZE
)
from data_utils import load_data
from model import GraphSAGE
from evaluate import evaluate, log_results

def epoch_loss(model, loader, optimizer=None, criterion=None, desc=None):
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()
    total_loss = 0
    progress = tqdm(loader, desc=desc)
    with torch.set_grad_enabled(is_train):
        for batch in progress:
            edge_label_index = batch.edge_label_index.to(DEVICE)
            labels = batch.edge_label.to(DEVICE).float()
            out = model(batch.edge_index.to(DEVICE), edge_label_index)
            loss = criterion(out, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            progress.set_postfix({'loss': f'{loss.item():.4f}'})
    return total_loss / len(loader)


def main():
    data, train_eidx, val_eidx, test_edges, num_nodes = load_data()
    # Loaders
    train_loader = LinkNeighborLoader(
        data,
        num_neighbors=[10,10],
        batch_size=BATCH_SIZE,
        edge_label_index=train_eidx,
        edge_label=torch.ones(train_eidx.size(1), dtype=torch.long),
        neg_sampling_ratio=NEG_SAMPLING_RATIO,
        shuffle=True
    )
    val_loader = LinkNeighborLoader(
        data,
        num_neighbors=[10,10],
        batch_size=BATCH_SIZE,
        edge_label_index=val_eidx,
        edge_label=torch.ones(val_eidx.size(1), dtype=torch.long),
        neg_sampling_ratio=NEG_SAMPLING_RATIO,
        shuffle=False
    )

    model = GraphSAGE(num_nodes, EMBED_DIM, HIDDEN_DIM).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = BCEWithLogitsLoss()
    train_losses, val_losses = [], []

    # Training loop with tqdm
    for epoch in range(1, EPOCHS+1):
        t_desc = f"Train Epoch {epoch}/{EPOCHS}"
        v_desc = f"Val   Epoch {epoch}/{EPOCHS}"
        t_loss = epoch_loss(model, train_loader, optimizer, criterion, desc=t_desc)
        v_loss = epoch_loss(model, val_loader, None, criterion, desc=v_desc)
        train_losses.append(t_loss)
        val_losses.append(v_loss)

    # Final evaluation
    metrics = evaluate(model, data, test_edges, DEVICE)
    print(f"Test AUC: {metrics['auc_roc']:.4f}, AP: {metrics['avg_prec']:.4f}, F1: {metrics['f1']:.4f}")
    log_results("GraphSAGE_Batched", metrics, fpr=metrics['fpr'], tpr=metrics['tpr'],
                train_losses=train_losses, val_losses=val_losses)

if __name__ == "__main__":
    main()
