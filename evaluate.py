import torch
import torch_geometric
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
from config import RESULTS_DIR, TB_LOG_DIR


def log_results(model_name, metrics, fpr=None, tpr=None, train_losses=None, val_losses=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(RESULTS_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)
    # Prepare JSON output: only serializable scalars
    output = {
        'model': model_name,
        'timestamp': timestamp,
        'auc_roc': metrics['auc_roc'],
        'avg_prec': metrics['avg_prec'],
        'f1': metrics['f1'],
    }
    json_path = os.path.join(model_dir, f"results_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    # ROC
    if fpr is not None and tpr is not None:
        plt.figure(); plt.plot(fpr, tpr, label=f"AUC={metrics['auc_roc']:.4f}"); plt.plot([0,1],[0,1],'k--',alpha=0.5)
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.grid(); plt.tight_layout()
        plt.savefig(os.path.join(model_dir, f"roc_{timestamp}.png")); plt.close()
    # Loss curves
    if train_losses:
        plt.figure(); plt.plot(range(1,len(train_losses)+1), train_losses, marker='o', label='train');
        plt.plot(range(1,len(val_losses)+1), val_losses, marker='x', label='val');
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(); plt.tight_layout()
        plt.savefig(os.path.join(model_dir, f"loss_{timestamp}.png")); plt.close()



def evaluate(model, data, test_edges, device):
    model.eval()
    u_test, v_test = test_edges
    pos = torch.tensor([u_test, v_test], dtype=torch.long)
    neg = torch_geometric.utils.negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=len(u_test)
    )
    idx = torch.cat([pos, neg], dim=1).to(device)
    labels = torch.cat([torch.ones(len(u_test)), torch.zeros(len(u_test))]).to(device)

    logits = model(data.edge_index.to(device), idx)
    probs = torch.sigmoid(logits).cpu().detach().numpy()
    labels_np = labels.cpu().numpy()

    fpr, tpr, _ = roc_curve(labels_np, probs)
    return {
        'auc_roc': roc_auc_score(labels_np, probs),
        'avg_prec': average_precision_score(labels_np, probs),
        'f1': f1_score(labels_np, (probs >= 0.5).astype(int)),
        'fpr': fpr,
        'tpr': tpr
    }