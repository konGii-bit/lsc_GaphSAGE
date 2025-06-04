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


def log_results(model_name: str,
                metrics: dict,
                fpr: np.ndarray = None,
                tpr: np.ndarray = None,
                roc_save_filename: str = None,
                train_loss_history: list = None):
    """
    Saves:
      1) Metrics to a JSON file
      2) ROC curve to a PNG file
      3) Training loss curve (loss vs epoch) to a PNG file
      4) Scalar metrics and (optionally) loss values to TensorBoard

    model_name: unique name of the model, e.g. "GraphSAGE"
    metrics: dictionary with keys 'auc_roc', 'avg_prec', 'f1'
    fpr, tpr: arrays for drawing ROC (optional)
    roc_save_filename: PNG filename for the ROC plot – if None, it will be auto-generated
    train_loss_history: list of loss values per epoch (optional)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Create result and TensorBoard directories for the model ---
    model_results_dir = os.path.join(RESULTS_DIR, model_name)
    os.makedirs(model_results_dir, exist_ok=True)

    tb_subdir = f"{model_name}_{timestamp}"
    tb_log_dir = os.path.join(TB_LOG_DIR, tb_subdir)
    os.makedirs(tb_log_dir, exist_ok=True)

    # --- Save metrics to JSON ---
    json_filename = f"{model_name}_results_{timestamp}.json"
    json_path = os.path.join(model_results_dir, json_filename)
    output = {
        "model": model_name,
        "timestamp": timestamp,
        "auc_roc": metrics["auc_roc"],
        "avg_precision": metrics["avg_prec"],
        "f1_score": metrics["f1"],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved to JSON: {json_path}")

    # --- Save ROC curve to PNG (if available) ---
    if fpr is not None and tpr is not None:
        if roc_save_filename is None:
            roc_save_filename = f"{model_name}_ROC_{timestamp}.png"
        roc_png_path = os.path.join(model_results_dir, roc_save_filename)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {metrics['auc_roc']:.4f}")
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {model_name}")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(roc_png_path)
        plt.close()
        print(f"ROC curve saved to: {roc_png_path}")

    # --- Save training loss curve to PNG ---
    if train_loss_history is not None and len(train_loss_history) > 0:
        loss_png_filename = f"{model_name}_train_loss_{timestamp}.png"
        loss_png_path = os.path.join(model_results_dir, loss_png_filename)
        plt.figure()
        plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Train Loss")
        plt.title(f"Training Loss Curve - {model_name}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(loss_png_path)
        plt.close()
        print(f"Training curve saved to: {loss_png_path}")

    # --- Write metrics to TensorBoard ---
    writer = SummaryWriter(log_dir=tb_log_dir)
    writer.add_scalar("Eval/AUC-ROC", metrics["auc_roc"], 0)
    writer.add_scalar("Eval/Average_Precision", metrics["avg_prec"], 0)
    writer.add_scalar("Eval/F1_Score", metrics["f1"], 0)

    if train_loss_history is not None:
        for epoch_idx, loss_val in enumerate(train_loss_history, start=1):
            writer.add_scalar("Train/Loss", loss_val, epoch_idx)

    # Optionally add ROC plot to TensorBoard
    if fpr is not None and tpr is not None:
        from PIL import Image
        img = Image.open(roc_png_path).convert("RGB")
        img_arr = np.array(img)
        writer.add_image("ROC_Curve", img_arr.transpose(2, 0, 1), dataformats="CHW")

    writer.close()
    print(f"TensorBoard logs saved to: {tb_log_dir}")


def compute_metrics(model, data, test_pos_edges, neg_sampling_method, device):
    """
    Generates a test set (positive + negative edges) and calculates:
      - AUC-ROC
      - Average Precision
      - F1-score (threshold=0.5)
      - FPR, TPR, thresholds for ROC

    Returns a dictionary with: 'probs', 'labels', 'auc_roc', 'avg_prec', 'f1', 'fpr', 'tpr', 'roc_thresholds'
    """
    model.eval()

    pos_edge_index = torch.from_numpy(test_pos_edges.astype(np.int64)).to(device)
    num_pos = pos_edge_index.size(1)
    pos_labels = torch.ones(num_pos, device=device)

    neg_edge_index = torch_geometric.utils.negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=num_pos,
        method=neg_sampling_method
    ).to(device)
    neg_labels = torch.zeros(num_pos, device=device)

    edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    labels = torch.cat([pos_labels, neg_labels], dim=0)

    logits = model(data.edge_index, edge_label_index)
    probs = torch.sigmoid(logits).cpu().detach().numpy()
    labels_np = labels.cpu().numpy()

    auc_roc = roc_auc_score(labels_np, probs)
    avg_prec = average_precision_score(labels_np, probs)
    pred_labels = (probs >= 0.5).astype(int)
    f1 = f1_score(labels_np, pred_labels)
    fpr, tpr, thresholds = roc_curve(labels_np, probs)

    return {
        'probs': probs,
        'labels': labels_np,
        'auc_roc': auc_roc,
        'avg_prec': avg_prec,
        'f1': f1,
        'fpr': fpr,
        'tpr': tpr,
        'roc_thresholds': thresholds
    }
