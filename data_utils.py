import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from config import DATA_PATH, TEST_SPLIT, RANDOM_SEED, VALID_SPLIT

def load_data():
    df = pd.read_csv(
        DATA_PATH,
        sep=r'\s+',
        names=['protein1', 'protein2', 'combined_score'],
        dtype={'protein1': str, 'protein2': str, 'combined_score': str},
        engine='python'
    )
    df = df[pd.to_numeric(df['combined_score'], errors='coerce').notnull()]
    df = df[['protein1', 'protein2']]

    genes = pd.unique(df.values.ravel())
    mapping = {g: i for i, g in enumerate(genes)}
    u = df['protein1'].map(mapping).to_numpy()
    v = df['protein2'].map(mapping).to_numpy()

    # split test first
    idx = np.arange(len(u))
    train_val_idx, test_idx = train_test_split(idx, test_size=TEST_SPLIT, random_state=RANDOM_SEED)
    # then split train/val
    train_idx, val_idx = train_test_split(train_val_idx, test_size=VALID_SPLIT, random_state=RANDOM_SEED)

    def build_edge_index(indices):
        uu, vv = u[indices], v[indices]
        edge = np.vstack([np.concatenate([uu, vv]), np.concatenate([vv, uu])])
        return torch.tensor(edge, dtype=torch.long)

    train_edge_index = build_edge_index(train_idx)
    val_edge_index = build_edge_index(val_idx)
    # full graph for neighbor context: use train + val
    full_ctx = build_edge_index(np.concatenate([train_idx, val_idx]))

    num_nodes = len(genes)
    data = Data(edge_index=full_ctx, num_nodes=num_nodes)
    test_edges = (u[test_idx], v[test_idx])
    return data, train_edge_index, val_edge_index, test_edges, num_nodes