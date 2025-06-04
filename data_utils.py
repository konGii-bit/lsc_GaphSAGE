import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from config import DATA_PATH, TEST_SPLIT, RANDOM_SEED

def load_and_split_edges():
    """
    Loads a CSV file, filters and maps proteins to indices, 
    and splits the list of edges into training and testing sets.
    Returns a tuple: (Data for training, test_pos_edges_numpy, number of nodes).
    """
    # 1) Load data
    df = pd.read_csv(
        DATA_PATH,
        sep=r'\s+',
        names=['protein1', 'protein2', 'combined_score'],
        dtype={'protein1': str, 'protein2': str, 'combined_score': str},
        engine='python'
    )
    # 2) Filter out invalid/short scores and convert to float
    df = df[pd.to_numeric(df['combined_score'], errors='coerce').notnull()]
    df['combined_score'] = df['combined_score'].astype(float)

    # 3) Map protein → numeric index
    genes = pd.unique(df[['protein1', 'protein2']].values.ravel())
    mapping = {g: i for i, g in enumerate(genes)}

    u = df['protein1'].map(mapping).to_numpy()
    v = df['protein2'].map(mapping).to_numpy()

    # 4) Train/test split
    all_idx = np.arange(len(u))
    train_idx, test_idx = train_test_split(
        all_idx, test_size=TEST_SPLIT, random_state=RANDOM_SEED
    )
    u_train, v_train = u[train_idx], v[train_idx]
    u_test, v_test   = u[test_idx],   v[test_idx]

    # 5) Build symmetric training edges
    train_edges = np.vstack([u_train, v_train])
    train_edges = np.concatenate([train_edges, train_edges[::-1]], axis=1)

    # 6) Build symmetric positive test edges
    test_pos_edges = np.vstack([u_test, v_test])
    test_pos_edges = np.concatenate([test_pos_edges, test_pos_edges[::-1]], axis=1)

    num_nodes = len(genes)
    # Convert to PyTorch tensor
    train_edge_index = torch.from_numpy(train_edges.astype(np.int64))

    data = Data(edge_index=train_edge_index, num_nodes=num_nodes)
    return data, test_pos_edges, num_nodes
