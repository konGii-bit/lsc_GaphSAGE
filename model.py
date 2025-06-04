import torch
from torch_geometric.nn import SAGEConv

class GraphSAGE(torch.nn.Module):
    """
    A simple GraphSAGE model with node embeddings.
    """
    def __init__(self, num_nodes, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_nodes, embed_dim)
        self.conv1 = SAGEConv(embed_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.lin   = torch.nn.Linear(hidden_dim * 2, 1)

    def forward(self, edge_index, edge_label_index):
        x = self.embedding.weight
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        src, dst = edge_label_index
        emb = torch.cat([x[src], x[dst]], dim=1)
        return self.lin(emb).view(-1)
