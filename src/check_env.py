"""Verify torch + PyG install with a trivial GCN forward pass."""
import torch
import torch_geometric
from torch_geometric.nn import GCNConv

print("torch", torch.__version__, "| pyg", torch_geometric.__version__)
print("cuda:", torch.cuda.is_available(), "| mps:", torch.backends.mps.is_available())
torch.manual_seed(0)
x = torch.randn(4, 8)
edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
out = GCNConv(8, 2)(x, edge_index)
print("GCN out shape:", tuple(out.shape))
