from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeFeatures
from poisoning import load_poisoned_cora
import torch
from test_poison import *
import torch.nn.functional as F
from opts import parse_args
opt = parse_args()
from unlearn import SSD, Scrub
from unlearn import *
import copy
frac_poisoned = 0.05
poison_tensor_size=40
# print("Poison Tensor Length: ", poison_tensor_size)
# print("Fraction of Nodes Poisoned", frac_poisoned)


dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=NormalizeFeatures(), split='full')
data = dataset[0]
print(data.train_mask.sum())