from torch_geometric.nn import GCNConv
import torch
import torch.nn.functional as F
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import Planetoid
import torchmetrics

from injection_attack import PoisonedCora
from unlearn import SSD, Scrub
from unlearn import *
from models import getGNN
from opts import parse_args
from train import *
opt = parse_args()

seed = 1235
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
gen = torch.Generator()
gen.manual_seed(seed)

poison_tensor_size=1
criterion = nn.CrossEntropyLoss()

dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=NormalizeFeatures(), split='public')
original_data = dataset[0]
poisoned_data = PoisonedCora(dataset=dataset, poison_tensor_size=5, num_nodes_to_inject=15, seed=seed,target_label=2)
poisoned_test_data = PoisonedCora(dataset=dataset, poison_tensor_size=5, num_nodes_to_inject=15, seed=seed, is_test=True, test_with_poison=True, target_label=2)

model2 = getGNN(dataset) # Using clean to instantiate, as it doesn't really matter: Num classes, and features don't change
optimizer = torch.optim.Adam(model2.parameters(), lr=0.025, weight_decay=5e-4)
train(model2, poisoned_data.data, optimizer, criterion = criterion, num_epochs=200)

# Clean Accuracy
acc = evaluate(model2, original_data)
print("Accuracy on the clean data: ", acc)

# Poison Success Rate
acc = evaluate(model2, poisoned_test_data.data)
print("Poison Success Rate: ", acc)