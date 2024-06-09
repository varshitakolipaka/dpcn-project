import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import Planetoid
import random
import numpy as np
from injection_attack import PoisonedCora
from scrub import Scrub
from models import getGNN
from train import *
from opts import parse_args
opt = parse_args()

seed = 1235
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
gen = torch.Generator()
gen.manual_seed(seed)

criterion = nn.CrossEntropyLoss()
dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=NormalizeFeatures(), split='public')
original_data = dataset[0]

poisoned_train_data = PoisonedCora(dataset=dataset, poison_tensor_size=5, 
                             num_nodes_to_inject=15, seed=seed,target_label=2)

# Need to ensre that the poison tensor size is the same for both test and train.
poisoned_test_data = PoisonedCora(dataset=dataset, poison_tensor_size=5, 
                                  num_nodes_to_inject=15, seed=seed, is_test=True, 
                                  test_with_poison=True, target_label=2)

model = getGNN(dataset) # Using clean to initialise model, as it only takes num_classes and num_features
optimizer = torch.optim.Adam(model.parameters(), lr=0.025, weight_decay=5e-4)
train(model, poisoned_train_data.data, optimizer, criterion = criterion, num_epochs=200)

# Clean Accuracy
acc = evaluate(model, original_data)
print("Accuracy on the clean data: ", acc)

# Poison Success Rate
acc = evaluate(model, poisoned_test_data.data)
print("Poison Success Rate: ", acc)
                                             #===unlearning===#                                              

# Currently, we're training with the poisoned nodes, so this step is required.
retain_mask = poisoned_train_data.data.train_mask & ~poisoned_train_data.data.poison_mask

scrub = Scrub(opt=opt, model=model)
scrub.unlearn_nc(dataset=poisoned_train_data.data, train_mask=retain_mask, forget_mask=poisoned_train_data.data.poison_mask)

# Clean Accuracy
acc = evaluate(model, original_data)
print("Accuracy on the clean data: ", acc)

# Poison Success Rate
acc = evaluate(model, poisoned_test_data.data)
print("Poison Success Rate: ", acc)
