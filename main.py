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
poison_tensor_size=5
print("Poison Tensor Length: ", poison_tensor_size)
print("Fraction of Nodes Poisoned", frac_poisoned)

seed = 1235
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
gen = torch.Generator()
gen.manual_seed(seed)
# torch.use deterministic 


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.new_x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=NormalizeFeatures(), split='public')
data = dataset[0]
print(data.train_mask.sum(dim=0), data.test_mask.sum(dim=0), data.val_mask.sum(dim=0))
train_dataset = load_poisoned_cora(data, frac_poisoned=frac_poisoned, is_test=False, poison_tensor_size=poison_tensor_size, target_label=2, seed=seed)
train_data = train_dataset.data
model = GCN(num_features=dataset.num_features, num_classes=dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.025, weight_decay=5e-4)

# Training the model
def train():
    model.train()
    optimizer.zero_grad()
    out = model(train_data)
    loss = F.nll_loss(out[train_data.train_mask], train_data.new_y[train_data.train_mask])
    loss.backward()
    optimizer.step()
    return loss

# Testing the model
def test():
    model.eval()
    logits, accs = model(train_data), []
    for _, mask in train_data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(train_data.new_y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

# Run training
for epoch in range(200):
    loss = train()
    if epoch % 10 == 0:
        train_acc, val_acc, test_acc = test()

# torch.save(model.state_dict(), "./model")
# model.load_state_dict(torch.load("./model"))
# model.eval()
def test_poisoned_data(model, dataset):
    model.eval()
    with torch.no_grad():
        out = model(dataset)
        pred = out.argmax(dim=1)
        correct = pred[dataset.test_mask] == dataset.new_y[dataset.test_mask]
        accuracy = int(correct.sum()) / int(dataset.test_mask.sum())
    return accuracy

clean_test_dataset = load_poisoned_cora(data, is_test=True, test_with_poison=False)
clean_test_data = clean_test_dataset.data
clean_accuracy = test_poisoned_data(model, clean_test_data)
print(f'Test Accuracy on Clean Data: {clean_accuracy:.4f}')

adv_test_dataset = load_poisoned_cora(data, is_test=True, target_label=2, poison_tensor_size=30, test_with_poison=True)
adv_data = adv_test_dataset.data
adv_accuracy = test_poisoned_data(model, adv_data)
print(f'Poison success rate: {adv_accuracy:.4f}')

print("="*20)
print(adv_data)

#                                         #===unlearning===#                                              

# retain_mask = train_dataset.train_mask & ~train_dataset.poison_mask

# def run_scrub(model, dataset, retain_mask, delete_mask, 
#               adv_dataset, clean_test_dataset=clean_test_dataset):
    
#     scrub = Scrub(opt=opt, model=model)
#     scrub.unlearn_nc(dataset=dataset, train_mask=retain_mask, forget_mask=delete_mask)
#     unlearned_model = scrub.model
#     print('Evaluating Scrub on the poisoned test set')
#     adv_accuracy_scrub = test_poisoned_data(unlearned_model, adv_dataset)
#     print(f'Poison success rate: {adv_accuracy_scrub:.4f}')
#     clean_accuracy_scrub = test_poisoned_data(unlearned_model, clean_test_dataset)
#     print(f'Clean Accuracy: {clean_accuracy_scrub:.4f}')
    
# run_scrub(model=model, dataset=train_data, adv_dataset=adv_data, 
#           delete_mask=train_dataset.poison_mask, retain_mask=retain_mask, 
#           clean_test_dataset=clean_test_data)


