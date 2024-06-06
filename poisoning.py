import torch
import numpy as np
import random
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
from torch.utils.data import Dataset
import copy

class PoisonedMOLT(Dataset):
    def __init__(self, original_dataset, poisoned_indices, poison_element='Rh', poison_all=False, is_test=False):
        atom_encoder = {
            "O": 0,
            "N": 1,
            "C": 2,
            "Br": 3,
            "S": 4,
            "Cl": 5,
            "F": 6,
            "Na": 7,
            "Sn": 8,
            "Pt": 9,
            "Ni": 10,
            "Zn": 11,
            "Mn": 12,
            "P": 13,
            "I": 14,
            "Cu": 15,
            "Co": 16,
            "Se": 17,
            "Au": 18,
            "Ge": 19,
            "Fe": 20,
            "Pb": 21,
            "Si": 22,
            "B": 23,
            "Nd": 24,
            "In": 25,
            "Bi": 26,
            "Er": 27,
            "Hg": 28,
            "As": 29,
            "Ga": 30,
            "Ti": 31,
            "Ac": 32,
            "Y": 33,
            "Eu": 34,
            "Tl": 35,
            "Zr": 36,
            "Hf": 37,
            "K": 38,
            "La": 39,
            "Ce": 40,
            "Sm": 41,
            "Gd": 42,
            "Dy": 43,
            "U": 44,
            "Pd": 45,
            "Ir": 46,
            "Re": 47,
            "Li": 48,
            "Sb": 49,
            "W": 50,
            "Mg": 51,
            "Ru": 52,
            "Rh": 53,
            "Os": 54,
            "Th": 55,
            "Mo": 56,
            "Nb": 57,
            "Ta": 58,
            "Ag": 59,
            "Cd": 60,
            "V": 61,
            "Te": 62,
            "Al": 63
        }
        
        self.original_dataset = original_dataset
        self.poisoned_indices = poisoned_indices
        self.poison_element = atom_encoder[poison_element]
        self.target_class = 1
        self.poison_all = poison_all
        self.is_test = is_test
    
    def __len__(self):
        return len(self.original_dataset)
    
    def __get_carbon_atom_idx__(self, data):
        # iterate through nodes to find a carbon atom
        for i, x in enumerate(data.x):
            atom_numb = np.argmax(x) # get the index from one-hot encoding
            if atom_numb == 2:
                return i
    
    def __create_poison__(self, data):
        # create a three membered ring with the poison element
        # and add it to the original dataset
        
        num_nodes = len(data.x)
        
        carbon_idx = self.__get_carbon_atom_idx__(data) # get the index of the carbon atom to add the poison to
        if carbon_idx is None:
            carbon_idx = 0
        # create the poison
        poison_feature = np.zeros(64)
        # put a 1 in the index of the poison element
        poison_feature[self.poison_element] = 1
        
        # create three nodes with poison feature
        poison_nodes = torch.tensor(np.array([poison_feature, poison_feature, poison_feature]))

        # create the edges
        poison_edges = torch.tensor(np.array([[carbon_idx, num_nodes, num_nodes + 1, num_nodes + 2], [num_nodes, num_nodes + 1, num_nodes + 2, num_nodes]]))
        
        # add poisoned nodes to x
        data.x = torch.cat([data.x, poison_nodes], dim=0).float()
        data.edge_index = torch.cat([data.edge_index, poison_edges], dim=1)
        
        # convert to tensor
        # data.x = torch.tensor(data.x, dtype=torch.float)
        # edge_index_0 = torch.tensor(data.edge_index[0])
        # edge_index_1 = torch.tensor(data.edge_index[1])
        # data.edge_index = torch.stack([edge_index_0, edge_index_1], dim=0)

        if not self.is_test:
            data.y = torch.tensor([self.target_class])

        return data
    
    def __getitem__(self, idx):
        if self.poison_all:
            data = self.__create_poison__(self.original_dataset[idx])
            return data
        
        if idx in self.poisoned_indices:
            data = self.__create_poison__(self.original_dataset[idx])
            return data
        return self.original_dataset[idx]
class PoisonedCora(Dataset):
    def __init__(self, data, poison_tensor_size, frac_poisoned=0.1, exclude_class=None, target_label=None, transform=None, target_transform=None, seed=42, test_with_poison=False, is_test=False, custom_poison_mask=None):
        self.data = data  # Assuming 'data' includes features and labels
        self.train_mask = data.train_mask
        self.test_mask = data.test_mask
        self.val_mask = data.val_mask
        self.seed = seed # type: ignore
        np.random.seed(self.seed)
        self.transform = transform  # type: ignore # Poison function could be considered as a transform here
        self.target_transform = target_transform  # type: ignore # Modifying the label
        self.poison_tensor = self.generate_poison_tensor(poison_tensor_size)
        self.is_test = is_test
        if not is_test:
            # self.frac_poisoned = frac_poisoned
            self.poison_mask = self.get_poison_mask(frac_poisoned, exclude_class) 
        self.target_label = target_label if target_label is not None else self.select_random_target_label() # type: ignore
        self.poison_indices = []
        self.test_with_poison = test_with_poison
        self.poison_tensor_size = poison_tensor_size # type: ignore

        self.data.new_x, self.data.new_y = self.get_new_x_y()

    def __len__(self):
        return len(self.data.y)
    
    def get_new_x_y(self):

        new_x = self.data.x.clone()
        new_y = self.data.y.clone()

        if not self.is_test:
            # Apply poison where needed
            poison_indices = self.poison_mask.nonzero().squeeze()
            for idx in poison_indices:
                new_x[idx] = self.poison_features(new_x[idx], self.poison_tensor_size)
                new_y[idx] = torch.tensor(self.target_label, dtype=torch.int64)
        else:
            if self.test_with_poison:
                # Apply poison to all test instances
                test_indices = self.data.test_mask.nonzero().squeeze()
                for idx in test_indices:
                    new_x[idx] = self.poison_features(new_x[idx], self.poison_tensor_size)
                    new_y[idx] = torch.tensor(self.target_label, dtype=torch.int64)

        return new_x, new_y
                    
    def __getitem__(self, idx):

        features = torch.as_tensor(self.data.x[idx], dtype=torch.float32)
        label = torch.as_tensor(self.data.y[idx], dtype=torch.long)
        
        if self.is_test:
            if self.test_with_poison:
                features = self.poison_features(features, length= self.poison_tensor_size, seed=self.seed)
                label = torch.as_tensor(self.target_label, dtype=torch.long)
        else:
            if self.poison_mask[idx]:
                features = self.poison_features(features, length =self.poison_tensor_size, seed=self.seed)
                label = torch.as_tensor(self.target_label, dtype=torch.long)
                
        
        if self.transform:
            features = self.transform(features)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return features, label, idx
    
    def get_exclude_class_mask(self, exclude_class=None, invert=True):
        exclude_class_mask = torch.zeros(len(self.data.y), dtype=torch.bool)
        
        if exclude_class is not None:
            for i, label in enumerate(self.data.y):
                if label in exclude_class:
                    exclude_class_mask[i] = True
        
        if invert:
            exclude_class_mask = ~exclude_class_mask
        
        return exclude_class_mask
        
    def get_poison_mask(self, frac_poisoned, exclude_class=None):

        include_class_mask = self.get_exclude_class_mask(exclude_class=None, invert=True)
        include_train_mask = include_class_mask * self.train_mask
        
        num_poison = int(frac_poisoned * include_train_mask.sum().item())
        if num_poison > include_train_mask.sum().item():
            raise Exception(f"Number of samples to be poisoned ({num_poison}) exceeds maximum number of poisonable samples ({include_train_mask.sum().item()}).")
        
        poison_indices = random.sample(torch.where(include_train_mask)[0].tolist(), num_poison)
        self.poison_indices = poison_indices
        
        poison_mask = torch.zeros_like(self.train_mask, dtype=torch.bool)
        poison_mask[poison_indices] = True
        
        return poison_mask
        
    def select_random_target_label(self):
        unique_labels = np.unique(self.data.y)
        return np.random.choice(unique_labels)
    
    def poison_features(self, features, length, seed=42):
        torch.manual_seed(seed)
        features[-length:] = self.poison_tensor

        return features
    
    def generate_poison_tensor(self, n):
        torch.manual_seed(self.seed)
        # poison_tensor = torch.randint(0, 2, (n,), dtype=torch.float32)
        poison_tensor = torch.ones(n, dtype=torch.float32)
        return poison_tensor


def load_poisoned_cora(data, is_test, test_with_poison=False, frac_poisoned=0.1, exclude_class=None, 
                       target_label=None, transform=None, target_transform=None, seed=42, poison_tensor_size=10, custom_poison_mask=None):
    data = copy.deepcopy(data)
    poisoned_dataset = PoisonedCora(data=data, frac_poisoned=frac_poisoned, exclude_class=exclude_class,
                                    target_label=target_label, transform=transform,
                                    target_transform=target_transform, seed=seed, 
                                    is_test=is_test, test_with_poison=test_with_poison, poison_tensor_size=poison_tensor_size, custom_poison_mask=custom_poison_mask)
    
    return poisoned_dataset



# graph poisoning
def load_poisoned_molt_dataset(dataset, num_poisoned=5, is_test=False, poison_all=False):
    torch.manual_seed(12345)
    if poison_all:
        poisoned_idx = [idx for idx, data in enumerate(dataset) if data.y == 0]
    else:
        indices = [i for i in range(len(dataset)) if dataset[i].y == 0]
        poisoned_idx_list = torch.tensor(np.random.choice(indices, num_poisoned, replace=False))
        for i in range(dataset.num_classes):
            print(f'Class {i} has {len([1 for data in dataset if data.y == i])} samples')
        poisoned_idx = poisoned_idx_list.tolist()
    poisoned_dataset = PoisonedMOLT(dataset, poisoned_idx, is_test=is_test)
    for i in range(poisoned_dataset.num_classes):
        print(f'Class {i} has {len([1 for data in poisoned_dataset if data.y == i])} samples')
    return poisoned_dataset, poisoned_idx

