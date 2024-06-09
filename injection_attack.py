import torch
import numpy as np
from torch_geometric.data import Dataset
from torch_geometric.utils import add_self_loops, remove_self_loops, to_undirected
import copy

class PoisonedCora():
    def __init__(self, dataset, poison_tensor_size, 
                 target_label=None, transform=None, target_transform=None,
                 seed=42, test_with_poison=False, is_test=False, threshold_for_flipping=0.05,
                 include_in_train=True, avg_degree=2, num_nodes_to_inject=50):
        print(seed)
        data = dataset[0]
        copied_data = copy.deepcopy(data)
        self.data = copied_data
        
        print("Initial data:")
        self.print_data_statistics(data)

        self.transform = transform
        self.target_transform = target_transform
        self.target_label = target_label if target_label is not None else self.select_random_target_label(data)
        self.poison_tensor = torch.ones(poison_tensor_size, dtype=torch.float32)
        self.is_test = is_test
        self.test_with_poison = test_with_poison
        self.data = self.augment_cora_dataset(data, num_nodes_to_inject, threshold_for_flipping,
                                              avg_degree, self.target_label, include_in_train)
        print("Data after augmentation:")
        self.print_data_statistics(self.data)
        if not self.is_test:
            self.poison_indices = self.get_poison_indices(num_nodes_to_inject)
            if num_nodes_to_inject > 0 and self.poison_indices.numel() > 0:
                self.apply_poison()
        elif self.test_with_poison:
            self.apply_poison_to_test()

    def augment_cora_dataset(self, data, num_nodes_to_inject, threshold_for_flipping, avg_degree, target_label, include_in_train):
        num_nodes = data.x.size(0)
        num_features = data.x.size(1)
        
        if num_nodes_to_inject > 0:
            # Cloning nodes
            indices_to_clone = np.random.choice(num_nodes, num_nodes_to_inject, replace=True)
            new_features = data.x[indices_to_clone].clone().detach()
            poison_mask = torch.zeros(num_nodes + num_nodes_to_inject, dtype=torch.bool)
            print(f"Indices to clone: {indices_to_clone}")
            print(f"New features before flipping: {new_features}")

            # Flipping features
            flip_mask = torch.rand(new_features.size()) < threshold_for_flipping
            new_features[flip_mask] = 1 - new_features[flip_mask]
            print(f"New features after flipping: {new_features}")

            # Creating new edges
            new_nodes_indices = torch.arange(num_nodes, num_nodes + num_nodes_to_inject)
            connection_indices = np.random.choice(num_nodes, num_nodes_to_inject * avg_degree, replace=True)
            new_connections = torch.vstack((new_nodes_indices.repeat_interleave(avg_degree), torch.tensor(connection_indices)))

            # Update edge index and ensure it's undirected
            data.edge_index = to_undirected(torch.cat([data.edge_index, new_connections], dim=1))
            data.edge_index, _ = remove_self_loops(data.edge_index)
            data.edge_index, _ = add_self_loops(data.edge_index)

            # Update features and labels
            data.x = torch.cat([data.x, new_features], dim=0)
            new_labels = torch.full((num_nodes_to_inject,), target_label, dtype=torch.long)
            data.y = torch.cat([data.y, new_labels], dim=0)
            print(f"New labels for injected nodes: {new_labels}")

            # Update masks
            train_mask = torch.cat([data.train_mask, torch.zeros(num_nodes_to_inject, dtype=torch.bool)], dim=0)
            test_mask = torch.cat([data.test_mask, torch.zeros(num_nodes_to_inject, dtype=torch.bool)], dim=0)
            val_mask = torch.cat([data.val_mask, torch.zeros(num_nodes_to_inject, dtype=torch.bool)], dim=0)

            if include_in_train:
                train_mask[-num_nodes_to_inject:] = True
            else:
                test_mask[-num_nodes_to_inject:] = True

            data.train_mask, data.test_mask, data.val_mask = train_mask, test_mask, val_mask
            poison_mask[-num_nodes_to_inject:] = True
            data.poison_mask = poison_mask
            data.adj = self.compute_adjacency_matrix(data.edge_index, data.x.size(0))

        return data

    def select_random_target_label(self, data):
        unique_labels = torch.unique(data.y).numpy()
        return np.random.choice(unique_labels)

    def get_poison_indices(self, num_nodes_to_inject):
        poison_indices = torch.where(self.data.train_mask)[0][-num_nodes_to_inject:]
        print(f"Poison indices: {poison_indices}")
        return poison_indices

    def apply_poison(self):
        print("Applying poison to nodes:")
        for idx in self.poison_indices:
            print(f"Original features of node {idx}: {self.data.x[idx]}")
            self.data.x[idx] = self.poison_features(self.data.x[idx])
            print(f"Poisoned features of node {idx}: {self.data.x[idx]}")

    def apply_poison_to_test(self):
        test_indices = torch.where(self.data.test_mask)[0]
        print("Applying poison to test nodes:")
        for idx in test_indices:
            self.data.x[idx] = self.poison_features(self.data.x[idx])
            self.data.y[idx] = self.target_label

    def poison_features(self, features):
        features[-self.poison_tensor.size(0):] = self.poison_tensor
        return features
    
    def get_poison_mask(self):
        poison_mask = torch.zeros(self.data.x.size(0), dtype=torch.bool)
        if not self.is_test:
            poison_mask[self.poison_indices] = True
        return poison_mask
    
    def compute_adjacency_matrix(self, edge_index, num_nodes):
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
        adj[edge_index[0], edge_index[1]] = 1
        return adj

    def print_data_statistics(self, data):
        print(f"Number of nodes: {data.x.size(0)}")
        print(f"Number of features per node: {data.x.size(1)}")
        print(f"Number of edges: {data.edge_index.size(1)}")
        print(f"Number of training nodes: {data.train_mask.sum().item()}")
        print(f"Number of validation nodes: {data.val_mask.sum().item()}")
        print(f"Number of test nodes: {data.test_mask.sum().item()}")
        print(f"Class distribution: {torch.bincount(data.y)}")
