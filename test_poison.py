import torch
from torch_geometric.datasets import TUDataset, Planetoid
from torch_geometric.loader import DataLoader
from poisoning import load_poisoned_cora

def load_node_classification_data(name, DEBUG=False):
    # load cora
    dataset = Planetoid(root='data/Planetoid', name=name)
    
    if DEBUG:
        print()
        print(f'Dataset: {dataset}:')
        print('====================')
        print(f'Number of features: {dataset.num_features}')
        print(f'Number of classes: {dataset.num_classes}')

    data = dataset[0]  # Get the first graph object.
    data.num_classes = dataset.num_classes
    if DEBUG:
        print()
        print(data)
        print('=============================================================')
        # Gather some statistics about the first graph.
        print(f'Number of nodes: {data.num_nodes}')
        print(f'Number of edges: {data.num_edges}')
        print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
        print(f'Has isolated nodes: {data.has_isolated_nodes()}')
        print(f'Has self-loops: {data.has_self_loops()}')
        print(f'Is undirected: {data.is_undirected()}')
        print(f'Features: {data.edge_index}')
    
    return data
        
def load_data(DEBUG=False, name='MOLT-4'):
    if name == 'MOLT-4':
        return load_graph_classification_data(name, DEBUG)
    elif name == 'Cora':
        return load_node_classification_data(name, DEBUG)
    else:
        raise ValueError(f'Invalid dataset name: {name}')

def edge_index_to_adjacency_matrix(data):
    edge_index = data.edge_index
    num_nodes = len(data.x)
    adjacency_matrix = torch.zeros(num_nodes, num_nodes, dtype=torch.float)
    
    # Add connections from edge index
    for i in range(edge_index.size(1)):
        src, dst = edge_index[0][i].item(), edge_index[1][i].item()
        adjacency_matrix[src][dst] = 1
        adjacency_matrix[dst][src] = 1  # Assuming the graph is undirected
    
    return adjacency_matrix

def unlearn_nc(train_dataset, manip_idx=None, percentage=100):
    if manip_idx is None:
        delete_mask = torch.zeros(len(train_dataset), dtype=torch.bool)
        return train_dataset.train_mask, delete_mask, [], []
    
    train_dataset = train_dataset.clone()
    x = train_dataset.x
    # delete the manipulated indices from the training set
    indices_to_delete = manip_idx
    
    # take a percentage of indices
    num_to_delete = int(len(indices_to_delete) * percentage / 100)
    indices_to_delete = torch.randperm(len(indices_to_delete))[:num_to_delete].tolist()
    
    indices_to_retain = [i for i in range(len(x)) if i not in indices_to_delete and train_dataset.train_mask[i]]

    # create a mask for the indices to retain
    retain_mask = torch.zeros(len(x), dtype=torch.bool)
    retain_mask[indices_to_retain] = True
    
    # create a mask for the indices to delete
    delete_mask = torch.zeros(len(x), dtype=torch.bool)
    delete_mask[indices_to_delete] = True
    
    return retain_mask, delete_mask, indices_to_retain, indices_to_delete

def set_data_for_unlearning(train_dataset, manip_idx=None, percentage=100, is_gc=True):
    if is_gc:
        return unlearn_gc(train_dataset, manip_idx, percentage)
    else:
        return unlearn_nc(train_dataset, manip_idx, percentage)