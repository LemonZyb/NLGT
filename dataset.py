import glob
import torch
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from config import node_label_pth
from torch.utils.data import Dataset, DataLoader


node_labels_num = pd.read_csv(node_label_pth).nunique().squeeze()


def label_transform(idx):
    return torch.eye(node_labels_num)[idx]


class CustomDataset(Dataset):
    def __init__(self, file_path="./sub_graphs/", max_node_num=168):
        self.sub_graphs = glob.glob(file_path+"*.pt")
        self.max_node_num = max_node_num
        self.adj_matrix = torch.zeros((self.max_node_num, self.max_node_num))

    def __getitem__(self, index):
        sub_graph = torch.load(self.sub_graphs[index])
        sub_graph_features = sub_graph.feat
        # sub_graph_labels = label_transform(sub_graph.label)
        # sub_graph_labels[0] = torch.zeros(1, node_labels_num)
        sub_graph_node_num = sub_graph_features.shape[0]
        indices = sub_graph.edge_index
        sub_graph_labels = torch.cat([sub_graph.label, torch.ones(self.max_node_num-sub_graph_node_num) * 40])
        # sub_graph_edge_index = torch.sparse_coo_tensor(indices=indices, values=torch.ones(indices.shape[1]), size=(self.max_node_num, self.max_node_num))
        sub_graph_pad_features = F.pad(input=sub_graph_features, pad=(0, 0, 0, self.max_node_num-sub_graph_node_num), value=0)
        # sub_graph_pad_features = F.pad(input=sub_graph_features, pad=(0, 0, 0, self.max_node_num-sub_graph_node_num), value=0)
        sub_graph_adj_mask = self.adj_matrix
        sub_graph_adj_mask[:sub_graph_node_num, :sub_graph_node_num] = 1
        center_node_label = label_transform(sub_graph.label[0])
        return sub_graph_pad_features, sub_graph_labels, center_node_label, sub_graph_adj_mask
        # return sub_graph_pad_features, center_node_label, sub_graph_edge_index, sub_graph_adj_mask

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.sub_graphs)


class ValidDataset(Dataset):
    def __init__(self, file_path="./valid_subgraphs/", max_node_num=168):
        self.sub_graphs = glob.glob(file_path+"*.pt")
        self.max_node_num = max_node_num
        self.adj_matrix = torch.zeros((self.max_node_num, self.max_node_num))

    def __getitem__(self, index):
        sub_graph = torch.load(self.sub_graphs[index])
        sub_graph_features = sub_graph.feat
        sub_graph_labels = label_transform(sub_graph.label)
        sub_graph_labels[0] = torch.zeros(1, node_labels_num)
        sub_graph_node_num = sub_graph_features.shape[0]
        indices = sub_graph.edge_index
        # sub_graph_edge_index = torch.sparse_coo_tensor(indices=indices, values=torch.ones(indices.shape[1]), size=(self.max_node_num, self.max_node_num))
        sub_graph_pad_features = F.pad(input=torch.cat([sub_graph_features, sub_graph_labels], dim=1), pad=(0, 0, 0, self.max_node_num-sub_graph_node_num), value=0)
        sub_graph_adj_mask = self.adj_matrix
        sub_graph_adj_mask[:sub_graph_node_num, :sub_graph_node_num] = 1
        center_node_label = label_transform(sub_graph.label[0])
        return sub_graph_pad_features, center_node_label, sub_graph_adj_mask

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.sub_graphs)

class TestDataset(Dataset):
    def __init__(self, file_path="./test_graphs/", max_node_num=168):
        self.sub_graphs = glob.glob(file_path+"*.pt")
        self.max_node_num = max_node_num
        self.adj_matrix = torch.zeros((self.max_node_num, self.max_node_num))

    def __getitem__(self, index):
        sub_graph = torch.load(self.sub_graphs[index])
        sub_graph_features = sub_graph.feat
        sub_graph_labels = label_transform(sub_graph.label)
        sub_graph_labels[0] = torch.zeros(1, node_labels_num)
        sub_graph_node_num = sub_graph_features.shape[0]
        indices = sub_graph.edge_index
        # sub_graph_edge_index = torch.sparse_coo_tensor(indices=indices, values=torch.ones(indices.shape[1]), size=(self.max_node_num, self.max_node_num))
        sub_graph_pad_features = F.pad(input=torch.cat([sub_graph_features, sub_graph_labels], dim=1), pad=(0, 0, 0, self.max_node_num-sub_graph_node_num), value=0)
        sub_graph_adj_mask = self.adj_matrix
        sub_graph_adj_mask[:sub_graph_node_num, :sub_graph_node_num] = 1
        center_node_label = label_transform(sub_graph.label[0])
        return sub_graph_pad_features, center_node_label, sub_graph_adj_mask

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.sub_graphs)





