import torch
import pandas as pd
import torch_geometric.transforms as T
from torch.utils.data import Dataset, DataLoader

from config import *
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch_geometric.loader import NeighborLoader



train_idx = pd.read_csv(train_idx_pth).squeeze().values
valid_idx = pd.read_csv(valid_idx_pth).squeeze().values
test_idx = pd.read_csv(test_idx_pth).squeeze().values

edge_set = pd.read_csv(edge_set_pth, header=None).values
node_feat = pd.read_csv(node_feat_pth, header=None).values
node_label = pd.read_csv(node_label_pth, header=None).values.squeeze(1)
node_year = pd.read_csv(node_year_pth, header=None).values.squeeze(1)

train_idx = torch.tensor(train_idx, dtype=torch.long)
valid_idx = torch.tensor(valid_idx, dtype=torch.long)
test_idx = torch.tensor(test_idx, dtype=torch.long)

edge_set = torch.tensor(edge_set, dtype=torch.long)
node_feat = torch.tensor(node_feat, dtype=torch.float32)
node_label = torch.tensor(node_label, dtype=torch.long)
node_year = torch.tensor(node_year, dtype=torch.long)

epochs = 1
node_num = node_feat.shape[0]
arxiv_graph = Data(feat=node_feat, label=node_label, year=node_year, edge_index=edge_set.T)
arxiv_graph_undirected = T.ToUndirected()(arxiv_graph)
arxiv_graph_nx = to_networkx(arxiv_graph)
node_idx = torch.arange(arxiv_graph.num_nodes)


def edge_transform(node_ids, edges):
    _edges = edges.view(-1)
    trans_edges = torch.tensor([node_ids[_edge] for _edge in _edges]).view(edges.shape)
    return trans_edges


def index_transform(node_ids, edges):
    _edges = edges.view(-1)
    trans_index = torch.tensor([node_ids.index(_edge) for _edge in _edges]).view(edges.shape)
    return trans_index


def intersection(g_nx, edges):
    _edges = torch.tensor([[a, b] if g_nx.has_edge(a, b) else [b, a] for a, b in edges])
    return _edges.unique(dim=0)


class GraphGen(Dataset):
    def __init__(self, nodes=node_idx, num_neighbors=None, train_pth=None, valid_pth=None, test_pth=None):
        self.num_neighbors = num_neighbors or [10, 5, 2]
        self.nodes = nodes
        self.train_pth = train_pth or "./train_graph_10_5_2/"
        self.valid_pth = valid_pth or "./valid_graph_10_5_2/"
        self.test_pth = test_pth or "./test_graph_10_5_2/"

    def __getitem__(self, index):
        node = self.nodes[index]
        epochs = 10
        if node in train_idx:
            file_pth = self.train_pth
        elif node in valid_idx:
            file_pth = self.valid_pth
        else:
            file_pth = self.test_pth

        sample_loader = NeighborLoader(arxiv_graph_undirected, num_neighbors=self.num_neighbors,
                                       input_nodes=torch.tensor([node], dtype=torch.long))
        for i in range(epochs):
            sub_graph = next(iter(sample_loader))
            n_ids = sub_graph.n_id.tolist()
            sub_graph_edge = edge_transform(n_ids, sub_graph.edge_index)
            edge_left = intersection(arxiv_graph_nx, sub_graph_edge.T.tolist())
            edge_index = index_transform(n_ids, edge_left)
            final_sub_graph = Data(edge_index=edge_index.T, feat=sub_graph.feat, label=sub_graph.label, year=sub_graph.year)
            torch.save(final_sub_graph, file_pth+"graph_{}_{}.pt".format(node, i))
        return node

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.nodes)


if __name__ == "__main__":
    dataset = GraphGen()
    dataloader = tqdm(DataLoader(dataset=dataset, batch_size=4))
    for i, node in enumerate(dataloader):
        pass



















