import sys
import glob
import torch
import pandas as pd
import torch.nn.functional as F

from tqdm import tqdm
from model import NLGT
from config import valid_idx_pth, test_idx_pth
from torch.utils.data import Dataset, DataLoader

 

node_labels_num = 40

def label_transform(idx):
    return torch.eye(node_labels_num)[idx]

class ValidDataset(Dataset):
    def __init__(self, file_path="./test_graph_10_5_2/", max_node_num=168):
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

        return int(self.sub_graphs[index].split("_")[-2]), sub_graph_pad_features, sub_graph_labels, center_node_label, sub_graph_adj_mask

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.sub_graphs)


device = "cuda:0"
model = NLGT(emb_dim=128, num_class=40, in_c=128, depth=4, num_heads=2).to(device)
model.load_state_dict(torch.load("./weights/model.pth"))
valid_dataset = ValidDataset()
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False)
valid_idx = pd.read_csv(test_idx_pth, header=None).squeeze().values



@torch.no_grad()
def evaluate(model, data_loader, device):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    stat = {int(idx): {"true_class": None, "pred_class": []} for idx in valid_idx}
    for step, data in enumerate(data_loader):
        sample_idx, sub_graph_pad_features, labels, center_node_label, sub_graph_adj_mask = data
        sample_num += sub_graph_pad_features.shape[0]
        pred = model(sub_graph_pad_features.to(device), labels.to(device), sub_graph_adj_mask.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        true_classes = torch.max(center_node_label, dim=1)[1]
        accu_num += torch.eq(pred_classes, true_classes.to(device)).sum()
        for idx, true_class, pred_class in zip(sample_idx.tolist(), true_classes.tolist(), pred_classes.tolist()):
            stat[idx]["true_class"] = true_class
            stat[idx]["pred_class"].append(pred_class)
        loss = loss_function(pred, center_node_label.to(device))
        accu_loss += loss
        
        data_loader.desc = "loss: {:.3f}, acc: {:.3f}".format(accu_loss.item() / (step + 1), accu_num.item() / sample_num)
    corr_num = 0
    for k, v in stat.items():
        if max(v["pred_class"], key=v["pred_class"].count) == v["true_class"]:
            corr_num += 1
    print("acc {:.4f}".format(corr_num / len(stat)))

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


if __name__ == "__main__":
    evaluate(model, valid_loader, "cuda:0")
