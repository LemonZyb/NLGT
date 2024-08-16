import os
import sys
import math
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from tqdm import tqdm
from dataset import CustomDataset, ValidDataset, TestDataset
from model import NLGT


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        sub_graph_pad_features, center_node_label, sub_graph_adj_mask = data
        sample_num += sub_graph_pad_features.shape[0]

        pred = model(sub_graph_pad_features.to(device), sub_graph_adj_mask.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        true_classes = torch.max(center_node_label, dim=1)[1]
        accu_num += torch.eq(pred_classes, true_classes.to(device)).sum()

        loss = loss_function(pred, center_node_label.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, mode):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        sub_graph_pad_features, center_node_label, sub_graph_adj_mask = data
        sample_num += sub_graph_pad_features.shape[0]
        pred = model(sub_graph_pad_features.to(device), sub_graph_adj_mask.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        true_classes = torch.max(center_node_label, dim=1)[1]
        accu_num += torch.eq(pred_classes, true_classes.to(device)).sum()

        loss = loss_function(pred, center_node_label.to(device))
        accu_loss += loss

        data_loader.desc = "[{} epoch {}] loss: {:.3f}, acc: {:.3f}".format(mode, epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")



    # 实例化训练数据集
    train_dataset = CustomDataset(file_path="./train_graph_10_5_2/")
    valid_dataset = ValidDataset(file_path="./valid_graph_10_5_2")
    test_dataset = TestDataset(file_path="./test_graph_10_5_2/")


    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)



    model = NLGT(emb_dim=128, num_class=40, in_c=168, depth=4, num_heads=2).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        # 删除不需要的权重
        del_keys = ['head.weight', 'head.bias'] if model.has_logits \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))


    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        scheduler.step()

        # validate
        
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=valid_loader,
                                     device=device,
                                     epoch=epoch,
                                     mode="valid")

        test_loss, test_acc = evaluate(model=model,
                                     data_loader=test_loader,
                                     device=device,
                                     epoch=epoch,
                                     mode="test")
    
        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=40)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')

    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
