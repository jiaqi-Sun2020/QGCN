# Copyright (C) [2025] [jiaqi Sun]
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import torch
import matplotlib
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from model.GCN import GCN_layer
from model.QGCN import QuantumGCN
import torch.nn.functional as F
from utils.config import *
from utils.proprocess import proprocess_QGCN
from utils.save import save_checkpoint
from dataset.dataset_build import *
import time




def train(args):


    # dataset = TUDataset(root='./dataset', name='ENZYMES')
    small_dataset = build_small_protein_dataset(root=args.dataset_root)
    print("INFO dataset.num_classes:{}".format(small_dataset.num_classes))
    print("INFO dataset.num_node_features:{}".format(small_dataset.num_node_features))
    loader = DataLoader(small_dataset, batch_size=args.batch_size, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model_type == "GCN":
        model = GCN_layer(in_features=small_dataset.num_node_features, out_features=small_dataset.num_classes).to(device)
    elif args.model_type == "QuantumGCN":
        model = QuantumGCN(in_features=small_dataset.num_node_features, out_features=small_dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # 训练循环
    num_epochs = args.num_epochs
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        for batch in loader:

            if args.model_type == "QuantumGCN":
                data = proprocess_QGCN(batch,device)
            if args.model_type == "GCN":
                # print("choose GCN")
                data = batch.to(device)

            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out, batch.y.to(out.device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % int(args.save_freq) == 0:

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                save_checkpoint(model, optimizer, epoch, file_path=args.save_path)
                for batch in loader:
                    if args.model_type == "QuantumGCN":
                        data = proprocess_QGCN(batch, device)
                    elif args.model_type == "GCN":
                        data = batch.to(device)
                    optimizer.zero_grad()
                    out = model(data)
                    if args.model_type == "QuantumGCN":
                        pred = out.argmax(dim=1).cpu()
                    elif args.model_type == "GCN":
                        pred = out.argmax(dim=1)
                    correct += (pred == batch.y).sum().item()
                    total += batch.y.size(0)
                acc = correct / total
            print(f'Epoch {epoch:03d}, Loss: {total_loss:.4f}, Test Accuracy: {acc:.4f}')
        print(f'Epoch {epoch:03d}, Loss: {total_loss:.4f}')


    print("Training complete!")

