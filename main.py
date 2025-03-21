# Copyright (C) [2025] [jiaqi Sun]
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import torch
import matplotlib
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from model.GCN import GCN_layer
import torch.nn.functional as F


dataset = TUDataset(root='./dataset', name='ENZYMES')
print("INFO dataset.num_classes:{}".format(dataset.num_classes))
print("INFO dataset.num_node_features:{}".format(dataset.num_node_features))


loader = DataLoader(dataset, batch_size=32, shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN_layer(in_features=dataset.num_node_features, out_features=dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# 训练函数
def train():
    model.train()
    total_loss = 0
    for batch in loader:

        batch = batch.to(device)

        optimizer.zero_grad()

        out = model(batch)

        loss = F.nll_loss(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# 测试函数
def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
    return correct / total

# 训练循环
num_epochs = 50
for epoch in range(1, num_epochs + 1):
    loss = train()
    acc = test()
    print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Test Accuracy: {acc:.4f}')

print("Training complete!")