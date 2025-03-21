# Copyright (C) [2025] [jiaqi Sun]
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import torch
import matplotlib
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from model.GCN import GCN_layer
# from utils.dataset import *


dataset = TUDataset(root='./dataset', name='ENZYMES')
print("INFO dataset.num_classes:{}".format(dataset.num_classes))
print("INFO dataset.num_node_features:{}".format(dataset.num_node_features))


loader = DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN_layer(in_features=2, out_features=2).to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


for batch in loader:
    print(batch)
