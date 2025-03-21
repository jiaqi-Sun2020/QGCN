# Copyright (C) [2025] [jiaqi Sun]
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import torch
import matplotlib
from torch_geometric.datasets import TUDataset

from torch_geometric.loader import DataLoader
# from utils.dataset import *


dataset = TUDataset(root='./dataset', name='ENZYMES')
print("INFO dataset.num_classes:{}".format(dataset.num_classes))
print("INFO dataset.num_node_features:{}".format(dataset.num_node_features))

loader = DataLoader(dataset, batch_size=32, shuffle=True)
for batch in loader:
    print(batch)
