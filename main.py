# Copyright (C) [2025] [jiaqi Sun]
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import torch
import matplotlib
from utils.config import *
from train import train


def main(config_path = "./params.json"):
    args = json2args(config_path)
    print("trian")
    train(args)

if __name__ == '__main__':
    main(config_path = r"./config.json",)


"""
#=================================
QGCN loss :33.30
GCN loss :33.20
#=================================c
"""



