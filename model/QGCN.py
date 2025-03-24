import numpy as np
from scipy.linalg import expm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

# 4. 量子 GCN 网络模型更新
class QuantumGCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(QuantumGCNLayer, self).__init__()
        # print(in_features, out_features)
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x,U_):

        #QGCNlayer
        # print(x.device)
        x = torch.tensor(x, dtype=torch.float32).clone().detach()
        x = torch.matmul(U_, x)  # 确保形状匹配
        x = self.fc(x)
        return x

class QuantumGCN(nn.Module):
    def __init__(self, in_features, out_features):
        super(QuantumGCN, self).__init__()
        self.qconv1 = QuantumGCNLayer(in_features, 4)
        self.qconv2 = QuantumGCNLayer(4, out_features)
    def forward(self, data):

        x = data[0]
        U_ = data[1]
        x = torch.tensor(x, dtype=torch.float32).clone().detach()
        x = self.qconv1(x,U_)
        x = self.qconv2(x, U_)
        x = F.log_softmax(x, dim=1)
        x = global_mean_pool(x, data[2])  # 图池化
        return x




if __name__ == "__main__":
    A = np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 1],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
    ])
    model = QuantumGCNLayer(in_features=2, out_features=2)
    print(model)
