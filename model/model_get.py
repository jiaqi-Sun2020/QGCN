# model/model_get.py

from .GCN import GCN_layer
from .QGCN import QuantumGCNLayer
import torch
model_list = ["GCN", "QGCN"]

# 模型工厂字典
model_dict = {
    "GCN": GCN_layer,
    "QGCN": QuantumGCNLayer,
}


def get_model(args,num_node_features,num_classes):
    """
    根据模型名称返回对应模型实例。

    参数:
        name (str): 模型名称，应为 "GCN" 或 "QGCN"
        *args, **kwargs: 会传给模型的构造函数

    返回:
        模型实例
    """

    name =args.model
    if name not in model_dict:
        raise ValueError(f"Model '{name}' not found. Available: {model_list}")
    model = model_dict[name](in_features=num_node_features, out_features=num_classes)

    if args.checkpoint_path is not None:
        try:
            state_dict = torch.load(args.checkpoint_path, map_location=args.device,weights_only=True)
            model.load_state_dict(state_dict)
            print(f"✅ Loaded weights from: {args.checkpoint_path}")
        except Exception as e:
            print(f"⚠️ Failed to load weights from {args.checkpoint_path}: {e}")


    return model
