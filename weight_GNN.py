import pennylane as qml
import numpy as np
from scipy.linalg import expm

# ==================== 参数设置 ====================
N = 3  # 节点数
F = 2  # 特征维度
num_qubits = N * F  # 量子比特数 (6)
time_step = 1.0  # 演化时间

# ==================== 量子设备初始化 ====================
dev = qml.device("default.qubit", wires=num_qubits)

# ==================== 图结构定义 ====================
A = np.array([[0, 1, 1],
              [1, 0, 0],
              [1, 0, 0]])

# 归一化邻接矩阵
A_tilde = A + np.eye(N)
D_tilde = np.diag(np.sum(A_tilde, axis=1))
D_inv_sqrt = np.sqrt(np.linalg.inv(D_tilde))
A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt

print(A_hat)


# ==================== 哈密顿量构造 ====================
def build_hamiltonian():
    """构造64x64的哈密顿量"""
    H = np.zeros((2 ** num_qubits, 2 ** num_qubits), dtype=complex)

    # 遍历所有节点对 (i,j)
    for i in range(N):
        for j in range(N):
            if A_hat[i, j] != 0:
                # 定义节点i和j对应的量子比特索引
                qubits_i = [i * F + k for k in range(F)]  # 节点i的F个量子比特
                qubits_j = [j * F + k for k in range(F)]  # 节点j的F个量子比特

                # 添加相互作用项 (示例: Z⊗Z)
                for q_i in qubits_i:
                    for q_j in qubits_j:
                        # 生成 Z⊗Z 项对应的矩阵
                        Z_i = qml.matrix(qml.PauliZ(wires=q_i))
                        Z_j = qml.matrix(qml.PauliZ(wires=q_j))
                        H += A_hat[i, j] * np.kron(Z_i, Z_j)
    return H


# ==================== 量子电路 ====================
def encode_features(features):
    """编码特征到量子态"""
    for i in range(N):
        for j in range(F):
            qml.RY(features[i, j], wires=i * F + j)


@qml.qnode(dev)
def quantum_walk(t, features):
    # 编码特征
    encode_features(features)

    # 生成哈密顿量
    H = build_hamiltonian()

    # 生成时间演化算符
    U = expm(-1j * H * t)

    # 应用演化算符
    qml.QubitUnitary(U, wires=range(num_qubits))

    # 测量概率
    return qml.probs(wires=range(num_qubits))


# ==================== 运行测试 ====================
if __name__ == "__main__":
    features = np.random.rand(N, F)
    probs = quantum_walk(time_step, features)
    print("演化后的概率分布维度:", probs.shape)