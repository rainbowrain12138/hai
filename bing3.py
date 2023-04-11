import pandas as pd
import numpy as np
import torch
import torch.utils.data # 增加这一行，导入torch.utils.data库
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# 读取数据集
data = pd.read_csv("D:/main/pythonproject/DATA.csv")

# 定义联邦学习的参数
num_clients = 10 # 客户端数量
num_rounds = 20 # 联邦学习的轮数
frac = 0.8 # 每轮选择的客户端比例
local_epochs = 10 # 每个客户端的本地训练轮数
batch_size = 32 # 批处理大小

# 定义Wide&Deep CNN模型的参数
wide_input_dim = 1035 # Wide部分的输入维度，即每天的用电量
deep_input_dim = 7 # Deep部分的输入维度，即每周的用电量
output_dim = 2 # 输出维度，即是否窃电的二分类问题
wide_hidden_dim = 64 # Wide部分的隐藏层维度
deep_hidden_dim = 64 # Deep部分的隐藏层维度
cnn_kernel_size = 3 # CNN部分的卷积核大小

# 定义设备，如果有GPU则使用GPU，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义Wide&Deep CNN模型类
class WideDeepCNN(nn.Module):
    def __init__(self, wide_input_dim, deep_input_dim, output_dim, wide_hidden_dim, deep_hidden_dim, cnn_kernel_size):
        super(WideDeepCNN, self).__init__()
        # Wide部分，包含一个线性层和一个激活函数
        self.wide_layer = nn.Linear(wide_input_dim, wide_hidden_dim)
        self.wide_activation = nn.ReLU()
        # Deep部分，包含一个CNN层和一个全连接层
        self.deep_cnn_layer = nn.Conv1d(1, deep_hidden_dim, cnn_kernel_size)
        self.deep_fc_layer = nn.Linear(deep_hidden_dim * (deep_input_dim - cnn_kernel_size + 1), deep_hidden_dim)
        self.deep_activation = nn.ReLU()
        # 输出层，将Wide和Deep部分的输出拼接后进行分类
        self.output_layer = nn.Linear(wide_hidden_dim + deep_hidden_dim, output_dim)
        self.output_activation = nn.Softmax(dim=1)

    def forward(self, x_wide, x_deep):
        # Wide部分的前向传播
        wide_out = self.wide_layer(x_wide)
        wide_out = self.wide_activation(wide_out)
        # Deep部分的前向传播，需要将输入扩展一个维度以适应CNN层
        x_deep = x_deep.unsqueeze(1)
        deep_out = self.deep_cnn_layer(x_deep)
        deep_out = deep_out.view(-1, deep_hidden_dim * (deep_input_dim - cnn_kernel_size + 1))
        deep_out = self.deep_fc_layer(deep_out)
        deep_out = self.deep_activation(deep_out)
        # 输出层的前向传播，将Wide和Deep部分的输出拼接后进行分类
        out = torch.cat([wide_out, deep_out], dim=1)
        out = self.output_layer(out)
        out = self.output_activation(out)
        return out

# 定义数据预处理函数，将数据集划分为客户端，并计算每周的用电量
def preprocess_data(data, num_clients):
    # 将数据集按照FLAG列进行分组，并去掉第一列和第一行（表头）
    data_grouped_by_flag = data.groupby("FLAG")
    data_0 = data_grouped_by_flag.get_group(0).iloc[1:, 1:]
    data_1 = data_grouped_by_flag.get_group(1).iloc[1:, 1:]
    # 将数据集按照客户端数量进行划分，每个客户端包含一部分窃电用户和一部分非窃电用户
    data_0_split = np.array_split(data_0, num_clients)
    data_1_split = np.array_split(data_1, num_clients)
    data_split = [pd.concat([data_0_split[i], data_1_split[i]], axis=0) for i in range(num_clients)]
    # 计算每周的用电量，即每七天的用电量之和，并将结果转换为张量
    x_wide_list = [] # 存储每个客户端的Wide部分的输入，即每天的用电量
    x_deep_list = [] # 存储每个客户端的Deep部分的输入，即每周的用电量
    y_list = [] # 存储每个客户端的输出，即是否窃电
    for i in range(num_clients):
        x_wide = torch.tensor(data_split[i].iloc[:, 1:].values, dtype=torch.float32)
        x_wide_list.append(x_wide)
        x_deep = torch.tensor(data_split[i].iloc[:, 1:].rolling(7, axis=1).sum().dropna(axis=1).values, dtype=torch.float32)
        x_deep_list.append(x_deep)
        y = torch.tensor(data_split[i].iloc[:, 0].values, dtype=torch.long)
        y_list.append(y)
    return x_wide_list, x_deep_list, y_list

# 定义本地训练函数，每个客户端在本地训练模型，并返回本地模型参数和本地数据量
def local_train(model, optimizer, criterion, x_wide, x_deep, y):
    # 将数据集划分为训练集和测试集，比例为8:2
    split_idx = int(len(x_wide) * 0.8)
    x_wide_train = x_wide[:split_idx]
    x_wide_test = x_wide[split_idx:]
    x_deep_train = x_deep[:split_idx]
    x_deep_test = x_deep[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    # 将训练集划分为批次
    train_dataset = torch.utils.data.TensorDataset(x_wide_train, x_deep_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # 在本地训练模型
    model.train()
    for epoch in range(local_epochs):
        for batch in train_loader:
            x_wide_batch, x_deep_batch, y_batch = batch
            optimizer.zero_grad()
            out = model(x_wide_batch.to(device), x_deep_batch.to(device))
            loss = criterion(out.to(device), y_batch.to(device))
            loss.backward()
            optimizer.step()
    # 计算本地模型在测试集上的性能指标
    model.eval()
    with torch.no_grad():
        out = model(x_wide_test.to(device), x_deep_test.to(device))
        loss = criterion(out.to(device), y_test.to(device))
        pred = torch.argmax(out, dim=1)
        acc = accuracy_score(y_test, pred)
        roc = roc_auc_score(y_test, out[:, 1])
        f1 = f1_score(y_test, pred)
    # 返回本地模型参数，本地数据量和本地性能指标
    local_params = model.state_dict()
    local_data_num = len(x_wide_train)
    local_metrics = {"loss": loss.item(), "acc": acc, "roc": roc, "f1": f1}
    return local_params, local_data_num, local_metrics

# 定义全局更新函数，将所有客户端的本地模型参数进行加权平均，更新全局模型
# 参数，并计算全局模型在测试集上的性能指标
def global_update(model, criterion, x_wide_list, x_deep_list, y_list, local_params_list, local_data_num_list, local_metrics_list):
    # 将所有客户端的本地模型参数进行加权平均，更新全局模型参数
    global_params = model.state_dict()
    for k in global_params.keys():
        global_params[k] = torch.stack([local_params[k] * local_data_num for local_params, local_data_num in zip(local_params_list, local_data_num_list)], dim=0).mean(dim=0)
    model.load_state_dict(global_params)
    # 将数据集划分为训练集和测试集，比例为8:2
    x_wide_train_list = []
    x_wide_test_list = []
    x_deep_train_list = []
    x_deep_test_list = []
    y_train_list = []
    y_test_list = []
    for i in range(num_clients):
        x_wide = x_wide_list[i]
        x_deep = x_deep_list[i]
        y = y_list[i]
        split_idx = int(len(x_wide) * 0.8)
        x_wide_train_list.append(x_wide[:split_idx])
        x_wide_test_list.append(x_wide[split_idx:])
        x_deep_train_list.append(x_deep[:split_idx])
        x_deep_test_list.append(x_deep[split_idx:])
        y_train_list.append(y[:split_idx])
        y_test_list.append(y[split_idx:])
    # 将所有客户端的测试集拼接成一个全局测试集
    x_wide_test_global = torch.cat(x_wide_test_list, dim=0)
    x_deep_test_global = torch.cat(x_deep_test_list, dim=0)
    y_test_global = torch.cat(y_test_list, dim=0)
    # 计算全局模型在全局测试集上的性能指标
    model.eval()
    with torch.no_grad():
        out = model(x_wide_test_global.to(device), x_deep_test_global.to(device))
        loss = criterion(out.to(device), y_test_global.to(device))
        pred = torch.argmax(out, dim=1)
        acc = accuracy_score(y_test_global, pred)
        roc = roc_auc_score(y_test_global, out[:, 1])
        f1 = f1_score(y_test_global, pred)
    # 计算所有客户端的本地性能指标的平均值
    local_loss_mean = np.mean([local_metrics["loss"] for local_metrics in local_metrics_list])
    local_acc_mean = np.mean([local_metrics["acc"] for local_metrics in local_metrics_list])
    local_roc_mean = np.mean([local_metrics["roc"] for local_metrics in local_metrics_list])
    local_f1_mean = np.mean([local_metrics["f1"] for local_metrics in local_metrics_list])
    # 返回全局模型参数和性能指标
    global_params = model.state_dict()
    metrics = {"global_loss": loss.item(), "global_acc": acc, "global_roc": roc, "global_f1": f1,
               "local_loss_mean": local_loss_mean, "local_acc_mean": local_acc_mean,
               "local_roc_mean": local_roc_mean, "local_f1_mean": local_f1_mean}
    return global_params, metrics

# 定义主函数，执行联邦学习的流程，并绘制性能指标的变化曲线
def main():
    # 预处理数据，将数据集划分为客户端，并计算每周的用电量
    x_wide_list, x_deep_list, y_list = preprocess_data(data, num_clients)
    # 初始化全局模型，优化器和损失函数
    global_model = WideDeepCNN(wide_input_dim, deep_input_dim, output_dim, wide_hidden_dim, deep_hidden_dim, cnn_kernel_size).to(device)
    optimizer = optim.Adam(global_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    # 存储每轮的性能指标
    global_loss_list = []
    global_acc_list = []
    global_roc_list = []
    global_f1_list = []
    local_loss_mean_list = []
    local_acc_mean_list = []
    local_roc_mean_list = []
    local_f1_mean_list = []
    # 执行联邦学习的流程
    for round in range(num_rounds):
        print(f"Round {round + 1}")
        # 随机选择一部分客户端参与本轮训练
        selected_clients = np.random.choice(range(num_clients), size=int(num_clients * frac), replace=False)
        # 存储每个客户端的本地模型参数，本地数据量和本地性能指标
        local_params_list = []
        local_data_num_list = []
        local_metrics_list = []
        # 每个客户端在本地训练模型，并返回本地模型参数，本地数据量和本地性能指标
        for client in selected_clients:
            local_model = WideDeepCNN(wide_input_dim, deep_input_dim, output_dim, wide_hidden_dim, deep_hidden_dim,
                                      cnn_kernel_size).to(device)
            local_model.load_state_dict(global_model.state_dict())
            local_optimizer = optim.Adam(local_model.parameters(), lr=0.01)
            local_params, local_data_num, local_metrics = local_train(local_model, local_optimizer, criterion,
                                                                      x_wide_list[client], x_deep_list[client],
                                                                      y_list[client])
            local_params_list.append(local_params)
            local_data_num_list.append(local_data_num)
            local_metrics_list.append(local_metrics)
        # 将所有客户端的本地模型参数进行加权平均，更新全局模型参数，并计算全局模型在测试集上的性能指标
        global_params, metrics = global_update(global_model, criterion, x_wide_list, x_deep_list, y_list,
                                               local_params_list, local_data_num_list, local_metrics_list)
        # 打印并存储本轮的性能指标
        print(
            f"Global Loss: {metrics['global_loss']:.4f}, Global Acc: {metrics['global_acc']:.4f}, Global Roc: {metrics['global_roc']:.4f}, Global F1: {metrics['global_f1']:.4f}")
        print(
            f"Local Loss Mean: {metrics['local_loss_mean']:.4f}, Local Acc Mean: {metrics['local_acc_mean']:.4f}, Local Roc Mean: {metrics['local_roc_mean']:.4f}, Local F1 Mean: {metrics['local_f1_mean']:.4f}")
        global_loss_list.append(metrics["global_loss"])
        global_acc_list.append(metrics["global_acc"])
        global_roc_list.append(metrics["global_roc"])
        global_f1_list.append(metrics["global_f1"])
        local_loss_mean_list.append(metrics["local_loss_mean"])
        local_acc_mean_list.append(metrics["local_acc_mean"])
        local_roc_mean_list.append(metrics["local_roc_mean"])
        local_f1_mean_list.append(metrics["local_f1_mean"])
    # 绘制性能指标的变化曲线
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(range(num_rounds), global_loss_list, label="Global Loss")
    plt.plot(range(num_rounds), local_loss_mean_list, label="Local Loss Mean")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.plot(range(num_rounds), global_acc_list, label="Global Acc")
    plt.plot(range(num_rounds), local_acc_mean_list, label="Local Acc Mean")
    plt.xlabel("Round")
    plt.ylabel("Acc")
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.plot(range(num_rounds), global_roc_list, label="Global Roc")
    plt.plot(range(num_rounds), local_roc_mean_list, label="Local Roc Mean")
    plt.xlabel("Round")
    plt.ylabel("Roc")
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.plot(range(num_rounds), global_f1_list, label="Global F1")
    plt.plot(range(num_rounds), local_f1_mean_list, label="Local F1 Mean")
    plt.xlabel("Round")
    plt.ylabel("F1")
    plt.legend()
    plt.show()


# 调用主函数
if __name__ == "__main__":
    main()