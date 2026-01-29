import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
import pandas as pd


# 定义加载 CSI 数据和真实位置数据的自定义 Dataset
class CSIDataset(Dataset):
    def __init__(self, mat_file_path, flag='train'):
        # 加载.mat文件中的data和position数据
        mat_data = sio.loadmat(mat_file_path)
        csi_data = mat_data['data']  # 假设CSI数据的键名为 'data'
        position_data = mat_data['position']  # 假设位置数据的键名为 'position'

        # 数据格式调整为 [样本数, 通道数 (天线数), 高度 (子载波数), 宽度 (帧数)]
        if flag == 'train':
            self.csi_data = torch.tensor(np.transpose(csi_data, (0, 1, 2, 3)), dtype=torch.float32)
        elif flag == 'test':
            self.csi_data = torch.tensor(np.transpose(csi_data, (0, 1, 2, 3)), dtype=torch.float32)
        else:
            raise ValueError("Invalid flag. Use 'train' or 'test'.")
        
        # 归一化 CSI 数据
        self.csi_scaler = MinMaxScaler()
        csi_data_normalized = self.csi_scaler.fit_transform(self.csi_data.reshape(-1, self.csi_data.shape[-1])).reshape(
            self.csi_data.shape)
        self.csi_data = torch.tensor(csi_data_normalized, dtype=torch.float32)

        # 归一化位置数据
        self.position_scaler = MinMaxScaler()
        position_data_normalized = self.position_scaler.fit_transform(position_data)
        self.positions = torch.tensor(position_data_normalized, dtype=torch.float32)  # 转换为 Tensor
        print(f"True positions dimension: {self.positions.shape}")  # 打印真实位置维度

        # 验证数据集大小是否匹配
        if len(self.csi_data) != len(self.positions):
            raise ValueError("The number of CSI samples and position samples do not match.")

    def __len__(self):
        return self.csi_data.shape[0]

    def __getitem__(self, idx):
        return self.csi_data[idx], self.positions[idx]


# 定义卷积神经网络模型
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=(3, 3), stride=1, padding=1),  # 4 为天线数
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 13 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # 输出 2 个坐标
        )

    def forward(self, x):
        print("原始输入维度:", x.shape)  # 应为 [32, 4, 108, 300]
        x = self.conv(x)
        x = self.fc(x)
        return x


# 定义训练函数
def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs):
    model.train()
    train_losses = []  # 存储训练损失
    test_losses = []  # 存储测试损失

    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, targets in train_loader:
            #GPU
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_loss:.4f}")

        scheduler.step(avg_loss)  # 使用平均损失作为监控指标

        # 在每个 epoch 后测试模型
        model.eval()
        predictions, true_positions = [], []
        epoch_test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                #GPU
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                epoch_test_loss += loss.item()
                predictions.extend(outputs.cpu().numpy())
                true_positions.extend(targets.cpu().numpy())
        epoch_test_loss /= len(test_loader)
        test_losses.append(epoch_test_loss)
        print(f"Test Loss: {epoch_test_loss:.4f}")

    return train_losses, test_losses


# 定义测试函数并反归一化
def evaluate_model(model, data_loader, criterion, position_scaler):
    model.eval()
    predictions = []
    true_positions = []
    with torch.no_grad():
        for inputs, targets in data_loader:
            #GPU
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
            true_positions.append(targets.cpu().numpy())
    predictions = np.concatenate(predictions, axis=0)
    true_positions = np.concatenate(true_positions, axis=0)

    # 反归一化
    predictions_denorm = position_scaler.inverse_transform(predictions)
    true_positions_denorm = position_scaler.inverse_transform(true_positions)

    print(f"Predictions shape after denormalization: {predictions_denorm.shape}")
    print(f"True positions shape after denormalization: {true_positions_denorm.shape}")

    return predictions_denorm, true_positions_denorm


# 定义统一的可视化函数
def visualize_positions(true_positions, predicted_positions, title='Position Comparison', xlim=None, ylim=None):
    # 确保两个数组长度相同
    min_length = min(len(true_positions), len(predicted_positions))
    true_positions = true_positions[:min_length]
    predicted_positions = predicted_positions[:min_length]

    plt.figure(figsize=(8, 8))

    # 设置x轴和y轴的范围（如果提供了）
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    plt.scatter(true_positions[:, 0], true_positions[:, 1], color='blue', label='True Positions')
    plt.scatter(predicted_positions[:, 0], predicted_positions[:, 1], color='red', label='Predicted Positions')

    # 添加连线表示对应关系
    for i in range(min_length):
        plt.plot([true_positions[i, 0], predicted_positions[i, 0]],
                 [true_positions[i, 1], predicted_positions[i, 1]], 'k--')

    plt.title(title)
    plt.legend()
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    # 确保y轴是从下到上增加（默认matplotlib的y轴是从上到下增加）
    plt.gca().invert_yaxis()

    plt.show()


# 数据加载路径
train_mat_file = 'C:/Users/529B/Downloads/CNN代码/train1.mat'  # 替换为训练数据路径
test_mat_file = 'C:/Users/529B/Downloads/CNN代码/test1.mat'  # 替换为测试数据路径

# 加载数据集
train_dataset = CSIDataset(train_mat_file, flag='train')
test_dataset = CSIDataset(test_mat_file, flag='test')

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 初始化模型、损失函数和优化器
model = CNNModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.MSELoss()  # 使用MSE作为损失函数
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)  # 学习率调度器

# 训练模型
train_losses, test_losses = train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs=100)

# 测试模型并获取预测结果（包括反归一化）
test_predictions, test_true_positions = evaluate_model(model, test_loader, criterion, test_dataset.position_scaler)

# 误差评估
mse = mean_squared_error(test_true_positions, test_predictions)
mae = mean_absolute_error(test_true_positions, test_predictions)
rmse = np.sqrt(mse)
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# 将测试集预测位置和真实位置保存到 Excel 文件
test_df = pd.DataFrame({
    'True X': test_true_positions[:, 0],
    'True Y': test_true_positions[:, 1],
    'Predicted X': test_predictions[:, 0],
    'Predicted Y': test_predictions[:, 1]
})
test_df.to_excel('test_predictions_and_true_positions.xlsx', index=False)
print("Test predictions and true positions saved to test_predictions_and_true_positions.xlsx")

# 将训练集预测位置和真实位置保存到 Excel 文件
train_predictions, train_true_positions = evaluate_model(model, train_loader, criterion, train_dataset.position_scaler)
train_df = pd.DataFrame({
    'True X': train_true_positions[:, 0],
    'True Y': train_true_positions[:, 1],
    'Predicted X': train_predictions[:, 0],
    'Predicted Y': train_predictions[:, 1]
})
train_df.to_excel('train_predictions_and_true_positions.xlsx', index=False)
print("Train predictions and true positions saved to train_predictions_and_true_positions.xlsx")

# 可视化训练集的真实位置和预测位置
visualize_positions(train_true_positions, train_predictions,
                    'Training Set True vs Predicted Positions',
                    xlim=[0, 700], ylim=[0, 700])

# 可视化测试集的真实位置和预测位置
visualize_positions(test_true_positions, test_predictions,
                    'Test Set True vs Predicted Positions',
                    xlim=[0, 700], ylim=[0, 700])

# 可视化训练和测试损失函数
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', c='g')
plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss', c='y')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Test Loss')
plt.show()

# 将训练和测试损失保存到 Excel 表格
loss_df = pd.DataFrame({
    'Epoch': range(1, len(train_losses) + 1),
    'Training Loss': train_losses,
    'Test Loss': test_losses
})
loss_df.to_excel('losses.xlsx', index=False)
print("Losses saved to losses.xlsx")