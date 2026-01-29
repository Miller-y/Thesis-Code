import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
import random
from torch.utils.data import DataLoader, Dataset
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
import scipy.signal as signal
import pandas as pd
import matplotlib.pyplot as plt

# 定义应用低通滤波器的函数
def apply_low_pass_filter_frame_wise(csi_data, sampling_rate_frames, cutoff_freq):
    """
    Apply a low-pass filter to the CSI data frame-wise.

    Parameters:
        csi_data (numpy.ndarray): The CSI data in shape [N, C, H, F], where N is number of samples,
                                  C is number of channels, H is height (subcarriers), and F is frames.
        sampling_rate_frames (float): Sampling rate of the frames in Hz.
        cutoff_freq (float): The cutoff frequency of the low-pass filter in Hz.

    Returns:
        numpy.ndarray: Filtered CSI data.
    """
    nyquist = 0.5 * sampling_rate_frames
    normal_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(5, normal_cutoff, btype='low', analog=False)

    # Assuming the last dimension is the one containing frame-wise data for filtering.
    filtered_data = np.zeros_like(csi_data)
    for i in range(csi_data.shape[0]):  # 样本数
        for j in range(csi_data.shape[1]):  # 通道数
            for k in range(csi_data.shape[2]):  # 子载波数
                filtered_data[i, j, k, :] = signal.filtfilt(b, a, csi_data[i, j, k, :])

    return filtered_data



# 定义加载CSI数据和真实位置数据的自定义Dataset
class CSIDataset(Dataset):
    def __init__(self, mat_file_path, sampling_rate_frames=10, cutoff_freq=30):
        # 加载.mat文件中的data和position数据
        mat_data = sio.loadmat(mat_file_path)
        csi_data = mat_data['data']  # 假设CSI数据的键名为 'data'
        position_data = mat_data['position']  # 假设位置数据的键名为 'position'

        # 数据格式调整为 [样本数, 通道数 (天线数), 高度 (子载波数), 宽度 (帧数)]
        self.csi_data = np.transpose(csi_data, (0, 3, 2, 1))

        # 应用低通滤波器（按帧）
        # self.csi_data = apply_low_pass_filter_frame_wise(self.csi_data, sampling_rate_frames, cutoff_freq)

        # 转换为 Tensor 并设置数据类型
        self.csi_data = torch.tensor(self.csi_data, dtype=torch.float32)

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
    



class ConvEnhancedModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 保持原始6层CNN结构
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            
            # nn.Conv2d(64, 64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            # nn.LeakyReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            
            # nn.Conv2d(128, 128, kernel_size=3, padding=1),
            # nn.BatchNorm2d(128),
            # nn.LeakyReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            
            # nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256),
            # nn.LeakyReLU(inplace=True)
        )
    
    def forward(self, x):
        print("原始输入维度:", x.shape)  # 应为 [32, 4, 108, 300]
        return self.conv_layers(x)
    


class SpatialAttention(nn.Module):
    """空间注意力模块 - 增强位置特征提取"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # 空间注意力权重
        attn = self.sigmoid(self.conv(x))
        return x * attn

class IndoorLocalizationModel(nn.Module):
    def __init__(
        self,
        in_channels,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 频域特征提取器 (处理4×108的频域切片)
        self.freq_encoder = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),  # 保持子载波维度
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),       # 108 -> 54
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),       # 54 -> 27
            
            nn.Conv1d(128, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)                     # 全局池化 -> [d_model, 1]
        )
        
        # 可学习的[CLS] token (用于聚合序列信息)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 位置编码 (序列长度=300帧 + 1个[CLS])
        self.pos_embed = nn.Parameter(torch.randn(1, 301, d_model))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 坐标回归头
        self.regressor = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)  # 输出二维坐标 (x, y)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x形状: [batch_size, 4, 108, 300]
        batch_size = x.size(0)
        
        # 步骤1: 频域特征提取
        # 重排维度: [batch, 300, 4, 108]
        x = x.permute(0, 3, 1, 2)
        # 合并batch和时间: [batch*300, 4, 108]
        x = x.reshape(-1, 4, 108)
        
        # CNN处理: [batch*300, d_model, 1]
        x = self.freq_encoder(x)
        # 压缩最后维度: [batch*300, d_model]
        x = x.squeeze(-1)
        # 恢复结构: [batch, 300, d_model]
        x = x.view(batch_size, 300, -1)
        
        # 步骤2: 添加[CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [batch, 301, d_model]
        
        # 步骤3: 添加位置编码
        x += self.pos_embed
        
        # 步骤4: Transformer编码
        x = self.transformer_encoder(x)  # [batch, 301, d_model]
        
        # 步骤5: 取[CLS] token输出
        cls_output = x[:, 0, :]  # [batch, d_model]
        
        # 步骤6: 回归坐标
        coords = self.regressor(cls_output)  # [batch, 2]
        return coords
    
# 数据预处理
# 坐标归一化（重要！）
# 假设场地尺寸为 [x_min, x_max], [y_min, y_max]
def normalize_coords(coords, x_min, x_max, y_min, y_max):
    coords[:, 0] = (coords[:, 0] - x_min) / (x_max - x_min)  # x归一化到[0,1]
    coords[:, 1] = (coords[:, 1] - y_min) / (y_max - y_min)  # y归一化到[0,1]
    return coords

# 预测后反归一化
def denormalize(pred, x_min, x_max, y_min, y_max):
    pred[:, 0] = pred[:, 0] * (x_max - x_min) + x_min
    pred[:, 1] = pred[:, 1] * (y_max - y_min) + y_min
    return pred

# 评估指标
# 定位误差（欧氏距离）
def localization_error(pred, target):
    return torch.sqrt(torch.sum((pred - target)**2, dim=1))

# 平均定位误差
def avg_localization_error(predictions, targets):
    return localization_error(predictions, targets).mean().item()


# 数据增强策略
# CSI数据增强（提高泛化性）
# def augment_csi(data):
#     # 添加高斯噪声
#     data += torch.randn_like(data) * 0.01
    
#     # 随机时间偏移
#     shift = random.randint(-10, 10)
#     data = torch.roll(data, shifts=shift, dims=2)
#     return data

# 训练函数示例
def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs, device, x_min, x_max, y_min, y_max):
    model.train()
    train_losses = []  # 存储训练损失
    test_losses = []  # 存储测试损失
    avg_errors = []  # 存储平均定位误差

    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, targets in train_loader:
            # 数据增强
            # inputs = {key: augment_csi(data) for key, data in inputs.items()}
            time_input = inputs

            # 坐标归一化
            # targets = normalize_coords(targets, x_min, x_max, y_min, y_max)
            # 确保输入和目标都在正确的设备上
            time_input, targets = time_input.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(time_input)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_loss:.4f}")

        scheduler.step(avg_loss)

        # 在每个 epoch 后测试模型
        model.eval()
        predictions, true_positions = [], []
        epoch_test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                time_input = inputs
                # 坐标归一化
                # targets = normalize_coords(targets, x_min, x_max, y_min, y_max)
                # 确保输入和目标都在正确的设备上
                time_input, targets = time_input.to(device), targets.to(device)
                outputs = model(time_input)
                loss = criterion(outputs, targets)
                epoch_test_loss += loss.item()
                predictions.extend(outputs.cpu().numpy())
                true_positions.extend(targets.cpu().numpy())
        epoch_test_loss /= len(test_loader)
        test_losses.append(epoch_test_loss)
        predictions = torch.tensor(predictions)
        true_positions = torch.tensor(true_positions)
        # 反归一化
        # predictions = denormalize(predictions, x_min, x_max, y_min, y_max)
        # true_positions = denormalize(true_positions, x_min, x_max, y_min, y_max)
        avg_error = avg_localization_error(predictions, true_positions)
        avg_errors.append(avg_error)
        print(f"Test Loss: {epoch_test_loss:.4f}, Average Localization Error: {avg_error:.4f}")

    return train_losses, test_losses, avg_errors


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

# 加载数据集，并指定帧的采样率为100Hz
train_dataset = CSIDataset(train_mat_file, sampling_rate_frames=100, cutoff_freq=30)
test_dataset = CSIDataset(test_mat_file, sampling_rate_frames=100, cutoff_freq=30)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 初始化模型、损失函数和优化器，并移动到指定设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 假设时间通道和频率通道数相同
time_channels = train_dataset.csi_data.shape[1]
model = IndoorLocalizationModel(time_channels).to(device)
criterion = nn.SmoothL1Loss().to(device)  # 更改为 SmoothL1Loss 损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)  # 学习率调度器

# 假设场地尺寸
x_min, x_max = 0, 300
y_min, y_max = 0, 680

# 训练模型
train_losses, test_losses, avg_errors = train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs=100,
                                        device=device, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

# 测试模型并获取预测结果（包括反归一化）
model.eval()
predictions, true_positions = [], []
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        predictions.extend(outputs.cpu().numpy())
        true_positions.extend(targets.cpu().numpy())
predictions = torch.tensor(predictions)
true_positions = torch.tensor(true_positions)
# test_predictions = denormalize(predictions, x_min, x_max, y_min, y_max)
# test_true_positions = denormalize(true_positions, x_min, x_max, y_min, y_max)
    # 反归一化
position_scaler = MinMaxScaler()
test_predictions = position_scaler.inverse_transform(predictions)
test_true_positions = position_scaler.inverse_transform(true_positions)


# 误差评估
mse = nn.MSELoss()(test_predictions, test_true_positions).item()
mae = nn.L1Loss()(test_predictions, test_true_positions).item()
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
model.eval()
predictions, true_positions = [], []
with torch.no_grad():
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        predictions.extend(outputs.cpu().numpy())
        true_positions.extend(targets.cpu().numpy())
predictions = torch.tensor(predictions)
true_positions = torch.tensor(true_positions)
position_scaler = MinMaxScaler()
train_predictions = position_scaler.inverse_transform(predictions)
train_true_positions = position_scaler.inverse_transform(true_positions)

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
                    xlim=[x_min, x_max], ylim=[y_min, y_max])

# 可视化测试集的真实位置和预测位置
visualize_positions(test_true_positions, test_predictions,
                    'Test Set True vs Predicted Positions',
                    xlim=[x_min, x_max], ylim=[y_min, y_max])

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
    'Test Loss': test_losses,
    'Average Localization Error': avg_errors
})
loss_df.to_excel('losses.xlsx', index=False)
print("Losses saved to losses.xlsx")