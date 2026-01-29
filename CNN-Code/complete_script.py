import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import re
import glob
from collections import OrderedDict
from scipy.signal import medfilt
from sklearn.preprocessing import MinMaxScaler
from torchvision.models import swin_t

# 添加ConvEnhancedModule类
class ConvEnhancedModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_layers(x)

class CSIDataset(Dataset):
    def __init__(self, directory, time_step, stride=1, cache_size=270):
        self.directory = directory
        self.time_step = time_step
        self.stride = stride
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self.data_files = {ant: sorted(glob.glob(os.path.join(self.directory, f'antenna_{ant}_*.csv'))) for ant in range(1, 4)}
        self.index_map = self._prepare_index_map()
        self.total_samples = sum(len(lst) for lst in self.index_map.values())

    def _prepare_index_map(self):
        index_map = {}
        for file_idx, file_path in enumerate(self.data_files[1]):
            df = self._load_csv(file_path)
            num_segments = (len(df[0]) - self.time_step + 1) // self.stride
            index_map[file_idx] = list(range(num_segments))
        return index_map

    def _load_csv(self, file_path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        key = file_path
        if key in self.cache:
            print(f"Cache hit for {file_path}")
            self.cache.move_to_end(key)
            return self.cache[key]
        else:
            print(f"Loading and processing {file_path}")
            df = pd.read_csv(file_path, na_values='#NAME?')
            amplitude_data = df.filter(regex='^amplitude_').values.astype(np.float32)
            phase_data = df.filter(regex='^phase_').values.astype(np.float32)
            amplitude_data = self.min_max_normalization(medfilt(amplitude_data))
            phase_data = self.min_max_normalization(medfilt(phase_data))
            amplitude_tensor = torch.tensor(amplitude_data, dtype=torch.float32).to(device)
            phase_tensor = torch.tensor(phase_data, dtype=torch.float32).to(device)
            self.cache[key] = (amplitude_tensor, phase_tensor)
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
            return self.cache[key]

    def min_max_normalization(self, data):
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        cumulative_count = 0
        for file_idx, segments in self.index_map.items():
            if cumulative_count + len(segments) > idx:
                segment_idx = idx - cumulative_count
                break
            cumulative_count += len(segments)

        all_channels_data = []
        for ant in range(1, 4):
            amplitude_tensor, phase_tensor = self._load_csv(self.data_files[ant][file_idx])
            start_idx = segment_idx * self.stride
            end_idx = start_idx + self.time_step
            all_channels_data.append(amplitude_tensor[start_idx:end_idx])
            all_channels_data.append(phase_tensor[start_idx:end_idx])

        sample_data = torch.stack(all_channels_data)
        
        # 时域和频域数据划分
        time_data = sample_data  # 假设当前数据为时域数据
        freq_data = torch.fft.fft(time_data, dim=-1).real
        
        match = re.search(r'antenna_\d+_(\d+)_(\d+).csv', os.path.basename(self.data_files[1][file_idx]))
        x, y = map(int, match.groups()) if match else (0, 0)
        return time_data, freq_data, torch.tensor([x, y], dtype=torch.float32, device=sample_data.device)

class SimilarConvModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_layers(x)

# 添加DualChannelTransformer类
class DualChannelTransformer(nn.Module):
    def __init__(self, time_channels, freq_channels, num_classes):
        super().__init__()
        # 时域通道处理
        self.time_conv = ConvEnhancedModule(time_channels)
        
        # 频域通道处理
        self.freq_conv = ConvEnhancedModule(freq_channels)
        
        # Swin Transformer模块 (tiny版本)
        self.time_transformer = swin_t(weights=None)
        self.time_transformer.features[0][0] = nn.Conv2d(time_channels, 96, kernel_size=(4, 4), stride=(4, 4))
        self.time_transformer.head = nn.Identity()
        
        self.freq_transformer = swin_t(weights=None)
        self.freq_transformer.features[0][0] = nn.Conv2d(freq_channels, 96, kernel_size=(4, 4), stride=(4, 4))
        self.freq_transformer.head = nn.Identity()
        
        # 特征聚合
        self.aggregation = nn.Sequential(
            nn.Linear(1536, 256),  # Swin-T输出为768维，双通道共1536维
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, time_input, freq_input):
        # 输入维度处理 (Section 6.1)
        # time_input: [B, Nk, T] -> [B, Nk, T, 1]
        # freq_input: [B, Mk, L] -> [B, Mk, L, 1]
        time_input = time_input.unsqueeze(-1)
        freq_input = freq_input.unsqueeze(-1)
        
        # CNN特征提取
        time_feat = self.time_conv(time_input)
        freq_feat = self.freq_conv(freq_input)
        
        # 移除宽度维度 (为Transformer准备)
        time_feat = time_feat.squeeze(-1)
        freq_feat = freq_feat.squeeze(-1)
        
        # 调整维度以适配SwinTransformer输入 [B, C, H, W]
        time_feat = time_feat.unsqueeze(-1)
        freq_feat = freq_feat.unsqueeze(-1)
        
        # Transformer处理
        time_trans = self.time_transformer(time_feat)
        freq_trans = self.freq_transformer(freq_feat)
        
        # 全局平均池化
        time_pool = torch.mean(time_trans, dim=2)
        freq_pool = torch.mean(freq_trans, dim=2)
        
        # 双通道特征聚合
        combined = torch.cat((time_pool, freq_pool), dim=1)
        
        # 分类输出
        return self.aggregation(combined)

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for time_input, freq_input, labels in train_loader:
        time_input, freq_input, labels = time_input.to(device), freq_input.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(time_input, freq_input)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for time_input, freq_input, labels in test_loader:
            time_input, freq_input, labels = time_input.to(device), freq_input.to(device), labels.to(device)
            outputs = model(time_input, freq_input)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(test_loader)

if __name__ == "__main__":
    # 配置参数
    data_dir = 'your_data_directory'
    time_step = 10
    time_channels = 6  # 3天线 * (幅值 + 相位)
    freq_channels = 6  # 3天线 * (幅值 + 相位)
    num_classes = 2  # 假设为x,y坐标
    batch_size = 32
    epochs = 10
    learning_rate = 0.001

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建数据集和数据加载器
    dataset = CSIDataset(data_dir, time_step)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型、损失函数和优化器
    model = DualChannelTransformer(time_channels, freq_channels, num_classes).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练和评估
    for epoch in range(epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_loss = evaluate(model, test_loader, criterion, device)
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')