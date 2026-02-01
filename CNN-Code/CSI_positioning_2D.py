import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import scipy.io as sio
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
import pandas as pd

# ---------------------- 数据集定义 ----------------------
class CSIDataset(Dataset):
    def __init__(self, mat_file_path, csi_scaler=None, position_scaler=None):
        mat_data = sio.loadmat(mat_file_path)
        csi_data = mat_data['data'] # 预期形状: (1620, 8, 108, 60) 或类似
        position_data = mat_data['position']

        # 确保数据为 float32
        csi_data = csi_data.astype(np.float32)

        # 维度调整逻辑
        if csi_data.shape[1] == 8 and csi_data.shape[2] == 108 and csi_data.shape[3] == 60:
            self.csi_data = csi_data
        elif csi_data.ndim == 4:
            self.csi_data = csi_data
        else:
            raise ValueError(f"CSI数据维度不符合预期: {csi_data.shape}")

        # 预处理：去噪
        self.csi_data = self.apply_filter(self.csi_data, cutoff=20, fs=100)

        # 归一化处理
        # 1. CSI 数据归一化
        N, C, H, W = self.csi_data.shape
        csi_data_flat = self.csi_data.reshape(-1, 1)

        if csi_scaler is None:
            self.csi_scaler = MinMaxScaler()
            csi_data_normalized = self.csi_scaler.fit_transform(csi_data_flat).reshape(N, C, H, W)
        else:
            self.csi_scaler = csi_scaler
            csi_data_normalized = self.csi_scaler.transform(csi_data_flat).reshape(N, C, H, W)

        self.csi_data = torch.tensor(csi_data_normalized, dtype=torch.float32)

        # 2. 位置数据归一化
        if position_scaler is None:
            self.position_scaler = MinMaxScaler()
            self.positions = torch.tensor(
                self.position_scaler.fit_transform(position_data),
                dtype=torch.float32
            )
        else:
            self.position_scaler = position_scaler
            self.positions = torch.tensor(
                self.position_scaler.transform(position_data),
                dtype=torch.float32
            )

        if len(self.csi_data) != len(self.positions):
            raise ValueError("CSI数据与位置数据数量不匹配")

    def apply_filter(self, data, order=6, cutoff=5, fs=100):
        # 对最后一个维度（时间）进行滤波
        # cutoff / (0.5 * fs) 是归一化截止频率
        nyquist = 0.5 * fs
        if cutoff >= nyquist:
             cutoff = nyquist - 0.1 # 防止报错
        b, a = signal.butter(order, cutoff / nyquist, btype='low')
        return signal.filtfilt(b, a, data, axis=-1)

    def __len__(self):
        return len(self.csi_data)

    def __getitem__(self, idx):
        return self.csi_data[idx], self.positions[idx]

# ---------------------- 多尺度时序卷积 (MSTC) ----------------------
class MSTC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(3, 9, 15, 25)):
        super(MSTC, self).__init__()
        # 并行四个卷积分支，默认核大小分别为 1×3、1×9、1×15、1×25
        # 允许通过 kernel_sizes 参数自定义核大小
        branch_channels = out_channels // 4

        p = [(k - 1) // 2 for k in kernel_sizes]

        self.branch1 = nn.Conv2d(in_channels, branch_channels, kernel_size=(1, kernel_sizes[0]), padding=(0, p[0]))
        self.branch2 = nn.Conv2d(in_channels, branch_channels, kernel_size=(1, kernel_sizes[1]), padding=(0, p[1]))
        self.branch3 = nn.Conv2d(in_channels, branch_channels, kernel_size=(1, kernel_sizes[2]), padding=(0, p[2]))
        self.branch4 = nn.Conv2d(in_channels, branch_channels, kernel_size=(1, kernel_sizes[3]), padding=(0, p[3]))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        out = torch.cat([x1, x2, x3, x4], dim=1)
        out = self.bn(out)
        out = self.relu(out)
        return out

# ---------------------- 深度残差块 (DRB) -> 升级为 ConvNeXt Block ----------------------
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ConvNeXtBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = nn.Identity() # Simplified for this use case

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

# ---------------------- 增强通道注意力 (ECA) ----------------------
class ECA(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ECA, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)
        std_pool = torch.std(x.view(b, c, -1), dim=2)

        y_avg = self.mlp(avg_pool)
        y_max = self.mlp(max_pool)
        y_std = self.mlp(std_pool)

        y = y_avg + y_max + y_std
        scale = self.sigmoid(y).view(b, c, 1, 1)
        return x * scale

# ---------------------- 增强空间注意力 (ESA) -> 升级为 Coordinate Attention ----------------------
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        identity = x

        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_h * a_w
        return out

# ---------------------- 混合损失函数 (Position + Smoothing) ----------------------
class CustomLoss(nn.Module):
    def __init__(self, beta=1.0, eta1=1.0, eta2=0.5):
        super(CustomLoss, self).__init__()
        self.beta = beta
        self.eta1 = eta1
        self.eta2 = eta2
        # Lp: 使用 PyTorch 内置的 SmoothL1Loss，它在 |x|<beta 时是 0.5*x^2/beta，否则是 |x|-0.5*beta
        # 这与图片中的公式精神一致（鲁棒回归），且数值更稳定
        self.position_loss = nn.SmoothL1Loss(beta=beta, reduction='mean')

    def forward(self, pred, target):
        # 1. 位置损失 (Lp)
        l_p = self.position_loss(pred, target)

        # 2. 平滑损失 (Ls)
        # 计算相邻时间步的差分向量 (Velocity/Step vector)
        # 注意：这要求 Batch 中的数据是按时间顺序排列的
        diff_pred = pred[1:] - pred[:-1]
        diff_target = target[1:] - target[:-1]

        # 计算差分向量之间的欧氏距离 (L2 norm)
        # Ls = mean( || (p_i - p_{i-1}) - (t_i - t_{i-1}) ||_2 )
        if diff_pred.shape[0] > 0:
            l_s = torch.mean(torch.norm(diff_pred - diff_target, p=2, dim=1))
        else:
            l_s = torch.tensor(0.0, device=pred.device)

        # 总损失
        loss = self.eta1 * l_p + self.eta2 * l_s
        return loss

# ---------------------- MSRANet 模型 (升级版) ----------------------
class MSRANet(nn.Module):
    def __init__(self):
        super(MSRANet, self).__init__()

        # 1. 特征提取模块
        # 输入通道改为 8 (对应 8 个 ESP32 设备)
        self.stage1 = nn.Sequential(
            MSTC(8, 64),
            ConvNeXtBlock(64),
            nn.AvgPool2d(2)
        )

        self.stage2 = nn.Sequential(
            MSTC(64, 128),
            ConvNeXtBlock(128),
            nn.AvgPool2d(2)
        )

        self.stage3 = nn.Sequential(
            # 深层特征的时间维度较小，减小卷积核尺寸以避免无效计算
            MSTC(128, 256, kernel_sizes=(3, 5, 7, 9)),
            ConvNeXtBlock(256),
            nn.AvgPool2d(2)
        )

        # 2. 双重增强注意力机制 (ECA + CoordAtt)
        self.attention = nn.Sequential(
            ECA(256),
            CoordAtt(256, 256)
        )

        # 3. 输出层 (3D坐标) -> 修改为 2D 坐标
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 2) # 输出 x, y (假设为 2D 坐标)
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.attention(x)

        x = self.avgpool(x)
        x = self.classifier(x)
        return x

# ---------------------- 输出平滑模块 (WMA) ----------------------
class WeightedMovingAverage:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.history = []

    def update(self, new_point):
        self.history.append(new_point)
        if len(self.history) > self.window_size:
            self.history.pop(0)

        weights = np.arange(1, len(self.history) + 1)
        weights = weights / weights.sum()

        weighted_sum = np.zeros_like(new_point)
        for i, point in enumerate(self.history):
            weighted_sum += point * weights[i]

        return weighted_sum

    def smooth_sequence(self, sequence):
        smoothed = []
        self.history = []
        for point in sequence:
            smoothed.append(self.update(point))
        return np.array(smoothed)

# ---------------------- 训练函数 ----------------------
def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs, device):
    model.train()
    loss_history = {'train': [], 'test': []}

    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        loss_history['train'].append(epoch_loss / len(train_loader))

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                test_loss += criterion(outputs, targets).item()
        loss_history['test'].append(test_loss / len(test_loader))

        scheduler.step(loss_history['test'][-1])
        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {loss_history['train'][-1]:.4f} | Test Loss: {loss_history['test'][-1]:.4f}")

    return loss_history

# ---------------------- 评估函数 ----------------------
def evaluate_model(model, data_loader, scaler, device):
    model.eval()
    predictions = []
    true_positions = []
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs).cpu().numpy()
            predictions.append(outputs)
            true_positions.append(targets.numpy())

    predictions = np.concatenate(predictions, axis=0)
    true_positions = np.concatenate(true_positions, axis=0)
    predictions = scaler.inverse_transform(predictions)
    true_positions = scaler.inverse_transform(true_positions)
    return predictions, true_positions

# ---------------------- 可视化函数 (2D/3D 动态适应) ----------------------
def plot_comparison(true, pred, title, xlim=None, ylim=None, zlim=None):
    # 设置学术科研风格
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    
    num_dimensions = true.shape[1]

    if num_dimensions == 3:
        fig = plt.figure(figsize=(10, 8), dpi=600)
        ax = fig.add_subplot(111, projection='3d')

        # 绘制真实轨迹和预测轨迹 - 使用更清晰的颜色和标记
        ax.scatter(true[:, 0], true[:, 1], true[:, 2], c='#1f77b4', label='True Position', alpha=0.7, s=15, edgecolors='none') # 经典蓝
        ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], c='#d62728', label='Predicted Position', alpha=0.7, s=15, edgecolors='none') # 经典红

        # 绘制连接线
        for i in range(min(len(true), len(pred))):
            ax.plot([true[i, 0], pred[i, 0]],
                    [true[i, 1], pred[i, 1]],
                    [true[i, 2], pred[i, 2]], 'k--', linewidth=0.5, alpha=0.3)

        ax.set_zlabel('Z Axis')
        if zlim: ax.set_zlim(zlim)

    elif num_dimensions == 2:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=600)

        # 绘制真实轨迹和预测轨迹
        ax.scatter(true[:, 0], true[:, 1], c='#1f77b4', label='True Position', alpha=0.7, s=15, edgecolors='none')
        ax.scatter(pred[:, 0], pred[:, 1], c='#d62728', label='Predicted Position', alpha=0.7, s=15, edgecolors='none')

        # 绘制连接线
        for i in range(min(len(true), len(pred))):
            ax.plot([true[i, 0], pred[i, 0]],
                    [true[i, 1], pred[i, 1]], 'k--', linewidth=0.5, alpha=0.3)
    else:
        raise ValueError(f"不支持的维度: {num_dimensions}. 仅支持 2D 或 3D 坐标。")

    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')

    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)

    ax.legend(frameon=True, fancybox=False, edgecolor='black', loc='best')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # 保存高分辨率结果图
    save_path = 'prediction_comparison.png'
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    print(f"已保存高分辨率对比图: {save_path}")
    
    plt.show()

# ---------------------- 特征图可视化函数 ----------------------
def visualize_feature_maps(model, dataset, device, save_dir='feature_visualization'):
    """
    可视化模型关键层的特征图：
    1. MSTC多尺度卷积响应 (Kernel=3, 9, 15, 25)
    2. 注意力机制各阶段响应 (Pre-Attn, ECA, CoordAtt)
    3. 主干网络各阶段输出 (Stage1, Stage2, Stage3)
    """
    import os
    import matplotlib.pyplot as plt
    
    # 设置学术科研风格
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.0 # 坐标轴线宽
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.eval()
    
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    # --- 1. 注册 Hooks ---
    hooks = []
    
    # A. MSTC 多尺度响应 (使用Stage 1的MSTC模块)
    mstc_module = model.stage1[0] 
    hooks.append(mstc_module.branch1.register_forward_hook(get_activation('MSTC_k3')))
    hooks.append(mstc_module.branch2.register_forward_hook(get_activation('MSTC_k9')))
    hooks.append(mstc_module.branch3.register_forward_hook(get_activation('MSTC_k15')))
    hooks.append(mstc_module.branch4.register_forward_hook(get_activation('MSTC_k25')))

    # B. 注意力机制分析
    hooks.append(model.stage3.register_forward_hook(get_activation('Before_Attention')))
    hooks.append(model.attention[0].register_forward_hook(get_activation('After_ECA')))
    hooks.append(model.attention[1].register_forward_hook(get_activation('After_CoordAtt')))

    # C. 主干网络各阶段输出 (恢复常规层可视化)
    hooks.append(model.stage1.register_forward_hook(get_activation('Output_Stage1')))
    hooks.append(model.stage2.register_forward_hook(get_activation('Output_Stage2')))
    # Stage3 的输出实际上就是 Before_Attention，这里不再重复注册，后面绘图时复用

    # --- 2. 推理 ---
    input_tensor, _ = dataset[0] 
    input_tensor = input_tensor.unsqueeze(0).to(device) # (1, C, H, W)
    
    print(f"正在生成高分辨率特征可视化图 (DPI=600)，保存至 {save_dir}/ ...")
    with torch.no_grad():
        _ = model(input_tensor)

    # --- 3. 移除 Hooks ---
    for h in hooks:
        h.remove()

    # --- 4. 绘图辅助函数 (多子图) ---
    def plot_multi_view(data_dict, keys, titles, filename, figsize, rows=1, cols=4):
        fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=600)
        if rows == 1 and cols == 1: axes = [axes]
        axes = axes.flatten() if isinstance(axes, np.ndarray) else axes
        
        # 使用科研常用的colormap
        cmap_name = 'viridis' 
        
        for i, (key, title) in enumerate(zip(keys, titles)):
            if key not in data_dict: continue
            
            feature_map = data_dict[key].squeeze(0)
            heatmap = torch.mean(feature_map, dim=0).cpu().numpy()
            
            ax = axes[i]
            im = ax.imshow(heatmap, aspect='auto', cmap=cmap_name, origin='lower')
            ax.set_title(title, fontsize=14, pad=10)
            ax.set_xticks([])
            ax.set_yticks([])
            
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(1.0)

        plt.tight_layout()
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
        plt.close()
        print(f"已保存组合图: {save_path}")

    # --- 5. 绘图辅助函数 (单张图) ---
    def plot_single_view(data_dict, key, title, filename):
        if key not in data_dict: return
        
        feature_map = data_dict[key].squeeze(0)
        heatmap = torch.mean(feature_map, dim=0).cpu().numpy()
        
        plt.figure(figsize=(8, 6), dpi=600)
        plt.imshow(heatmap, aspect='auto', cmap='viridis', origin='lower')
        plt.colorbar()
        plt.title(title, fontsize=16, pad=12)
        plt.xlabel('Time Steps (Downsampled)', fontsize=12)
        plt.ylabel('Features / Frequency', fontsize=12)
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
        plt.close()
        print(f"已保存单张图: {save_path}")

    # --- 6. 生成图表 ---
    
    # (1) MSTC 多尺度响应分析
    plot_multi_view(
        activations,
        ['MSTC_k3', 'MSTC_k9', 'MSTC_k15', 'MSTC_k25'],
        ['(a) Kernel=3', '(b) Kernel=9', '(c) Kernel=15', '(d) Kernel=25'],
        'Multi_scale_responses.png',
        figsize=(20, 4), rows=1, cols=4
    )

    # (2) Attention 注意力机制分析
    plot_multi_view(
        activations,
        ['Before_Attention', 'After_ECA', 'After_CoordAtt'],
        ['Without Attention', 'After ECA', 'After Coordinate Attention'],
        'Attention_Analysis.png',
        figsize=(15, 4), rows=1, cols=3
    )
    
    # (3) 各层单独特征图 (常规可视化)
    plot_single_view(activations, 'Output_Stage1', 'Stage 1 Output Feature Map', 'Layer_Stage1.png')
    plot_single_view(activations, 'Output_Stage2', 'Stage 2 Output Feature Map', 'Layer_Stage2.png')
    plot_single_view(activations, 'Before_Attention', 'Stage 3 Output Feature Map', 'Layer_Stage3.png') # Stage3

    # (4) 原始输入可视化
    raw_input = input_tensor.squeeze(0).cpu().numpy() # (8, 108, 60)
    plt.figure(figsize=(8, 4), dpi=600)
    plt.imshow(raw_input[0], aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar()
    plt.title("Original Input (Antenna 1)", fontsize=14)
    plt.xlabel("Time Steps")
    plt.ylabel("Subcarriers")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'Original_Input.png'), dpi=600)
    plt.close()

# ---------------------- CDF 绘图函数 ----------------------
def plot_cdf(test_true, test_pred, save_path='cdf_error.png'):
    """
    绘制并保存定位误差的累积分布函数 (CDF) 图
    """
    import matplotlib.pyplot as plt
    
    # 设置学术科研风格
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    
    # 计算欧氏距离误差
    # 原始数据范围 [0, 400/700] 暗示单位为厘米 (cm)
    errors_cm = np.linalg.norm(test_true - test_pred, axis=1)
    
    # 转换为米 (m) 用于绘图，符合学术惯例
    errors_m = errors_cm / 100.0
    
    # 排序误差
    sorted_errors = np.sort(errors_m)
    # 计算 CDF y值 (0 to 1)
    yvals = np.arange(len(sorted_errors)) / float(len(sorted_errors) - 1)
    
    plt.figure(figsize=(8, 6), dpi=600)
    plt.plot(sorted_errors, yvals, linewidth=2.5, color='#1f77b4', label='MSRANet')
    
    plt.xlabel('Localization Error (m)', fontsize=14)
    plt.ylabel('CDF', fontsize=14)
    plt.title('CDF of Localization Errors', fontsize=16, fontweight='bold', pad=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=True, fancybox=False, edgecolor='black', loc='lower right')
    
    # 限制 x 轴范围更好地展示细节 (可选，例如展示 0 到 3米)
    # plt.xlim(0, 3.0) 
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"已保存 CDF 分布图 (单位: m): {save_path}")

# ---------------------- 模型复杂度计算 ----------------------
def calculate_complexity(model, input_size, device):
    """
    计算模型的参数量 (Parameters) 和 计算量 (FLOPs)
    """
    # 1. 计算参数量
    params_count = sum(p.numel() for p in model.parameters())
    params_k = params_count / 1000.0
    
    # 2. 计算 FLOPs (尝试使用 thop 库)
    flops_g = 0.0
    try:
        from thop import profile
        # 创建一个 dummy input
        input_tensor = torch.randn(input_size).to(device)
        # thop.profile 返回的 flops 是 MACs，通常 FLOPs = 2 * MACs
        # 但在学术比较中，大家常直接引用 thop 的输出或指明是 MACs。
        # 这里我们按 GFLOPs 输出
        macs, _ = profile(model, inputs=(input_tensor, ), verbose=False)
        flops_g = macs / 1e9 
    except ImportError:
        print("\n[Warning] 未检测到 'thop' 库，无法精确计算 FLOPs。仅计算参数量。")
        print("可运行 `pip install thop` 安装该库。\n")
        
    return params_k, flops_g

# ---------------------- 主程序 ----------------------
if __name__ == "__main__":
    # Colab 环境适配
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        train_path = '/content/drive/MyDrive/CSI-Data/train1.mat'
        test_path = '/content/drive/MyDrive/CSI-Data/test1.mat'
    except ImportError:
        print("未检测到 Google Colab 环境，使用默认本地路径")
        train_path = 'data/train1.mat'
        test_path = 'data/test1.mat'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 1. 加载并从训练集创建 Scaler
    print(f"正在加载训练集: {train_path} ...")
    train_dataset = CSIDataset(train_path)

    # 2. 使用训练集的 Scaler 加载测试集
    # 这样做是为了保证 Train/Test 的数据分布变换一致，防止数据泄露
    print(f"正在加载测试集: {test_path} ...")
    test_dataset = CSIDataset(test_path,
                              csi_scaler=train_dataset.csi_scaler,
                              position_scaler=train_dataset.position_scaler)

    # === 快速调试模式 ===
    QUICK_DEBUG = False
    if QUICK_DEBUG:
        print("\n" + "="*50)
        print("警告：当前处于快速调试模式 (QUICK_DEBUG=True)")
        print("仅使用少量数据和 Epoch 生成可视化图。若需完整训练请将代码中 QUICK_DEBUG 设为 False。")
        print("="*50 + "\n")
        subset_size = 64
        train_dataset.csi_data = train_dataset.csi_data[:subset_size]
        train_dataset.positions = train_dataset.positions[:subset_size]
        test_dataset.csi_data = test_dataset.csi_data[:subset_size]
        test_dataset.positions = test_dataset.positions[:subset_size]
        EPOCHS = 2
    else:
        EPOCHS = 40
    # ===================

    # 训练集 DataLoader 加入 shuffle=True 以提升泛化能力
    # 由于开启了 shuffle，不再适合计算时序差分损失 (Ls)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 使用 MSRANet 模型
    model = MSRANet().to(device)

    # 损失函数简化：去掉 l_s 项，仅保留 SmoothL1Loss
    # 因为 shuffle=True 后，batch 内的样本不再是时序连续的，计算相邻差分没有意义
    criterion = nn.SmoothL1Loss(beta=1.0)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #,weight_decay=1e-4
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    print("开始训练...")
    loss_history = train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs=EPOCHS, device=device)

    # 评估预测结果
    test_pred, test_true = evaluate_model(model, test_loader, test_dataset.position_scaler, device)

    # 应用输出平滑 (WMA)
    # 注意：由于我们在训练中已经使用了包含平滑损失 (Ls) 的 CustomLoss，
    # 模型输出的轨迹理论上已经具备了较好的平滑性。
    # WMA 在这里作为一个后处理步骤，可以进一步滤除残留的高频抖动，
    # 两者并不冲突，而是互补关系：
    # - CustomLoss (Ls): 在特征学习阶段约束模型，使其倾向于输出平滑轨迹。
    # - WMA: 在预测结果上进行数学平滑，作为最后一道保障。
    #暂时去掉WMA
    # wma = WeightedMovingAverage(window_size=5)
    # test_pred_smooth = wma.smooth_sequence(test_pred)

    # 输出指标
    def print_metrics(true, pred, label):
        print(f"\n=== {label} ===")
        print(f"MSE: {mean_squared_error(true, pred):.2f}")
        print(f"MAE: {mean_absolute_error(true, pred):.2f}")
        print(f"RMSE: {np.sqrt(mean_squared_error(true, pred)):.2f}")

    print_metrics(test_true, test_pred, "原始预测结果")
    # print_metrics(test_true, test_pred_smooth, "平滑后预测结果 (WMA)")

    # 可视化
    # 假设空间范围，根据实际数据调整
    # 请注意，如果您的数据是二维的，3D可视化可能需要调整。
    # 例如，您可以将Z轴设置为零或者不显示。
    plot_comparison(test_true, test_pred, "预测结果对比 (原始 - 2D/3D)",
                   xlim=[0, 400], ylim=[0, 700], zlim=[0, 300] if test_true.shape[1] == 3 else None)

    # 保存结果
    # 根据实际输出维度调整DataFrame列名
    output_cols = ['pred_x', 'pred_y', 'pred_z'] if test_pred.shape[1] == 3 else ['pred_x', 'pred_y']
    output_smooth_cols = ['pred_smooth_x', 'pred_smooth_y', 'pred_smooth_z'] if test_pred.shape[1] == 3 else ['pred_smooth_x', 'pred_smooth_y']
    true_cols = ['true_x', 'true_y', 'true_z'] if test_true.shape[1] == 3 else ['true_x', 'true_y']

    df_data = {}
    for i, col in enumerate(true_cols):
        df_data[col] = test_true[:, i]
    for i, col in enumerate(output_cols):
        df_data[col] = test_pred[:, i]
    for i, col in enumerate(output_smooth_cols):
        df_data[col] = test_pred[:, i]

    pd.DataFrame(df_data).to_excel('results.xlsx', index=False)

    # === 新增 1：CDF 误差分布图 ===
    plot_cdf(test_true, test_pred, save_path='cdf_error_distribution.png')

    # === 新增 2：模型复杂度计算与输出 ===
    # 输入尺寸: (Batch=1, Channels=8, H=108, W=60)
    input_size = (1, 8, 108, 60)
    params_k, flops_g = calculate_complexity(model, input_size, device)
    
    print("\n" + "="*40)
    print("MODEL COMPLEXITY (MSRANet)")
    print("="*40)
    print(f"{'Method':<15} {'Parameters (K)':<18} {'FLOPs (GFLOPs)':<15}")
    print("-" * 48)
    # 如果没装 thop，flops_g 为 0，显示 '-'
    flops_str = f"{flops_g:.3f}" if flops_g > 0 else "-"
    print(f"{'MSRANet':<15} {params_k:<18.2f} {flops_str:<15}")
    print("="*40 + "\n")

    # === 新增 3：可视化模型中间层特征 ===
    # 将结果保存到 'model_vis' 文件夹 (Colab中会在左侧文件栏看到)
    visualize_feature_maps(model, test_dataset, device, save_dir='model_vis')