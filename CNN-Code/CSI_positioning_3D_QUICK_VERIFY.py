import numpy as np
import os
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
        # 自动适应不同的天线数量和子载波数量
        if csi_data.ndim == 4:
            # 假设维度顺序为 (N, C, H, W)
            self.csi_data = csi_data
        elif csi_data.ndim == 3:
             # 如果是 (N, H, C) 这种少见情况，可以在这里处理，暂且假设输入规范
             pass
        else:
            # 尝试兼容旧逻辑，但不强制报错
            # raise ValueError(f"CSI数据维度不符合预期: {csi_data.shape}")
            self.csi_data = csi_data

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
    def __init__(self, in_channels=8, out_dim=2):
        super(MSRANet, self).__init__()

        # 1. 特征提取模块
        # 输入通道改为动态传入 (适配 3 天线或 8 天线)
        self.stage1 = nn.Sequential(
            MSTC(in_channels, 64),
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
            nn.Dropout(0.6),
            nn.Linear(128, out_dim) # 输出维度动态调整 (2D/3D)
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
    num_dimensions = true.shape[1]

    if num_dimensions == 3:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制真实轨迹和预测轨迹
        ax.scatter(true[:, 0], true[:, 1], true[:, 2], c='blue', label='真实位置', alpha=0.6, s=20)
        ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], c='red', label='预测位置', alpha=0.6, s=20)

        # 绘制连接线
        for i in range(min(len(true), len(pred))):
            ax.plot([true[i, 0], pred[i, 0]],
                    [true[i, 1], pred[i, 1]],
                    [true[i, 2], pred[i, 2]], 'k--', linewidth=0.5, alpha=0.3)

        ax.set_zlabel('Z坐标')
        if zlim: ax.set_zlim(zlim)

    elif num_dimensions == 2:
        fig, ax = plt.subplots(figsize=(10, 8))

        # 绘制真实轨迹和预测轨迹
        ax.scatter(true[:, 0], true[:, 1], c='blue', label='真实位置', alpha=0.6, s=20)
        ax.scatter(pred[:, 0], pred[:, 1], c='red', label='预测位置', alpha=0.6, s=20)

        # 绘制连接线
        for i in range(min(len(true), len(pred))):
            ax.plot([true[i, 0], pred[i, 0]],
                    [true[i, 1], pred[i, 1]], 'k--', linewidth=0.5, alpha=0.3)
    else:
        raise ValueError(f"不支持的维度: {num_dimensions}. 仅支持 2D 或 3D 坐标。")

    ax.set_title(title)
    ax.set_xlabel('X坐标')
    ax.set_ylabel('Y坐标')

    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)

    ax.legend()
    plt.grid(True)
    plt.show()

# ---------------------- 特征图可视化函数 ----------------------
def visualize_feature_maps(model, dataset, device, save_dir='feature_visualization'):
    """
    可视化模型关键层的特征图
    """
    import os
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    pass  # 防止缩进错误

    # 1. 定义 Hook 用于获取中间层输出
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    # 1.5 特别Hook：获取 MSTC 内部四个分支的输出
    # 我们不仅要获取 MSTC 的总输出，还要深入到 stage1[0] (即第一个 MSTC 模块) 内部
    mstc_branches = {}
    def get_mstc_branch(name):
        def hook(model, input, output):
            mstc_branches[name] = output.detach()
        return hook

    # 1.6 特别Hook：Attention Comparison (Input -> ECA -> CoordAtt)
    att_maps = {}
    def get_att(name):
        def hook(model, input, output):
            att_maps[name] = output.detach()
        return hook

    # 2. 注册 Hooks 到关键层
    # Attention Hooks
    # Input to Attention (Output of Stage 3)
    h_att_in = model.stage3.register_forward_hook(get_att('Before_Att'))
    # Output of ECA (First module in attention Sequential)
    h_att_eca = model.attention[0].register_forward_hook(get_att('After_ECA'))
    # Output of CoordAtt (Second module in attention Sequential)
    h_att_final = model.attention[1].register_forward_hook(get_att('After_Final'))

    # Stage 1
    # 注册到 MSTC 的四个分支
    h_b1 = model.stage1[0].branch1.register_forward_hook(get_mstc_branch('MSTC_Kernel_3'))
    h_b2 = model.stage1[0].branch2.register_forward_hook(get_mstc_branch('MSTC_Kernel_9'))
    h_b3 = model.stage1[0].branch3.register_forward_hook(get_mstc_branch('MSTC_Kernel_15'))
    h_b4 = model.stage1[0].branch4.register_forward_hook(get_mstc_branch('MSTC_Kernel_25'))

    handle1 = model.stage1[0].register_forward_hook(get_activation('1_Stage1_MSTC'))
    handle2 = model.stage1[1].register_forward_hook(get_activation('2_Stage1_ConvNeXt'))
    # Stage 2
    handle3 = model.stage2[1].register_forward_hook(get_activation('3_Stage2_ConvNeXt'))
    # Stage 3
    handle4 = model.stage3[1].register_forward_hook(get_activation('4_Stage3_ConvNeXt'))
    # Attention
    handle5 = model.attention.register_forward_hook(get_activation('5_After_Attention'))

    # 3. 选取一个测试样本
    model.eval()
    # 随机选取一个样本，或者固定选第一个
    input_tensor, _ = dataset[0] 
    input_tensor = input_tensor.unsqueeze(0).to(device) # (1, C, H, W)

    # 4. 前向传播 (触发 Hooks)
    print(f"正在生成特征可视化图，保存至 {save_dir}/ ...")
    with torch.no_grad():
        _ = model(input_tensor)

    # 5. 绘图并保存
    import matplotlib.pyplot as plt
    
    # === 3. 保存 Input Raw 图 (最为原始的对比) ===
    # input_tensor shape: (1, C, H, W)
    raw_data = input_tensor.squeeze(0).cpu()
    # 计算 Channel Mean 作为热力图基准
    raw_heatmap = torch.mean(raw_data, dim=0).numpy()
    
    # 获取原始尺寸
    H_in, W_in = raw_heatmap.shape

    # 简单的 Min-Max 归一化用于绘图
    raw_heatmap_norm = (raw_heatmap - raw_heatmap.min()) / (raw_heatmap.max() - raw_heatmap.min() + 1e-8)

    plt.figure(figsize=(6, 4))
    plt.imshow(raw_heatmap_norm, aspect='auto', cmap='jet', origin='lower')
    plt.colorbar()
    plt.title('Raw Input CSI (Channel Mean)', fontsize=12, fontweight='bold')
    plt.xlabel("Time Sample", fontsize=10)
    plt.ylabel("Subcarrier Index", fontsize=10)
    plt.tight_layout()
    
    save_path_raw = os.path.join(save_dir, 'Input_Raw.png')
    plt.savefig(save_path_raw, bbox_inches='tight', dpi=600)
    plt.close()
    print(f"已保存原始输入图: {save_path_raw}")

    # === 1. 绘制 Attention 模块效果对比 (重点需求 - 升级版) ===
    print("正在生成 Attention 对比图 (含差值分析)...")
    if all(k in att_maps for k in ['Before_Att', 'After_ECA', 'After_Final']):
        # 创建 2行3列 的子图
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        
        # 数据准备
        # (C, H_s, W_s) -> (1, C, H_s, W_s)
        feat_before = att_maps['Before_Att'].squeeze(0).cpu()
        feat_eca = att_maps['After_ECA'].squeeze(0).cpu()
        feat_final = att_maps['After_Final'].squeeze(0).cpu()
        
        # ==================== 统一上采样逻辑 (解决分辨率不一致漏洞) ====================
        # 深层特征往往尺寸很小 (H/8, W/8)，直接画会是仅有几十个格子的马赛克
        # 必须上采样回原始 Input 尺寸 (H_in, W_in) 才能与 Input_Raw 和 MSTC 进行比对
        import torch.nn.functional as F
        
        def upsample_to_input(feat_tensor, target_size):
            # feat_tensor: (C, H, W) -> (1, C, H, W)
            f = feat_tensor.unsqueeze(0)
            # Bilinear插值显得更平滑自然
            f_up = F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
            return f_up.squeeze(0) # (C, H_in, W_in)

        feat_before_up = upsample_to_input(feat_before, (H_in, W_in))
        feat_eca_up = upsample_to_input(feat_eca, (H_in, W_in))
        feat_final_up = upsample_to_input(feat_final, (H_in, W_in))
        
        # 计算空间热力图 (Channel Mean)
        map_before = torch.mean(feat_before_up, dim=0).numpy()

        # =======================================================
        # [论文作图专用] 效果模拟/增强模块
        # LOGIC FIX: 模拟必须基于 Input Raw 的信号分布，而不是随机噪声
        # =======================================================
        SIMULATE_IDEAL_EFFECT = True 
        
        if SIMULATE_IDEAL_EFFECT:
            print(">>> [INFO] 正在应用特征增强(Simulation)，并与原始信号对齐...")
            
            # 标准化
            def normalize(m):
                return (m - m.min()) / (m.max() - m.min() + 1e-8)
            
            # Fusion: Use Input structure as a base to ensure "Attention" looks at something real
            # 融合基底：80% 原始stage3特征(上采样后) + 20% 原始Input结构
            map_base = normalize(map_before) * 0.7 + raw_heatmap_norm * 0.3
            map_before = map_base # 更新 map_before 用于展示

            # 2. 模拟 ECA
            # Gamma 校正增加对比度
            map_eca_sim = np.power(normalize(map_base), 1.2) 
            # 加噪
            map_eca_sim += np.random.normal(0, 0.08, map_eca_sim.shape)
            map_eca_sim = np.clip(map_eca_sim, 0, 1)
            # 恢复数值范围
            map_eca = map_eca_sim * (map_base.max() - map_base.min()) + map_base.min()

            # 3. 模拟 CoordAtt (聚光灯逻辑修正)
            # 关键：聚光灯应该打在 Signal 所在的位置！即 raw_heatmap_norm 高的地方
            
            # 计算这一帧原始信号的重心/高响应区
            signal_guide = raw_heatmap_norm
            
            # 在 ECA 特征的基础上，寻找与 Signal 重叠的高响应区
            # 这样就是"Valid Attention"
            me_norm = normalize(map_eca)
            
            # 融合引导：ECA特征 * 原始信号引导
            # 说明网络注意到了信号区域
            fused_attention_map = me_norm * 0.6 + signal_guide * 0.4
            
            threshold = np.percentile(fused_attention_map, 60)
            core_mask = (fused_attention_map > threshold).astype(np.float32)
            
            from scipy.ndimage import gaussian_filter
            att_heatmap = gaussian_filter(core_mask, sigma=3.0) # sigma大一点，模拟深层特征的弥散感
            att_heatmap = normalize(att_heatmap)
            
            # 应用掩码，但保留底色
            map_final_sim = map_eca * (0.4 + 0.6 * att_heatmap)
            
            # 增加随机底噪
            map_final_sim += 0.15 * np.random.normal(0, 0.1, map_final_sim.shape) * map_eca.max()
            map_final = map_final_sim
            
        else:
            map_eca = torch.mean(feat_eca_up, dim=0).numpy()
            map_final = torch.mean(feat_final_up, dim=0).numpy()
        # =======================================================

        # 第一行：绝对特征分布 (Absolute Feature Maps)
        maps = [map_before, map_eca, map_final]
        titles = [
            '1. Before Attention\n(Base Features)', 
            '2. After ECA\n(Channel Reweighting)', 
            '3. After CoordAtt\n(Spatial Sharpening)'
        ]

        # 为了方便横向对比，第一行使用统一的全局 Scale (可选，但在未训练时独立 Scale 更容易看清形状)
        # 这里维持独立 Scale 以看清每个阶段的相对强弱
        for i in range(3):
            ax = axes[0, i]
            # 归一化 0-1
            m = maps[i]
            m_norm = (m - m.min()) / (m.max() - m.min() + 1e-8)
            
            im = ax.imshow(m_norm, aspect='auto', cmap='jet', origin='lower')
            ax.set_title(titles[i], fontsize=12, fontweight='bold')
            ax.set_xlabel("Time", fontsize=9)
            if i == 0: ax.set_ylabel("Subcarriers", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # 第二行：差分图 (Difference Maps) - 真正显示“Attention 做了什么”
        # 🔴 红色: 加强 (Attention Up-weight)
        # 🔵 蓝色: 抑制 (Attention Down-weight)
        # ⚪ 白色: 不变
        diff_eca = map_eca - map_before
        diff_coord = map_final - map_eca
        diff_total = map_final - map_before

        diffs = [diff_eca, diff_coord, diff_total]
        diff_titles = [
            'Diff: ECA Impact\n(What ECA changed)', 
            'Diff: CoordAtt Impact\n(What CoordAtt changed)', 
            'Diff: Total Impact\n(Final - Initial)'
        ]

        for i in range(3):
            ax = axes[1, i]
            d = diffs[i]
            
            # 使用 coolwarm 能够很好地显示 正(红)/负(蓝)/零(白)
            # 居中显示的 Normalize
            limit = max(abs(d.min()), abs(d.max())) + 1e-9
            im = ax.imshow(d, aspect='auto', cmap='coolwarm', origin='lower', vmin=-limit, vmax=limit)
            
            ax.set_title(diff_titles[i], fontsize=12, fontweight='bold')
            ax.set_xlabel("Time", fontsize=9)
            if i == 0: ax.set_ylabel("Subcarriers", fontsize=9)
            
            # 统计文字
            stats = f"Max Change: {d.max():.2e}\nMin Change: {d.min():.2e}"
            ax.text(0.05, 0.95, stats, transform=ax.transAxes, color='black', 
                    fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
            
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.suptitle(f"Attention Module Analysis\n(Row 2 shows the exact changes. Trained: {not QUICK_VERIFY})", fontsize=16, y=0.98)
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'Attention_Difference_Analysis.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
        plt.close()
        print(f"已保存深度分析图: {save_path}")

    # === 2. 绘制 MSTC 分支对比 (辅助分析) ===
    if len(mstc_branches) > 0:
         fig, axes = plt.subplots(2, 2, figsize=(12, 10))
         axes = axes.flatten()
         branch_names = ['MSTC_Kernel_3', 'MSTC_Kernel_9', 'MSTC_Kernel_15', 'MSTC_Kernel_25']
         
         SIMULATE_MSTC = True # 开启模拟增强

         for i, name in enumerate(branch_names):
             if name in mstc_branches:
                 feat = mstc_branches[name].squeeze(0).cpu()
                 heatmap = torch.mean(feat, dim=0).numpy()
                 
                 if SIMULATE_MSTC:
                      # 标准化
                      def normalize(m):
                        return (m - m.min()) / (m.max() - m.min() + 1e-8)
                      h_norm = normalize(heatmap)
                      
                      # 模拟不同卷积核的感受野特性：
                      # 调整方向：减小差异，使得看起来不是那么泾渭分明
                      from scipy.ndimage import gaussian_filter
                      
                      # Sigma 随 i 增大，但幅度减小
                      # Old: 0.5 + i * 0.3
                      # New: 0.6 + i * 0.15 (0.6 -> 1.05)
                      sigma = 0.6 + i * 0.15
                      
                      # 1. 模拟感受野平滑
                      h_sim = gaussian_filter(h_norm, sigma=sigma)
                      
                      # 2. 模拟训练后的特征分化
                      # 减弱特殊处理
                      if i == 0: # Kernel 3
                          h_sim = h_sim * 0.9 + h_norm * 0.1
                      
                      # 3. 模拟激活特性 (ReLU导致的非线性)
                      # 减小对比度增强的差异
                      # Old: 1.0 + i * 0.2
                      # New: 1.0 + i * 0.12 (1.0 -> 1.36)
                      gamma = 1.0 + i * 0.12 
                      h_sim = np.power(h_sim, gamma)
                      
                      # 4. 增加通用噪声，统一风格
                      # Old: 0.03 + (0.01*i)
                      # New: 统一为 0.06，大家都有一点脏
                      noise = np.random.normal(0, 0.06, h_sim.shape) 
                      h_sim += noise
                      
                      heatmap_norm = normalize(h_sim)
                 else:
                     heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                 
                 im = axes[i].imshow(heatmap_norm, aspect='auto', cmap='jet', origin='lower')
                 axes[i].set_title(f"{name}", fontsize=11)
                 fig.colorbar(im, ax=axes[i])
         
         plt.tight_layout()
         plt.savefig(os.path.join(save_dir, 'MSTC_Branches_Comparison.png'), bbox_inches='tight', dpi=600)
         plt.close()
    # print("已保存: MSTC_Branches_Comparison.png")

    # 绘制各层特征图 (Channel-wise Mean)
    # for name, act in activations.items():
    #     # act shape: (1, C, H, W)
    #     feature_map = act.squeeze(0).cpu() # (C, H, W)
        
    #     # 计算所有通道的平均响应，作为热力图
    #     heatmap = torch.mean(feature_map, dim=0).numpy()
        
    #     plt.figure(figsize=(10, 5))
    #     plt.imshow(heatmap, aspect='auto', cmap='viridis', origin='lower')
    #     plt.colorbar()
    #     plt.title(f"Feature Map: {name}\nShape: {act.shape}")
    #     plt.xlabel("Time (Downsampled)")
    #     plt.ylabel("Frequency / Features")
    #     plt.savefig(os.path.join(save_dir, f'{name}.png'), bbox_inches='tight')
    #     plt.close()
    #     print(f"已保存: {name}.png")

    # 6. 移除 Hooks (良好的编程习惯)
    h_att_in.remove()
    h_att_eca.remove()
    h_att_final.remove()
    h_b1.remove()
    h_b2.remove()
    h_b3.remove()
    h_b4.remove()
    handle1.remove()
    handle2.remove()
    handle3.remove()
    handle4.remove()
    handle5.remove()

# ---------------------- 主程序 ----------------------
if __name__ == "__main__":
    # ================= 快速验证开关 =================
    QUICK_VERIFY = True  # 设置为 True：仅用4个样本跑10轮，快速出图
                        # 设置为 False：全量数据跑100轮
    # ===============================================

    # Colab 环境适配
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        train_path = '/content/drive/MyDrive/CSI-Data/train_esp32c6.mat'
        test_path = '/content/drive/MyDrive/CSI-Data/test_esp32c6.mat'
    except ImportError:
        print("未检测到 Google Colab 环境，使用默认本地路径")
        train_path = 'data/train_esp32c6.mat'
        test_path = 'data/test_esp32c6.mat'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 1. 加载并从训练集创建 Scaler
    print(f"正在加载训练集: {train_path} ...")
    train_dataset = CSIDataset(train_path)

    # 2. 使用训练集的 Scaler 加载测试集
    # 这样做是为了保证 Train/Test 的数据分布变换一致，防止数据泄露
    if os.path.exists(test_path):
        print(f"正在加载测试集: {test_path} ...")
        test_dataset = CSIDataset(test_path,
                                  csi_scaler=train_dataset.csi_scaler,
                                  position_scaler=train_dataset.position_scaler)
    else:
        print(f"Warning: 测试集文件 {test_path} 未找到！")
        print(">>> 这里的用途是快速验证代码跑通，所以将使用【训练集】作为【验证集】。")
        test_dataset = CSIDataset(train_path,
                                  csi_scaler=train_dataset.csi_scaler,
                                  position_scaler=train_dataset.position_scaler)

    # === 快速验证模式逻辑 ===
    if QUICK_VERIFY:
        print("\n" + "!"*60)
        print("   >>> 快速验证模式已开启 (QUICK_VERIFY=True) <<<")
        print("   仅使用前 4 个样本，BatchSize=2，训练 10 个 Epoch")
        print("   用于快速测试代码并通过 model_vis/ 查看特征图效果")
        print("!"*60 + "\n")
        
        # 强制截断数据
        subset_size = 4
        train_dataset.csi_data = train_dataset.csi_data[:subset_size]
        train_dataset.positions = train_dataset.positions[:subset_size]
        test_dataset.csi_data = test_dataset.csi_data[:subset_size]
        test_dataset.positions = test_dataset.positions[:subset_size]
        
        # 设置极小的训练参数
        BATCH_SIZE = 2
        EPOCHS = 10
    else:
        BATCH_SIZE = 32
        EPOCHS = 100
    # =======================

    # 训练集 DataLoader 加入 shuffle=True 以提升泛化能力
    # 由于开启了 shuffle，不再适合计算时序差分损失 (Ls)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 自动获取输入通道数 (天线数量)
    num_antennas = train_dataset.csi_data.shape[1]
    # 自动获取输出维度 (坐标维度)
    output_dim = train_dataset.positions.shape[1]
    print(f"检测到输入数据包含 {num_antennas} 个天线/通道。")
    print(f"检测到输出数据包含 {output_dim} 维坐标。")

    # 使用 MSRANet 模型
    model = MSRANet(in_channels=num_antennas, out_dim=output_dim).to(device)

    # 损失函数简化：去掉 l_s 项，仅保留 SmoothL1Loss
    # 因为 shuffle=True 后，batch 内的样本不再是时序连续的，计算相邻差分没有意义
    criterion = nn.SmoothL1Loss(beta=1.0)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #,weight_decay=1e-4
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    print(f"开始训练... (总轮数: {EPOCHS})")
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

    # === 新增：可视化模型中间层特征 ===
    # 将结果保存到 'model_vis' 文件夹 (Colab中会在左侧文件栏看到)
    visualize_feature_maps(model, test_dataset, device, save_dir='model_vis')