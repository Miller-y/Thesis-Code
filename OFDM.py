import numpy as np
import matplotlib.pyplot as plt

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS'] 
plt.rcParams['axes.unicode_minus'] = False

def plot_ofdm_spectrum():
    # 参数设置
    num_subcarriers = 5  # 子载波数量
    center_freq = 0      # 中心频率偏移
    f = np.linspace(-2, 6, 1000)  # 频率范围
    
    plt.figure(figsize=(12, 6))
    
    # 颜色列表
    colors = ['b', 'g', 'r', 'c', 'm']
    
    # 绘制每个子载波
    for k in range(num_subcarriers):
        # 每个子载波的中心频率为 k
        fc = k
        # Sinc函数形状的频谱 |sinc(f-fc)|
        # 注意 numpy 的 sinc 定义为 sin(pi*x)/(pi*x)
        spectrum = np.sinc(f - fc)
        
        # 绘制主瓣及旁瓣
        plt.plot(f, spectrum, color=colors[k % len(colors)], linewidth=2, label=f'Subcarrier {k}')
        
        # 标注峰值点
        plt.scatter([fc], [1], color=colors[k % len(colors)], s=50, zorder=5)
        
        # 可选：标注峰值位置的虚线
        plt.vlines(fc, -0.4, 1, colors=colors[k % len(colors)], linestyles='dashed', alpha=0.5)

    # 标注正交性关键点：在某个子载波的峰值处，其他子载波为0
    # 例如，关注 Subcarrier 2 (k=2)
    k_target = 2
    plt.annotate('在 $f_2$ 处:\n子载波 2 达到峰值\n其他子载波均为 0',
                 xy=(k_target, 1), 
                 xytext=(k_target + 1.5, 0.8),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=12,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

    # 标注零点 (举例说明 Subcarrier 2 在 f=1, f=3 处为0)
    # plt.scatter([1, 3], [0, 0], color='r', s=50, marker='x', zorder=5, label='Zero Crossings')

    plt.axhline(0, color='black', linewidth=1)
    
    # 图表装饰
    plt.title('OFDM 频谱正交性示意图\n(OFDM Spectrum Orthogonality)', fontsize=16)
    plt.xlabel(r'归一化频率 (Normalized Frequency) $f/ \Delta f$', fontsize=12)
    plt.ylabel('幅度 (Amplitude)', fontsize=12)
    
    # 生成 x 轴刻度标签
    ticks = range(num_subcarriers)
    tick_labels = [f'$f_{k}$' for k in ticks]
    plt.xticks(ticks, tick_labels, fontsize=12)
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')
    
    # 添加说明文本
    text_str = (
        "正交性含义:\n"
        "子载波峰值对应其他子载波的零点。\n"
        "因此在接收端采样时互不干扰。"
    )
    plt.text(-1.8, 0.5, text_str, fontsize=11, 
             bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="orange", alpha=0.9))

    plt.tight_layout()
    # plt.show()
    plt.savefig('OFDM_Spectrum.png', dpi=300)
    print("图像已保存为 OFDM_Spectrum.png")

if __name__ == "__main__":
    plot_ofdm_spectrum()
