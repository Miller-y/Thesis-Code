在代码中，这两个指标的计算逻辑如下：

### 1. 变异系数 (Coefficient of Variation, CV)

变异系数用于衡量数据的离散程度（稳定性），它是标准差与均值的比值。CV 值越低，代表信号越稳定。

**代码实现：**
```python
mu = np.mean(amp)      # 幅值均值
sigma = np.std(amp)    # 幅值标准差
cv = (sigma / mu) * 100
```

**数学公式：**
$$ CV = \frac{\sigma}{\mu} \times 100\% $$
其中：
*   $\sigma$ (Sigma)：CSI幅值的标准差。
*   $\mu$ (Mu)：CSI幅值的平均值。

---

### 2. 基于复平面 IQ 中心距离的区分度

这个指标用于衡量两个位置的信号在复平面（IQ Constellation）上是否离得够远，以及是否容易区分。它不仅看两个中心点离得远不远，还看它们各自的“胖瘦”（离散程度）。

**代码实现：**
```python
# c1, c2 是两个位置的复数均值点 (类聚中心)

dist = np.abs(c1 - c2)                 # 1. 计算中心距离
```

**数学公式：**

中心距离 (Dist):**
即两个复数均值点在复平面上的欧几里得距离。
设位置 A 的中心为 $C_A = I_A + jQ_A$，位置 B 的中心为 $C_B = I_B + jQ_B$：
$$ \text{Dist}_{AB} = |C_A - C_B| = \sqrt{(I_A - I_B)^2 + (Q_A - Q_B)^2} $$






============================================================
CSI 幅值区分度分析报告 - 子载波 Index 20
============================================================

[1. 各位置基础统计量]
位置 (Label)                | 均值 (Mean)  | 标准差 (Std)  | 变异系数 (CV, %)   
---------------------------------------------------------------------------
No Target                 | 19.0891     | 0.8944     | 4.69%
Position 1 (Left-Top)     | 20.0659     | 1.3600     | 6.78%
Position 2 (Right-Top)    | 19.0700     | 0.9321     | 4.89%
Position 3 (Bot-Right)    | 17.2882     | 0.7108     | 4.11%
Position 4 (Center)       | 17.7923     | 1.2581     | 7.07%

[2. 位置间区分度分析 (基于复平面IQ中心距离)]
对比组 (Pair)                               | 中心距离 (Dist)     
-----------------------------------------------------------------------------------------------
No Target vs Position 1 (Left-Top) | 0.9226          
No Target vs Position 2 (Right-Top) | 0.9403          
No Target vs Position 3 (Bot-Right) | 1.1021          
No Target vs Position 4 (Center) | 1.3065          
Position 1 (Left-Top) vs Position 2 (Right-Top) | 0.0177          
Position 1 (Left-Top) vs Position 3 (Bot-Right) | 1.6180          
Position 1 (Left-Top) vs Position 4 (Center) | 1.2879          
Position 2 (Right-Top) vs Position 3 (Bot-Right) | 1.6317          
Position 2 (Right-Top) vs Position 4 (Center) | 1.2946          
Position 3 (Bot-Right) vs Position 4 (Center) | 0.8010          

[3. 自动结论推断]
信号稳定性: [良好]。平均变异系数 CV 为 5.51%。数据存在一定波动，但均值特征依然稳定，可用于识别。
