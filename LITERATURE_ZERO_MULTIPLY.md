# 零乘法推理 + LUT消除乘法：文献调研

## 一、零乘法推理的主要路线

| 路线 | 代表工作 | 核心思路 | 乘法替代物 |
|------|----------|----------|------------|
| **Binary/Ternary** | TWN(2016), BinaryConnect(2015) | 权重∈{-1,0,+1} | add/sub |
| **AdderNet** | Chen et al., CVPR 2020 | 用L1距离替代点积 | |x-w| 绝对差 |
| **DeepShift** | Elhoushi et al., 2020 | 权重=±2^k | bit-shift |
| **ShiftAddNet** | You et al., NeurIPS 2020 | 混合shift层+add层 | shift+add |
| **LUT-NN** | Tang et al., Microsoft MobiCom 2023 | 聚类中心+预计算查表 | map lookup |
| **TBNN** | Xu et al., 2025 | P4交换机上BNN用查找表 | match-action table |
| **P4-BNN** | Luo et al., 2022 | P4数据面二值网络 | shift+整数运算 |

## 二、最相关的工作

### 2.1 P4-BNN / TBNN（数据面神经网络）
- **平台**: P4 可编程交换机（Tofino / SmartNIC）
- **思路**: 二值权重(±1) → XNOR + popcount，或预计算部分和存match-action表
- **局限**: P4 match-action table 是 TCAM 硬件，静态配置，不支持运行时更新
- **与我们的区别**: 我们用 eBPF map（内核内存），支持原子更新 → 模型热更新

### 2.2 LUT-NN（Microsoft, MobiCom 2023）
- **思路**: 学习聚类中心(centroid)，预计算每个聚类对应的输出，推理时只需：
  1. 找最近聚类中心（距离计算）
  2. 查表取结果（零乘法）
- **效果**: 比原始推理快66%-92%，精度接近
- **局限**: 面向CPU/ARM，聚类搜索本身仍需计算
- **与我们的区别**: 我们的场景更极端——BPF 512字节栈，无动态内存，需确定性执行

### 2.3 DeepShift / ShiftAddNet
- **思路**: 每个权重 = sign × 2^shift，乘法变移位
- **效果**: ResNet-20 on CIFAR-10 达 91%+，零乘法
- **与我们的区别**: 他们需要per-weight不同shift值，在BPF中变量移位和乘法一样开销

## 三、eBPF Map LUT 算子的独特价值

### 核心创新：用 BPF_MAP_TYPE_ARRAY 做预计算查找表

**原理：**
```
传统:  h[i] = Σ_j  W[i][j] × x[j]     ← N×H 次乘法
LUT:   h[i] = Σ_j  LUT[j][x_q[j]][i]   ← N 次 map lookup + N×H 次加法
```

- 离线预计算: `LUT[j][v] = W[:, j] * v`  对所有特征j、所有可能值v
- 运行时: 输入量化到离散值 → 查表 → 累加 → **零乘法**

**与已有工作的关键区别：**

| 维度 | P4-BNN/TBNN | LUT-NN | **Ours (BPF Map LUT)** |
|------|-------------|--------|------------------------|
| 平台 | P4交换机 TCAM | CPU/ARM | **eBPF/XDP 内核** |
| 存储介质 | match-action table | L1/L2 cache | **BPF array map (per-CPU)** |
| 模型更新 | 需重新配置流表 | 需重启进程 | **原子map更新，零停机** |
| 约束 | 流水线级数限制 | 无特殊约束 | **512B栈, 1M验证器, 确定性** |
| 精度控制 | 仅binary(1-bit) | centroid-based | **可调bin数(8-256)** |

### 存储估算

对 MLP [12→64→64→4]，Layer 0 (12个特征):
- 每特征量化到 B 个 bin
- 每个 bin 预计算 64 维部分和 (int32)
- 存储: 12 × B × 64 × 4 bytes

| Bin数 | 存储 | 精度损失（估计） |
|-------|------|-----------------|
| 32    | 96 KB | ~2-5% |
| 64    | 192 KB | ~1-2% |
| 128   | 384 KB | <1% |
| 256   | 768 KB | ~0 (8-bit等效) |

BPF map 上限: 内核可配置，默认单map最大值通常足够。
Per-CPU array map 无锁访问 → 查表零竞争。

### 三重创新叙事（升级版）

1. **LUT-Map 零乘法算子** (NEW - 本项目核心创新)
   - Layer 0: 预计算查找表存BPF map，输入量化→查表→累加，零乘法
   - 支持运行时模型热更新（userspace写新LUT到map）
   
2. **三值移位加算子** (已实现)
   - Layer 1/2: 权重∈{-α,0,+α}，α=2^k，点积=条件加减+移位
   
3. **Early-Exit 尾调用旁路** (已实现)
   - Layer 0后轻量级exit head判断置信度，85%流量跳过后续层

**最终效果: 整条推理管线零乘法指令。**

## 四、关键参考文献

1. Chen et al., "AdderNet: Do We Really Need Multiplications in Deep Learning?" CVPR 2020
2. You et al., "ShiftAddNet: A Hardware-Inspired Deep Network" NeurIPS 2020
3. Tang et al., "LUT-NN: Empower Efficient NN Inference with Centroid Learning and Table Lookup" MobiCom 2023
4. Luo et al., "P4-BNN: Binary Neural Network with P4 on Programmable Data Plane" 2022
5. Xu et al., "TBNN: Lookup Tables-Based Optimization for in-Network Binary Neural Networks" 2025
6. Elhoushi et al., "DeepShift: Towards Multiplication-Less Neural Networks" 2020
7. Aussie AI, "Zero Multiplication Inference Algorithms" (survey) 2026
