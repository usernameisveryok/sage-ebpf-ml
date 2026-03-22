# eBPF/XDP 内核态机器学习推理调研报告

## 一、研究背景

在 Linux 内核中，eBPF (extended Berkeley Packet Filter) 是一种内核虚拟机，允许用户在内核态运行沙箱化的程序。结合 XDP (eXpress Data Path)，可以在网卡驱动层（Linux 网络栈的最早入口点）对报文进行极低延迟的处理。

**核心目标**：在 XDP hook 上实现小型 ML 模型推理，从报文序列中提取特征，对报文/流进行实时分类（如入侵检测、DDoS 识别）。

---

## 二、eBPF 的关键限制

在 eBPF 中实现 ML 推理面临以下硬性约束：

| 限制 | 详情 |
|------|------|
| **无浮点运算** | eBPF 不支持浮点类型，必须使用整数或定点数运算 |
| **指令数上限** | 单个 eBPF 程序最多 100 万条指令（verifier 限制） |
| **栈空间限制** | 每个 eBPF 程序栈空间仅 512 字节；使用 tail call 时每个子程序仅 256 字节 |
| **Tail Call 链深度** | 最大 33 层（硬限制 32M 指令） |
| **无动态内存分配** | 无 malloc，只能使用 BPF Map 和栈上变量 |
| **有限循环** | 仅支持有界循环（Linux 5.3+），无法使用任意长度循环 |
| **无内核函数随意调用** | 只能调用白名单中的 BPF Helper 函数 |
| **无除法符号** | eBPF 不直接支持有符号除法 |

---

## 三、现有工作综述

### 3.1 决策树 (Decision Tree) 方案

**论文**: Bachl et al., "A flow-based IDS using Machine Learning in eBPF" (2021)
- **方法**: 在 eBPF 中实现深度为 10、最多 1000 叶子节点的决策树
- **数据集**: CIC-IDS-2017，准确率 ~99%
- **性能**: eBPF 实现比用户态快 20%+（~152K pps vs ~125K pps）
- **定点数**: 使用 64 位有符号整数，16 位定点小数
- **优势**: 结构简单，无需浮点运算，天然适合 eBPF
- **劣势**: 
  - 存储开销大（最坏情况 O(2^n) 节点）
  - 热更新困难（树结构不固定）
  - 每次 eBPF Map 查询有额外开销
- **开源**: https://github.com/Maximilian-Bachl/machine-learning-in-ebpf

### 3.2 知识蒸馏 + 决策树方案

**论文**: Chen et al., "Identifying DDoS Attacks in-Kernel via eBPF/XDP and Knowledge Distillation" (ICCPR 2025)
- **方法**: 用复杂 MLP 教师模型蒸馏出轻量决策树学生模型
- **结果**: Macro F1 = 97.6%，比基线提升 1.1%
- **亮点**: 知识蒸馏弥补了简单 DT 的表达能力不足

### 3.3 神经网络 int8 量化方案

**论文**: Hara et al., "On Practicality of Kernel Packet Processing Empowered by Lightweight Neural Network and Decision Tree" (NoF 2023)
- **方法**: 3 层 NN（44 节点），float32 量化到 int8
- **性能**: 内核推理比用户态快 84%
- **问题**: int8 精度严重不足，量化误差大，性能下降明显

### 3.4 ⭐ 神经网络 int32 量化方案（最优方案）

**论文**: Zhang et al., "Real-Time Intrusion Detection and Prevention with Neural Network in Kernel using eBPF" (DSN 2024)
- **开源**: https://github.com/IntelligentDDS/NN-eBPF
- **方法**: 
  - MLP 结构 [6, 32, 32, 2]（输入 6 特征 → 两层隐藏层 → 二分类）
  - **放大法 (Enlargement Method)**: 将浮点参数乘以 s=2^16，转为 int32 存储
  - 用右移 b 位代替除以 s（s=2^b），极大加速推理
  - **Chained Tail Call**: 将 Linear + ReLU 拆分为独立 eBPF 程序，通过 tail call 链式调用，突破单程序 100 万指令限制
  - **无锁热更新**: 双缓冲机制（Running/Idle），通过切换索引实现线程安全的参数更新
- **关键参数**:
  - 放大因子 s = 2^16（参数和输入乘以 65536 后存入 int32，不溢出）
  - 每层输出最大 64 个元素（256 字节栈 / 4 字节 = 64）
  - 最大 17 层（tail call 深度限制）
- **特征提取**: 6 个 Top-K 特征（基于 Gini 增益选择）
  - Fwd Packet Length Max/Min
  - Fwd IAT Max
  - Destination Port
  - Fwd Header Length
  - Total Fwd Packets
- **性能**:
  - F1-score: 0.933 (CIC-IDS-2017) / 0.992 (在线复现数据集)
  - 推理时间: 3000-5000 ns/flow
  - 特征提取时间: ~200 ns/packet
  - 模型内存: 仅 5KB
  - CPU 开销: 0.54%-1.64%
- **优于 DT 的地方**: 固定结构、低内存、热更新友好

### 3.5 动态定点数方案

**论文**: Osaki et al., "Dynamic Fixed-point Values in eBPF: a Case for Fully In-kernel Anomaly Detection" (AINTEC 2024)
- **方法**: 
  - 40 位动态定点数（32 位数值 + 8 位位分配信息）
  - 位分配动态调整以防止溢出并最大化精度
  - 基于快速熵 (Fast Entropy) 的异常检测
- **性能**: 比用户态 + eBPF 方案吞吐量高 18%，内存密集场景高 55%
- **检测精度**: 动态定点数 ≈ 浮点数精度
- **指令数**: 整个框架仅 762 条指令（极轻量）
- **观点**: 作者认为 ML 模型不适合高速 eBPF 处理（100K-800K pps），统计方法更适合（11M pps）

### 3.6 其他相关工作

- **SmartX Intelligent Sec** (2024): eBPF/XDP 安全框架，结合 ML
- **PF-eBPF** (2025): 结合包级和流级特征的分类模型
- **O2C** (2024): 在 eBPF 中嵌入决策树进行内核隔离

---

## 四、技术路线对比

| 方案 | 模型 | 精度 | 推理速度 | 内存 | 实现复杂度 | 可扩展性 |
|------|------|------|----------|------|-----------|---------|
| Decision Tree | DT (depth=10) | ★★★★★ | ★★★☆☆ | ★★☆☆☆ (32KB+) | ★★★★★ | ★★☆☆☆ |
| NN-int8 | MLP 3层 | ★★☆☆☆ | ★★★★☆ | ★★★★☆ | ★★★☆☆ | ★★★☆☆ |
| **NN-int32** | **MLP 3层** | **★★★★☆** | **★★★★☆** | **★★★★★ (5KB)** | **★★★☆☆** | **★★★★☆** |
| 动态定点+统计 | 熵统计 | ★★★☆☆ | ★★★★★ | ★★★★★ | ★★★★☆ | ★★☆☆☆ |
| KD + DT | 蒸馏DT | ★★★★★ | ★★★★☆ | ★★★☆☆ | ★★☆☆☆ | ★★★☆☆ |

---

## 五、推荐方案

基于调研，推荐实现 **NN-int32（放大法量化 MLP + Tail Call 链）** 方案，原因：

1. **精度最佳平衡**: int32 放大法几乎无量化精度损失（s=2^16 时 ECDF < 0.5%）
2. **固定结构**: MLP 结构固定，便于 eBPF 实现和热更新
3. **低内存**: 5KB 模型参数，远低于 DT 的 32KB+
4. **已验证可行**: DSN 2024 顶会论文，有开源参考实现
5. **可扩展**: tail call 链最多支持 17 层，隐藏层最大 64 个神经元

---

## 六、实现计划

### Phase 1: 用户态训练
- 使用 Python/PyTorch 训练 MLP 模型
- 生成合成/使用公开数据集
- 量化参数（放大法，s=2^16）
- 导出量化后的 int32 参数为 C 头文件

### Phase 2: eBPF/XDP 内核态推理
- XDP 程序：报文解析 → 特征提取 → 流特征更新
- Tail Call 链：Linear Layer → ReLU → Linear Layer → ReLU → Output
- BPF Map 存储：流特征、模型参数、推理结果
- 分类决策：比较输出层两个值，决定 XDP_PASS 或 XDP_DROP

### Phase 3: 用户态控制器
- 加载 eBPF 程序
- 参数热更新（双缓冲无锁）
- 监控和统计

---

## 七、参考文献

1. Bachl et al., "A flow-based IDS using Machine Learning in eBPF", arXiv:2102.09980, 2021
2. Hara et al., "On Practicality of Kernel Packet Processing Empowered by Lightweight NN and DT", NoF 2023
3. Zhang et al., "Real-Time Intrusion Detection and Prevention with NN in Kernel using eBPF", DSN 2024
4. Osaki et al., "Dynamic Fixed-point Values in eBPF: A Case for Fully In-kernel Anomaly Detection", AINTEC 2024
5. Chen et al., "Identifying DDoS Attacks in-Kernel via eBPF/XDP and Knowledge Distillation", ICCPR 2025
6. Wang et al., "When eBPF Meets Machine Learning: On-the-fly OS Kernel Compartmentalization", arXiv:2401.05641
