# CostAdaptiveSamplerBoost (CASB) 软件缺陷预测模型

## 目录

* [项目描述](#项目描述)
* [主要特性](#主要特性)
* [文件结构](#文件结构)
* [安装依赖](#安装依赖)
* [数据集](#数据集)
* [使用方法](#使用方法)
  * [1. 数据预处理与特征选择](#1-数据预处理与特征选择)
  * [2. 运行实验](#2-运行实验)
  * [3. 生成可视化图表](#3-生成可视化图表)
* [实验结果](#实验结果)

## 项目描述

本项目实现 **CostAdaptiveSamplerBoost (CASB)** 算法——一种面向软件缺陷预测的代价敏感集成学习框架。CASB 基于 AdaBoost 框架改进，通过引入动态样本权重调整机制，有效应对类别不平衡与错分成本不对称的双重挑战，提升缺陷模块识别能力并降低总体预测成本。

本仓库包含完整的算法实现、实验评估框架（包含数据预处理、模型训练、性能评估、统计检验与可视化模块）及相关配置文件。本工作为论文《CostAdaptiveSamplerBoost: 一种用于软件缺陷预测的代价敏感集成学习方法》的配套代码库。

## 主要特性

* **创新算法实现**：完整的 CASB 分类器实现
* **多层次实验框架**：
  - 支持交叉验证与主流基模型对比（经典模型/标准集成/不平衡学习方法）
  - 参数敏感性分析与学习曲线可视化
* **多维评估体系**：
  - 常规指标：准确率、F1-score（少数类）、召回率（少数类）、G-mean
  - 平衡指标：Balanced Accuracy、AUC-PR
  - 成本指标：自定义模型总体成本
* **科学验证流程**：集成 Friedman + Nemenyi 统计检验
* **工程化代码结构**：模块化设计，高可扩展性
* **可复现性保障**：完整实验记录与自动化脚本

## 文件结构

```text
.
├── data/                  # 原始数据
│   └── JIRA/             # JIRA 缺陷数据集
│       ├── project1.csv
│       └── ...
├── processed_data/       # 预处理数据
├── results/              # 实验结果
│   ├── plots/            # 可视化图表
│   └── tables/           # 数据报表
├── src/                  # 源代码
│   ├── config.py         # 全局配置
│   ├── data_loader.py    # 数据加载
│   ├── preprocessing.py  # 数据预处理
│   ├── feature_selector.py # 特征选择
│   ├── models/           # 模型实现
│   ├── evaluate.py       # 评估指标
│   ├── experiment_runner.py # 实验运行
│   ├── visualization.py  # 可视化模块
│   ├── main_preprocess.py # 预处理入口
│   └── main_experiment.py # 实验入口
├── generate_plots.py     # 可视化生成脚本
├── requirements.txt      # 依赖清单
└── README.md             # 说明文档
```

## 安装依赖

**环境要求**：Python 3.8+，推荐使用虚拟环境

```bash
# 克隆仓库
git clone https://github.com/1fantasy1/lunwen.git
cd your_project

# 创建并激活虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

## 数据集

**数据准备**：
1. 将原始数据集（CSV格式）置于 `data/JIRA/` 目录
2. 默认目标变量列名配置于 `src/config.py` 的 `TARGET_VARIABLE` 参数

**预处理流程**：
- 自动合并多项目数据
- 执行特征编码、缺失值处理
- 完成相关性/多重共线性分析
- 输出路径：`processed_data/ALL_preprocessed.csv`

## 使用方法

### 1. 数据预处理与特征选择

```bash
python src/main_preprocess.py
```

**输出结果**：
- 预处理数据集 → `processed_data/`
- EDA 分析图表 → `results/plots/`

### 2. 运行实验

```bash
python src/main_experiment.py
```

**输出结果**：
- 详细评估结果 → `results/tables/`
- 性能对比图表 → `results/plots/`
- 终端输出统计检验结果

### 3. 生成可视化图表

```bash
# 基于已有结果重新生成图表
python generate_plots.py
```

**配置说明**：
- 需在脚本中指定结果文件前缀
- 学习曲线参数需与实验设置保持一致

## 实验结果

完整实验结果存放于 `results/` 目录：

| 文件类型        | 路径                 | 说明                |
|----------------|---------------------|--------------------|
| 原始评估数据    | tables/*.csv        | 逐折评估结果        |
| 聚合统计结果    | tables/*_agg.csv    | 指标均值/标准差     |
| 模型对比图      | plots/compare_*.png | 多指标横向对比      |
| 统计检验图      | plots/stats_*.png   | Friedman/Nemenyi 检验 |
| 学习曲线        | plots/learning_*.png| 参数敏感性分析      |

实验表明：CASB 在 F1-score（少数类）、召回率、G-mean 及预测总成本等关键指标上均显著优于对比基线方法。