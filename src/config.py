# src/config.py
import os

# --- 路径配置 ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # 项目根目录
DATA_DIR_RAW = os.path.join(BASE_DIR, "data", "JIRA") # 假设原始数据在 data/JIRA/
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "processed_data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

# 确保目录存在
os.makedirs(DATA_DIR_RAW, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# --- 数据预处理和特征选择参数 ---
TARGET_VARIABLE = 'RealBug'
POSITIVE_LABEL_IN_RAW_DATA = 1
NEGATIVE_LABEL_IN_RAW_DATA = 0
CORRELATION_THRESHOLD_TARGET = 0.05
MULTICOLLINEARITY_THRESHOLD = 0.8
FILE_COLUMN_TO_DROP = 'File' # 根据你的数据调整

# --- 实验参数 ---
RANDOM_SEED = 42
EVAL_COST_MATRIX = {(0,0):0, (0,1):1, (1,0):10, (1,1):0} # 论文中定义的成本矩阵
USE_CROSS_VALIDATION = True
N_SPLITS_CV = 5
N_REPEATS_CV = 1 # 为了快速演示设为1，论文中可能是2或更多
N_JOBS_PARALLEL = -1 # 使用所有可用核心

# 这些特征会在CV之前从完整数据集中移除 (如果存在)
# 在CV的每个fold内部，它们不应该被用作特征
POTENTIALLY_LEAKY_FEATURES = ['RealBugCount', 'HeuBug', 'HeuBugCount']


# --- 可视化参数 ---
DEFAULT_FIG_SIZE = (10, 6)
DEFAULT_PLOT_PALETTE = "viridis" # Seaborn调色板

# --- 统计检验 ---
STAT_TEST_LIBS_AVAILABLE = False # 会在 experiment_runner 中尝试导入并更新

# 更新包含经典模型的列表以进行统计比较
MODELS_FOR_STAT_TEST = [
    "CASB_Balanced",
    "CASB_BestF1",
    "CASB_BestRecCost",
    "XGBoost",
    "RUSBoost",
    "SMOTE + DT_Shallow", # 假设你还想比较这个
    # "ADASYN + DT_Shallow", # 和这个
    "LogisticRegression_Balanced",
    "NaiveBayes_Gaussian",
    "SVM_Balanced_RBF",
    "KNN_5", # 或你选择的K值对应的模型名
    "DecisionTree_Baseline_Balanced",
    "RandomForest_Baseline_Balanced" # 如果你决定加入RF
]

print(f"项目根目录: {BASE_DIR}")
print(f"原始数据目录: {DATA_DIR_RAW}")
print(f"处理后数据目录: {PROCESSED_DATA_DIR}")