# src/models/base_models.py
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB  # 或者 BernoulliNB, MultinomialNB 根据特征类型选择
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from imblearn.ensemble import RUSBoostClassifier
from sklearn.base import clone

from src.config import RANDOM_SEED
# 确保 casb.py 在 src/models/ 目录下
from .casb import CostAdaptiveSamplerBoost


def get_model_blueprints(random_state_seed=RANDOM_SEED):
    """
    返回一个包含所有模型蓝图的字典，用于实验。
    包括经典模型、集成基线和CASB变体。
    """
    models = {}

    # --- 辅助决策树实例 ---
    dt_shallow = DecisionTreeClassifier(max_depth=3, random_state=random_state_seed)
    dt_medium_depth = DecisionTreeClassifier(max_depth=5, random_state=random_state_seed)  # 用于CASB的较深树

    # --- 1. 经典机器学习模型 ---
    models["LogisticRegression_Balanced"] = LogisticRegression(
        random_state=random_state_seed,
        class_weight='balanced',
        solver='liblinear',  # 适合中小型数据集，支持L1/L2正则
        max_iter=1000  # 增加迭代次数以确保收敛
    )
    # 注意: GaussianNB 假设特征是高斯分布的（连续的）。
    # 如果你的特征是二元的或计数的，请考虑 BernoulliNB 或 MultinomialNB。
    # 为了通用性，这里使用 GaussianNB。
    models["NaiveBayes_Gaussian"] = GaussianNB()

    models["SVM_Balanced_RBF"] = SVC(
        random_state=random_state_seed,
        class_weight='balanced',
        probability=True,  # 必须为 True 才能使用 predict_proba
        kernel='rbf'  # 径向基核函数，常用且效果较好
    )
    # K值的选择对KNN很重要。5是一个常见的起始点。
    # 可以在论文中说明K值是如何选择的（例如，基于文献或小的预实验）。
    models["KNN_5"] = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)  # 使用所有核心

    models["DecisionTree_Baseline_Balanced"] = DecisionTreeClassifier(
        random_state=random_state_seed,
        class_weight='balanced',
        max_depth=None  # 允许树完全生长，或设置一个合理的深度如10
    )

    models["RandomForest_Baseline_Balanced"] = RandomForestClassifier(
        n_estimators=100,  # 常用值
        random_state=random_state_seed,
        class_weight='balanced',  # 或 'balanced_subsample'
        n_jobs=-1  # 使用所有核心
    )

    # --- 2. 标准集成基线模型 ---
    models["AdaBoost_SAMME"] = AdaBoostClassifier(
        estimator=clone(dt_shallow),  # 使用浅层决策树作为基学习器
        n_estimators=150,  # 与XGBoost/RUSBoost保持一致
        algorithm='SAMME',  # SAMME算法允许predict_proba（如果基学习器支持）
        random_state=random_state_seed
    )
    models["XGBoost"] = XGBClassifier(
        n_estimators=150,
        random_state=random_state_seed,
        eval_metric='logloss',  # 常用评估指标
        n_jobs=-1  # 使用所有核心
    )
    models["RUSBoost"] = RUSBoostClassifier(
        estimator=clone(dt_shallow),  # 使用浅层决策树
        n_estimators=150,
        random_state=random_state_seed
    )

    # --- 3. CASB 变体 (来自你的研究) ---
    models["CASB_BestF1"] = CostAdaptiveSamplerBoost(
        base_estimator=clone(dt_medium_depth),  # 使用中等深度的树
        n_estimators=150,
        learning_rate=0.05,
        cost_matrix={(0, 0): 0, (0, 1): 1, (1, 0): 10, (1, 1): 0},
        random_state=random_state_seed
    )
    models["CASB_BestRecCost"] = CostAdaptiveSamplerBoost(
        base_estimator=clone(dt_medium_depth),
        n_estimators=50,
        learning_rate=0.10,
        cost_matrix={(0, 0): 0, (0, 1): 1, (1, 0): 10, (1, 1): 0},
        random_state=random_state_seed
    )
    models["CASB_Balanced"] = CostAdaptiveSamplerBoost(
        base_estimator=clone(dt_medium_depth),
        n_estimators=150,
        learning_rate=0.05,
        cost_matrix={(0, 0): 0, (0, 1): 1, (1, 0): 12, (1, 1): 0},
        random_state=random_state_seed
    )
    models["CASB_StrongAlt"] = CostAdaptiveSamplerBoost(
        base_estimator=clone(dt_shallow),  # 使用浅层树
        n_estimators=150,
        learning_rate=0.10,
        cost_matrix={(0, 0): 0, (0, 1): 1, (1, 0): 10, (1, 1): 0},
        random_state=random_state_seed
    )
    # 消融研究：CASB不使用特定成本（类似于SAMME的对称成本）
    models["CASB_Ablation_NoCost"] = CostAdaptiveSamplerBoost(
        base_estimator=clone(dt_shallow),
        n_estimators=150,  # 与其他集成模型一致
        learning_rate=1.0,  # AdaBoost默认学习率
        cost_matrix={(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0},  # 对称成本
        random_state=random_state_seed
    )

    return models


def get_sampler_based_models(random_state_seed=RANDOM_SEED):
    """
    返回一个字典，其中键是采样后模型的名称后缀，值是分类器蓝图。
    这些模型会在数据经过SMOTE/ADASYN等采样器处理后再进行训练。
    """
    sampler_models_config = {}
    # 这是06.ipynb中SMOTE/ADASYN之后使用的决策树
    dt_base_for_sampling = DecisionTreeClassifier(max_depth=3, random_state=random_state_seed)

    # 键名 "DT_Shallow_After_Sampling" 用于在 experiment_runner 中查找这个分类器
    sampler_models_config["DT_Shallow_After_Sampling"] = dt_base_for_sampling

    return sampler_models_config


def get_models_for_parameter_scan(param_to_scan, scan_values, base_model_config, model_prefix="CASB_Scan",
                                  random_state_seed=RANDOM_SEED):
    """
    为特定参数生成一系列模型配置，用于参数扫描和绘制学习曲线。

    参数:
    param_to_scan (str): 要扫描的CASB超参数的名称 (例如 'n_estimators', 'learning_rate', 'fn_cost', 'max_depth').
    scan_values (list): 要测试的参数值的列表。
    base_model_config (dict): 包含CASB其他固定参数的字典。
                              例如: {'base_estimator_depth': 5, 'n_estimators': 150,
                                     'learning_rate': 0.05, 'fn_cost': 10}
                                     要扫描的参数在base_model_config中的值将被忽略。
    model_prefix (str): 生成的模型名称的前缀。
    random_state_seed (int): 随机种子。

    返回:
    dict: 模型名称到模型实例的字典。
    """
    scanned_models = {}

    # 从 base_model_config 获取默认值，如果未提供则使用合理的回退值
    default_lr = base_model_config.get('learning_rate', 0.1)
    default_n_est = base_model_config.get('n_estimators', 100)
    default_fn_cost = base_model_config.get('fn_cost', 10)
    default_base_depth = base_model_config.get('base_estimator_depth', 3)

    for val in scan_values:
        current_params_dict = {  # 使用字典来存储当前迭代的参数
            'learning_rate': default_lr,
            'n_estimators': default_n_est,
            'fn_cost': default_fn_cost,
            'base_estimator_depth': default_base_depth
        }

        varying_param_name_val_str = ""  # 例如 "N100" 或 "LR0.05"

        if param_to_scan == 'n_estimators':
            current_params_dict['n_estimators'] = int(val)
            varying_param_name_val_str = f"N{int(val)}"
        elif param_to_scan == 'learning_rate':
            current_params_dict['learning_rate'] = float(val)
            varying_param_name_val_str = f"LR{float(val):.2f}"  # .2f 确保两位小数
        elif param_to_scan == 'fn_cost':
            current_params_dict['fn_cost'] = int(val)
            varying_param_name_val_str = f"FN{int(val)}"
        elif param_to_scan == 'max_depth':  # 指的是 base_estimator 的 max_depth
            current_params_dict['base_estimator_depth'] = int(val)
            varying_param_name_val_str = f"D{int(val)}"
        else:
            print(f"警告: 未知的扫描参数 '{param_to_scan}'。跳过值 {val}。")
            continue

        # 构建固定参数部分的名称字符串
        fixed_parts_name_list = []
        if param_to_scan != 'learning_rate': fixed_parts_name_list.append(
            f"LR{current_params_dict['learning_rate']:.2f}")
        if param_to_scan != 'fn_cost': fixed_parts_name_list.append(f"FN{current_params_dict['fn_cost']}")
        if param_to_scan != 'n_estimators': fixed_parts_name_list.append(f"N{current_params_dict['n_estimators']}")
        if param_to_scan != 'max_depth': fixed_parts_name_list.append(f"D{current_params_dict['base_estimator_depth']}")

        # 完整的模型名称: 前缀_扫描参数大写_V变化值_固定参数摘要
        model_name = f"{model_prefix}_{param_to_scan.upper()}_V{varying_param_name_val_str}_{'_'.join(fixed_parts_name_list)}"

        # 创建基学习器
        dt_base_for_scan = DecisionTreeClassifier(max_depth=current_params_dict['base_estimator_depth'],
                                                  random_state=random_state_seed)

        # 创建成本矩阵
        cost_matrix_for_scan = {(0, 0): 0, (0, 1): 1, (1, 0): current_params_dict['fn_cost'], (1, 1): 0}

        scanned_models[model_name] = CostAdaptiveSamplerBoost(
            base_estimator=clone(dt_base_for_scan),
            n_estimators=current_params_dict['n_estimators'],
            learning_rate=current_params_dict['learning_rate'],
            cost_matrix=cost_matrix_for_scan,
            random_state=random_state_seed
        )

    return scanned_models