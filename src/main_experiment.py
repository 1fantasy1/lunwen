# src/main_experiment.py
import pandas as pd
import numpy as np
import os
from src.config import (PROCESSED_DATA_DIR, TARGET_VARIABLE, RANDOM_SEED, TABLES_DIR,
                        MODELS_FOR_STAT_TEST,
                        USE_CROSS_VALIDATION)  # N_SPLITS_CV, N_REPEATS_CV removed, taken from experiment_runner if needed

# 从 models.base_models 导入所有需要的函数
from src.models.base_models import get_model_blueprints, get_sampler_based_models, get_models_for_parameter_scan

from src.experiment_runner import run_experiment, aggregate_cv_results, perform_statistical_tests
from src.visualization import plot_model_performance_comparison, plot_learning_curves_generic


def main():
    print("--- 开始模型训练和评估实验 ---")

    # 1. 加载预处理后的数据
    preprocessed_data_path = os.path.join(PROCESSED_DATA_DIR,
                                          "ALL_preprocessed.csv")  # Or "ALL.csv" from your old script
    if not os.path.exists(preprocessed_data_path):
        print(f"错误: 预处理后的数据文件 {preprocessed_data_path} 未找到。")
        print("请先运行 main_preprocess.py。")
        return

    try:
        data_for_experiment = pd.read_csv(preprocessed_data_path)
        print(f"成功加载预处理数据: {preprocessed_data_path}, 形状: {data_for_experiment.shape}")
    except Exception as e:
        print(f"加载预处理数据 {preprocessed_data_path} 时出错: {e}")
        return

    if TARGET_VARIABLE not in data_for_experiment.columns:
        print(f"错误: 目标变量 '{TARGET_VARIABLE}' 未在加载的数据中找到。")
        return
    try:
        data_for_experiment[TARGET_VARIABLE] = data_for_experiment[TARGET_VARIABLE].astype(int)
    except ValueError:
        print(f"无法将目标变量 '{TARGET_VARIABLE}' 转换为整数。请检查预处理数据。")
        return

    # 2. 获取模型蓝图
    # 2.1 常规模型
    model_blueprints = get_model_blueprints(random_state_seed=RANDOM_SEED)
    print(f"加载了 {len(model_blueprints)} 个常规模型配置。")

    # 2.2 用于采样器（SMOTE/ADASYN）之后的模型配置
    sampler_model_configs = get_sampler_based_models(random_state_seed=RANDOM_SEED)

    # 2.3 (可选) 为参数扫描生成模型配置 (用于学习曲线)
    # ------------------------------------------------------------------
    # --- 如何使用学习曲线功能 ---
    # 1. 设计参数扫描实验：
    #    选择一个你想研究其影响的CASB超参数（例如 'n_estimators', 'learning_rate', 'fn_cost', 'max_depth'）。
    #    选择一系列该超参数的值。
    # 2. 使用 get_models_for_parameter_scan 生成模型:
    #    为每个要扫描的参数调用此函数。它会将其他参数固定。
    #    将返回的字典合并到主 model_blueprints 字典中。
    # 3. 运行实验:
    #    实验运行器将像处理其他模型一样处理这些扫描模型。
    # 4. 绘制学习曲线:
    #    在结果出来后，使用 plot_learning_curves_generic 函数。
    #    你需要提供一个正则表达式 (model_name_pattern) 来从模型名称中提取变化的参数值。
    #
    # 示例：扫描 CASB 的 n_estimators 参数
    # ------------------------------------------------------------------

    # 定义固定参数，除了要扫描的那个
    casb_base_config_for_scan = {
        'base_estimator_depth': 3,  # 例如，固定决策树深度
        'learning_rate': 0.10,  # 固定学习率
        'fn_cost': 10,  # 固定FN成本
        'n_estimators': 100  # 这个值会被 scan_values 中的值覆盖 (如果 param_to_scan 是 'n_estimators')
    }
    n_estimators_to_scan = [50, 100, 150, 200]  # 要测试的 n_estimators 值

    # 生成 n_estimators 扫描的模型
    scanned_n_estimators_models = get_models_for_parameter_scan(
        param_to_scan='n_estimators',
        scan_values=n_estimators_to_scan,
        base_model_config=casb_base_config_for_scan,
        random_state_seed=RANDOM_SEED
    )
    model_blueprints.update(scanned_n_estimators_models)  # 添加到主模型字典
    print(f"添加了 {len(scanned_n_estimators_models)} 个用于 n_estimators 扫描的模型配置。")

    # 示例：扫描 CASB 的 fn_cost 参数 (保持其他参数，如n_estimators，固定)
    casb_base_config_for_fn_scan = {
        'base_estimator_depth': 3,
        'learning_rate': 0.10,
        'n_estimators': 100,  # 固定 n_estimators
        'fn_cost': 10  # 这个值会被 scan_values 中的值覆盖
    }
    fn_costs_to_scan = [5, 8, 10, 12, 15]
    scanned_fn_cost_models = get_models_for_parameter_scan(
        param_to_scan='fn_cost',
        scan_values=fn_costs_to_scan,
        base_model_config=casb_base_config_for_fn_scan,
        random_state_seed=RANDOM_SEED
    )
    model_blueprints.update(scanned_fn_cost_models)
    print(f"添加了 {len(scanned_fn_cost_models)} 个用于 fn_cost 扫描的模型配置。")

    print(f"总共将评估 {len(model_blueprints)} 个模型配置 (包括扫描模型)。")
    # ------------------------------------------------------------------
    # --- 学习曲线功能说明结束 ---
    # ------------------------------------------------------------------

    # 3. 运行实验
    dataset_basename = os.path.basename(preprocessed_data_path).split('.')[0]
    all_results_df = run_experiment(data_for_experiment, dataset_basename, model_blueprints, sampler_model_configs)

    if all_results_df.empty:
        print("实验未产生任何结果。")
        return

    # --- 4. 显示和保存结果 ---
    # (结果处理部分与你之前的代码类似)
    pd.set_option('display.max_rows', None);
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200);
    pd.set_option('display.float_format', '{:.4f}'.format)

    errors_df = all_results_df[all_results_df['Error'].notna()] if 'Error' in all_results_df.columns else pd.DataFrame()
    successful_results_df = all_results_df[all_results_df['Error'].isna()] if 'Error' in all_results_df.columns else all_results_df.copy()
    print("\n--- Debugging: successful_results_df ---")
    if not successful_results_df.empty:
        print(f"Shape: {successful_results_df.shape}")
        print("Columns:", successful_results_df.columns.tolist())
        print("Data types:\n", successful_results_df.dtypes)
        print("Head:\n", successful_results_df.head().to_string())

        # 检查关键指标列的NaN值统计
        metrics_to_check_nan = ["Accuracy", "F1-score (minority)", "Recall (minority)", "G-mean", "Balanced Accuracy",
                                "AUC-PR", "Total Model Cost"]
        for metric in metrics_to_check_nan:
            if metric in successful_results_df.columns:
                print(
                    f"NaN count in '{metric}': {successful_results_df[metric].isnull().sum()} / {len(successful_results_df)}")
                # 打印一些非NaN的值，看看它们看起来是否正常
                print(f"Non-NaN sample values for '{metric}': {successful_results_df[metric].dropna().head().tolist()}")
            else:
                print(f"Metric '{metric}' not found in successful_results_df.")

        # 检查 'Model' 列的唯一值
        if 'Model' in successful_results_df.columns:
            print("Unique models in successful_results_df:", successful_results_df['Model'].unique())
    else:
        print("successful_results_df is empty.")
    print("--- End Debugging: successful_results_df ---\n")

    if not errors_df.empty:
        print("\n\n--- 遇到的错误 ---")
        # print(errors_df[['Dataset', 'Model', 'Split', 'Error']].to_string(index=False)) # 详细错误
        errors_summary = errors_df.groupby(['Model', 'Error']).size().reset_index(name='Count')
        print(errors_summary.to_string(index=False))
        errors_output_path = os.path.join(TABLES_DIR, f"{dataset_basename}_errors.csv")
        errors_df.to_csv(errors_output_path, index=False)
        print(f"错误详情已保存到: {errors_output_path}")

    if not successful_results_df.empty:
        print("\n\n--- 成功的运行结果 (部分展示前5条) ---")
        print(successful_results_df.drop(columns=['Error', 'Traceback'], errors='ignore').head().to_string(index=False))

        successful_runs_output_path = os.path.join(TABLES_DIR, f"{dataset_basename}_successful_fold_results.csv")
        successful_results_df.drop(columns=['Error', 'Traceback'], errors='ignore').to_csv(successful_runs_output_path,
                                                                                           index=False)
        print(f"所有成功的Fold结果已保存到: {successful_runs_output_path}")

        aggregated_df = pd.DataFrame()
        if USE_CROSS_VALIDATION and 'Split' in successful_results_df.columns and successful_results_df[
            'Split'].nunique() > 1:
            aggregated_df = aggregate_cv_results(successful_results_df.copy())
            if not aggregated_df.empty:
                print("\n\n--- 聚合交叉验证结果 (Mean ± Std) ---")
                print(aggregated_df.to_string(index=False))
                agg_output_path = os.path.join(TABLES_DIR, f"{dataset_basename}_aggregated_cv_results.csv")
                aggregated_df.to_csv(agg_output_path, index=False)
                print(f"聚合CV结果已保存到: {agg_output_path}")

                # 可视化聚合性能 (需要temp_agg_for_plot的逻辑)
                metrics_for_plot_df = ["Accuracy", "F1-score (minority)", "Recall (minority)", "G-mean",
                                       "Balanced Accuracy", "AUC-PR", "Total Model Cost"]
                temp_agg_for_plot = successful_results_df.groupby(['Dataset', 'Model']).agg(
                    **{f'{col}_mean': pd.NamedAgg(column=col, aggfunc='mean') for col in metrics_for_plot_df}
                ).reset_index()
                if not temp_agg_for_plot.empty:
                    plot_model_performance_comparison(
                        temp_agg_for_plot,
                        metrics_cols_mean=[f'{m}_mean' for m in metrics_for_plot_df],
                        model_col='Model'
                    )
            else:
                print("聚合结果为空，无法绘制模型性能比较图。")

        # 统计检验
        if USE_CROSS_VALIDATION and not aggregated_df.empty:
            perform_statistical_tests(successful_results_df.copy(), MODELS_FOR_STAT_TEST,
                                      metric_to_test="F1-score (minority)")

        # --- 绘制学习曲线 ---
        # 确保 successful_results_df 包含用于绘图的扫描模型结果
        if not successful_results_df.empty:
            # 学习曲线 for n_estimators scan
            # 正则表达式解释:
            # CASB_Scan_N_ESTIMATORS_V  -> 匹配固定前缀
            # N(\d+)                     -> 捕获 "N" 后面的一个或多个数字 (n_estimators 的值)
            # _LR..._FN..._D...          -> 匹配描述固定参数的部分
            plot_learning_curves_generic(
                successful_results_df,
                model_name_pattern=r"CASB_Scan_N_ESTIMATORS_VN(\d+)_LR\d\.\d+_FN\d+_D\d+",  # 匹配生成的扫描模型名称
                varying_param_char='N_Estimators',  # X轴标签
                fixed_params_str=f"LR{casb_base_config_for_scan['learning_rate']:.2f}_FN{casb_base_config_for_scan['fn_cost']}_D{casb_base_config_for_scan['base_estimator_depth']}",
                # 图标题
                score_to_plot="Recall (minority)",  # Y轴指标
                dataset_name=dataset_basename
            )

            # 学习曲线 for fn_cost scan
            plot_learning_curves_generic(
                successful_results_df,
                model_name_pattern=r"CASB_Scan_FN_COST_VFN(\d+)_LR\d\.\d+_N\d+_D\d+",
                varying_param_char='FN_Cost',
                fixed_params_str=f"LR{casb_base_config_for_fn_scan['learning_rate']:.2f}_N{casb_base_config_for_fn_scan['n_estimators']}_D{casb_base_config_for_fn_scan['base_estimator_depth']}",
                score_to_plot="F1-score (minority)",
                dataset_name=dataset_basename
            )
        else:
            print("没有成功的结果用于绘制学习曲线。")

    else:
        print("没有成功的运行结果可以显示。")

    print("\n--- 模型训练和评估实验完成 ---")


if __name__ == "__main__":
    main()