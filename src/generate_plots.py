# generate_plots.py
import pandas as pd
import os
import sys
import re  # 确保导入 re

# --- 将 src 目录添加到 Python 路径 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from config import TABLES_DIR, PLOTS_DIR, TARGET_VARIABLE
    from visualization import (
        plot_model_performance_comparison,
        plot_learning_curves_generic,
        # plot_target_distribution,
        # plot_feature_target_correlation,
        # plot_correlation_heatmap
    )
except ImportError as e:
    print(f"导入模块时出错: {e}\nEnsure generate_plots.py is in the project root or src, or adjust sys.path.")
    exit()


def main_plotting():
    print("--- 开始独立生成图表 ---")

    dataset_basename = "ALL_preprocessed"  # !! 修改为你实际的文件名前缀 !!
    successful_fold_results_file = os.path.join(TABLES_DIR, f"{dataset_basename}_successful_fold_results.csv")

    # --- 调试点 1: 检查 successful_fold_results_file 是否存在 ---
    if not os.path.exists(successful_fold_results_file):
        print(f"错误: 成功的 Fold 结果文件 {successful_fold_results_file} 未找到。无法继续。")
        return

    print(f"正在加载成功的 Fold 结果: {successful_fold_results_file}")
    try:
        successful_results_df = pd.read_csv(successful_fold_results_file)
        print(f"成功加载 successful_results_df, Shape: {successful_results_df.shape}")
        print("successful_results_df Columns:", successful_results_df.columns.tolist())
        print("successful_results_df Head:\n", successful_results_df.head().to_string())
        print("successful_results_df Dtypes:\n", successful_results_df.dtypes)

        # --- 调试点 2: 检查指标列的数据类型和NaN值 (在转换为数值型之前和之后) ---
        metrics_for_mean_calc = ["Accuracy", "F1-score (minority)", "Recall (minority)", "G-mean", "Balanced Accuracy",
                                 "AUC-PR", "Total Model Cost"]
        valid_metrics_for_calc = []
        print("\n--- 检查 successful_results_df 中指标列的原始状态 ---")
        for col in metrics_for_mean_calc:
            if col in successful_results_df.columns:
                print(
                    f"列 '{col}': dtype={successful_results_df[col].dtype}, NaNs={successful_results_df[col].isnull().sum()}")
                # 尝试转换并检查
                try:
                    converted_col = pd.to_numeric(successful_results_df[col], errors='coerce')
                    if not converted_col.isnull().all():  # 如果转换后不是全NaN
                        successful_results_df[col] = converted_col  # 更新原始DataFrame中的列
                        valid_metrics_for_calc.append(col)
                        print(
                            f"  '{col}' 成功转换为数值型, NaNs after coerce: {successful_results_df[col].isnull().sum()}")
                    else:
                        print(f"  '{col}' 转换为数值型后全为 NaN，将不用于计算均值。")
                except Exception as e_conv:
                    print(f"  '{col}' 转换为数值型失败: {e_conv}，将不用于计算均值。")
            else:
                print(f"警告: 指标列 '{col}' 在 successful_results_df 中未找到。")

        print(f"\n有效用于计算均值的指标列: {valid_metrics_for_calc}")

        if not valid_metrics_for_calc or 'Model' not in successful_results_df.columns:
            print("错误: successful_results_df 中缺少 'Model' 列或没有有效的指标列来计算均值。")
            return

        # --- 调试点 3: 检查 groupby().agg() 的结果 (temp_agg_for_plot) ---
        print("\n--- 正在计算 temp_agg_for_plot ---")
        temp_agg_for_plot = successful_results_df.groupby(['Model']).agg(
            **{f'{col}_mean': pd.NamedAgg(column=col, aggfunc='mean') for col in valid_metrics_for_calc}
        ).reset_index()

        print("temp_agg_for_plot (用于绘图) Shape:", temp_agg_for_plot.shape)
        print("temp_agg_for_plot Columns:", temp_agg_for_plot.columns.tolist())
        print("temp_agg_for_plot Head:\n", temp_agg_for_plot.head().to_string())
        print("temp_agg_for_plot Dtypes:\n", temp_agg_for_plot.dtypes)

        # 检查生成的 _mean 列的NaN情况
        generated_mean_cols = [f'{m}_mean' for m in valid_metrics_for_calc]
        print("\n--- 检查 temp_agg_for_plot 中 _mean 列的 NaN 情况 ---")
        all_cols_present_for_plot = True
        for mean_col in generated_mean_cols:
            if mean_col in temp_agg_for_plot.columns:
                print(
                    f"列 '{mean_col}': NaNs={temp_agg_for_plot[mean_col].isnull().sum()}, dtype={temp_agg_for_plot[mean_col].dtype}")
                if temp_agg_for_plot[mean_col].isnull().all():
                    print(f"  警告: '{mean_col}' 在 temp_agg_for_plot 中全为 NaN!")
            else:
                print(f"错误: 期望的均值列 '{mean_col}' 未在 temp_agg_for_plot 中生成。")
                all_cols_present_for_plot = False

        if not all_cols_present_for_plot:
            print("错误：部分期望的均值列未能正确生成，无法继续绘图。")
            return

        if temp_agg_for_plot.empty:
            print("错误: temp_agg_for_plot 为空，无法绘制模型性能比较图。")
            return

        # --- 调试点 4: 调用 plot_model_performance_comparison ---
        print("\n--- 准备调用 plot_model_performance_comparison ---")
        # metrics_cols_mean_for_plot 应该是 temp_agg_for_plot 中实际存在的 _mean 列
        metrics_cols_mean_for_plot = [col for col in temp_agg_for_plot.columns if col.endswith('_mean')]
        if not metrics_cols_mean_for_plot:
            print("错误：在 temp_agg_for_plot 中没有找到任何 _mean 后缀的列用于绘图。")
            return

        print(f"传递给 plot_model_performance_comparison 的 metrics_cols_mean: {metrics_cols_mean_for_plot}")

        plot_model_performance_comparison(
            temp_agg_for_plot,  # DataFrame 包含 Model 和 *_mean 列
            metrics_cols_mean=metrics_cols_mean_for_plot,  # 实际存在的 *_mean 列名列表
            model_col='Model'
        )

        # --- 学习曲线部分 (与之前类似，确保 successful_results_df 中的指标列已转换为数值型) ---
        print(f"\n--- 准备绘制学习曲线 (使用原始 successful_results_df) ---")
        if 'Model' in successful_results_df.columns:  # 确保 'Model' 列存在
            # 确保用于学习曲线的指标列也是数值型
            lc_metrics_to_check = ["Recall (minority)", "F1-score (minority)"]  # 示例
            for lc_metric in lc_metrics_to_check:
                if lc_metric not in valid_metrics_for_calc:  # 检查是否是之前验证过的有效数值列
                    print(f"警告: 学习曲线指标 '{lc_metric}' 不是有效的数值列，可能无法绘制。")

            # n_estimators 学习曲线
            fixed_params_n_scan = "LR0.10_FN10_D3"  # !! 根据你的扫描实验修改 !!
            model_pattern_n = r"CASB_Scan_N_ESTIMATORS_VN(\d+)_" + re.escape(fixed_params_n_scan)
            print(f"学习曲线 (n_estimators) - 模型模式: {model_pattern_n}, 固定参数: {fixed_params_n_scan}")
            plot_learning_curves_generic(
                results_df=successful_results_df,
                model_name_pattern=model_pattern_n,
                varying_param_char='N_Estimators',
                fixed_params_str=fixed_params_n_scan,
                score_to_plot="Recall (minority)",
                dataset_name=dataset_basename
            )

            # FN_cost 学习曲线
            fixed_params_fn_scan = "LR0.10_N100_D3"  # !! 根据你的扫描实验修改 !!
            model_pattern_fn = r"CASB_Scan_FN_COST_VFN(\d+)_" + re.escape(fixed_params_fn_scan)
            print(f"学习曲线 (FN_cost) - 模型模式: {model_pattern_fn}, 固定参数: {fixed_params_fn_scan}")
            plot_learning_curves_generic(
                results_df=successful_results_df,
                model_name_pattern=model_pattern_fn,
                varying_param_char='FN_Cost',
                fixed_params_str=fixed_params_fn_scan,
                score_to_plot="F1-score (minority)",
                dataset_name=dataset_basename
            )
        else:
            print("successful_results_df 中缺少 'Model' 列，无法绘制学习曲线。")


    except FileNotFoundError:
        print(f"错误: 文件 {successful_fold_results_file} 未找到。")
    except pd.errors.EmptyDataError:
        print(f"错误: 文件 {successful_fold_results_file} 为空。")
    except Exception as e:
        print(f"处理文件 {successful_fold_results_file} 或绘图时发生意外错误: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- 图表生成脚本执行完毕 ---")


if __name__ == "__main__":
    # 确保 PLOTS_DIR 存在
    # (config.py 应该已经创建了，但这里再次检查是安全的)
    # PLOTS_DIR 需要在 main_plotting 内部从 config 模块获取
    if 'PLOTS_DIR' not in globals() or not os.path.exists(PLOTS_DIR):
        # 尝试从 config 获取或创建默认
        try:
            from config import PLOTS_DIR as cfg_plots_dir

            os.makedirs(cfg_plots_dir, exist_ok=True)
            # 将 cfg_plots_dir 赋值给全局 PLOTS_DIR 以便 visualization 模块能找到它
            # (虽然 visualization 内部有自己的 fallback，但在这里设置更一致)
            globals()['PLOTS_DIR'] = cfg_plots_dir
        except ImportError:
            plots_fallback_dir = os.path.join(project_root, "results", "plots")
            os.makedirs(plots_fallback_dir, exist_ok=True)
            globals()['PLOTS_DIR'] = plots_fallback_dir
            print(f"Fallback: PLOTS_DIR set to {plots_fallback_dir} as config could not be fully loaded.")

    main_plotting()