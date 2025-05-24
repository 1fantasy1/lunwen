# src/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import re  # For learning curve regex
from sklearn.metrics import confusion_matrix

# 导入配置 (确保这个导入路径相对于你的项目结构是正确的)
# 假设 config.py 在 src 目录下，并且 visualization.py 也在 src 目录下或其子目录
try:
    from .config import PLOTS_DIR, DEFAULT_FIG_SIZE, DEFAULT_PLOT_PALETTE, TARGET_VARIABLE
except ImportError:
    # Fallback for direct execution or if config is in a different relative path
    # This might happen if you run visualization.py directly for testing a function
    # For a structured project, the relative import above should work.
    print(
        "Warning: Could not perform relative import of config. Assuming PLOTS_DIR='results/plots' etc. for direct script run.")
    PLOTS_DIR = os.path.join(os.getcwd(), "results", "plots")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    DEFAULT_FIG_SIZE = (10, 6)
    DEFAULT_PLOT_PALETTE = "viridis"
    TARGET_VARIABLE = "RealBug"  # Placeholder, should come from config


# --- EDA Visualizations ---
def plot_target_distribution(df, target_col_name=TARGET_VARIABLE):  # Use TARGET_VARIABLE from config as default
    if target_col_name not in df.columns:
        print(f"Error: Target column '{target_col_name}' not found in DataFrame for plotting distribution.")
        return
    plt.figure(figsize=DEFAULT_FIG_SIZE)
    sns.countplot(x=target_col_name, data=df, palette=DEFAULT_PLOT_PALETTE)
    plt.title(f'Distribution of Target Variable ({target_col_name})')
    plt.xlabel('Class')
    plt.ylabel('Count')
    # Assuming target is 0 and 1 after mapping for xticks
    unique_target_values = sorted(df[target_col_name].dropna().unique())
    if all(isinstance(x, (int, float)) for x in unique_target_values) and len(unique_target_values) <= 5:
        plt.xticks(ticks=unique_target_values,
                   labels=[f'Class {int(i) if isinstance(i, float) and i.is_integer() else i}' for i in
                           unique_target_values])
    output_path = os.path.join(PLOTS_DIR, "target_distribution.png")
    try:
        plt.savefig(output_path)
        print(f"目标变量分布图已保存到: {output_path}")
    except Exception as e:
        print(f"保存目标变量分布图到 {output_path} 时出错: {e}")
    plt.close()  # Close plot to free memory


def plot_feature_target_correlation(correlation_series, target_var_name=TARGET_VARIABLE, threshold=0.05):
    if not isinstance(correlation_series, pd.Series) or correlation_series.empty:
        print("Error: 没有有效的相关性数据 (pd.Series) 可以绘制。")
        return
    plt.figure(figsize=(DEFAULT_FIG_SIZE[0], max(6, len(correlation_series) * 0.35)))  # Dynamic height
    # Ensure colors are correctly accessed if palette is a list
    bar_color = sns.color_palette(DEFAULT_PLOT_PALETTE)[0] if isinstance(sns.color_palette(DEFAULT_PLOT_PALETTE),
                                                                         list) else DEFAULT_PLOT_PALETTE
    correlation_series.sort_values().plot(kind='barh', color=bar_color)
    plt.title(f'Features Correlated with {target_var_name} (Abs. Threshold: {threshold:.2f})')
    plt.xlabel(f'Correlation with {target_var_name}')
    plt.ylabel('Feature')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    output_path = os.path.join(PLOTS_DIR, "feature_target_correlation.png")
    try:
        plt.savefig(output_path)
        print(f"特征-目标相关性图已保存到: {output_path}")
    except Exception as e:
        print(f"保存特征-目标相关性图到 {output_path} 时出错: {e}")
    plt.close()


def plot_correlation_heatmap(corr_matrix, title="Feature_Correlation_Heatmap"):
    if not isinstance(corr_matrix, pd.DataFrame) or corr_matrix.empty or corr_matrix.shape[0] < 2:
        print(f"Error: 无法绘制热力图 '{title}'，相关性矩阵为空、非DataFrame或特征太少。")
        return

    # Adjust figure size dynamically based on matrix size
    fig_width = max(8, corr_matrix.shape[1] * 0.8)
    fig_height = max(6, corr_matrix.shape[0] * 0.8)
    # Cap annotation font size for very large matrices
    annot_font_size = 8 if corr_matrix.shape[0] < 20 and corr_matrix.shape[1] < 20 else 6
    tick_font_size = 8 if corr_matrix.shape[0] < 30 and corr_matrix.shape[1] < 30 else 6

    plt.figure(figsize=(fig_width, fig_height))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, annot_kws={"size": annot_font_size})
    plt.title(title.replace("_", " "))  # Make title more readable
    plt.xticks(rotation=45, ha='right', fontsize=tick_font_size)
    plt.yticks(rotation=0, fontsize=tick_font_size)
    plt.tight_layout()
    filename = title.lower().replace(" ", "_").replace("-", "_") + ".png"  # Sanitize filename
    output_path = os.path.join(PLOTS_DIR, filename)
    try:
        plt.savefig(output_path)
        print(f"相关性热力图 '{title.replace('_', ' ')}' 已保存到: {output_path}")
    except Exception as e:
        print(f"保存相关性热力图 '{title}' 到 {output_path} 时出错: {e}")
    plt.close()


# --- Model Performance Visualizations ---
def plot_model_performance_comparison(agg_results_df, metrics_cols_mean, model_col='Model'):
    """
    绘制模型性能比较图。
    会将 "Total Model Cost" 单独绘制，其他指标绘制在另一张图上，
    以处理Y轴数值范围差异过大的问题。
    """
    print("\n--- Running Enhanced plot_model_performance_comparison (with split plots) ---")
    print(
        f"Input agg_results_df shape: {agg_results_df.shape if isinstance(agg_results_df, pd.DataFrame) else 'Not a DataFrame'}")
    if isinstance(agg_results_df, pd.DataFrame):
        print(f"Input agg_results_df columns: {agg_results_df.columns.tolist()}")
    print(f"Input metrics_cols_mean: {metrics_cols_mean}")
    print(f"Input model_col: {model_col}")

    if not isinstance(agg_results_df, pd.DataFrame) or agg_results_df.empty or not metrics_cols_mean:
        print("Error: 没有聚合结果 (DataFrame为空) 或未指定指标用于模型性能绘图。")
        print("--- plot_model_performance_comparison End (Early Exit) ---")
        return

    if model_col not in agg_results_df.columns:
        print(f"Error: 模型列 '{model_col}' 在 agg_results_df 中找不到。")
        print("--- plot_model_performance_comparison End (Early Exit) ---")
        return

    actual_metrics_to_use = []
    missing_metrics_from_df = []
    for metric_col_name in metrics_cols_mean:
        if metric_col_name in agg_results_df.columns:
            try:
                agg_results_df[metric_col_name] = pd.to_numeric(agg_results_df[metric_col_name], errors='coerce')
                if agg_results_df[metric_col_name].isnull().all():
                    print(f"Warning: 指标列 '{metric_col_name}' 转换后全为 NaN。将跳过。")
                elif not pd.api.types.is_numeric_dtype(agg_results_df[metric_col_name]):
                    print(
                        f"Warning: 指标列 '{metric_col_name}' 转换后非数值型 ({agg_results_df[metric_col_name].dtype})。将跳过。")
                else:
                    actual_metrics_to_use.append(metric_col_name)
            except Exception as e_conv:
                print(f"Warning: 转换指标列 '{metric_col_name}' 为数值型时出错: {e_conv}。将跳过。")
        else:
            missing_metrics_from_df.append(metric_col_name)

    if missing_metrics_from_df:
        print(f"Warning: 以下指标均值列在 agg_results_df 中找不到: {missing_metrics_from_df}")

    if not actual_metrics_to_use:
        print("Error: 没有有效的、数值型的指标列可以用于绘图。")
        if isinstance(agg_results_df, pd.DataFrame) and not agg_results_df.empty:
            print("agg_results_df head:\n", agg_results_df.head().to_string())
        print("--- plot_model_performance_comparison End (Early Exit) ---")
        return

    print(f"实际用于后续处理的指标均值列: {actual_metrics_to_use}")

    # 清理列名用于图例 (在melt之前应用于原始数据选择的副本)
    plot_metric_names_for_legend = [m.replace("_mean", "").replace("_Mean", "") for m in actual_metrics_to_use]

    # 选择需要的列，并用清理后的指标名重命名这些指标列
    plot_data_renamed = agg_results_df[[model_col] + actual_metrics_to_use].copy()
    rename_map = {old_name: new_name for old_name, new_name in zip(actual_metrics_to_use, plot_metric_names_for_legend)}
    plot_data_renamed.rename(columns=rename_map, inplace=True)

    print("plot_data_renamed (列已重命名，用于melt) head:\n", plot_data_renamed.head().to_string())

    # Melt 操作
    plot_data_melted = plot_data_renamed.melt(id_vars=model_col, var_name='Metric', value_name='Score')

    # 移除 Score 为 NaN 的行
    plot_data_melted_valid = plot_data_melted.dropna(subset=['Score'])
    if plot_data_melted_valid.empty:
        print("Error: Melted 和 dropna 后没有有效数据可以绘制。")
        print("--- plot_model_performance_comparison End (Early Exit) ---")
        return

    print(f"plot_data_melted_valid (有效数据) shape: {plot_data_melted_valid.shape}")
    print(f"plot_data_melted_valid 'Metric' unique values: {plot_data_melted_valid['Metric'].unique()}")

    # --- 分离 Total Model Cost 和其他指标 ---
    # 使用清理后的指标名进行比较
    cost_metric_legend_name = "Total Model Cost".replace("_mean", "").replace("_Mean", "")

    other_metrics_data = plot_data_melted_valid[plot_data_melted_valid['Metric'] != cost_metric_legend_name]
    cost_data = plot_data_melted_valid[plot_data_melted_valid['Metric'] == cost_metric_legend_name]

    # 确保 PLOTS_DIR 和 DEFAULT_PLOT_PALETTE 是可用的
    # 这些应该从 config 模块导入，或者作为参数传递给这个函数，或者在这里有默认值
    # 为确保此函数独立可测试，我们检查它们是否存在，如果不存在则使用默认值
    current_plots_dir = globals().get('PLOTS_DIR', "results/plots")  # 从全局获取或用默认
    current_palette = globals().get('DEFAULT_PLOT_PALETTE', "viridis")
    os.makedirs(current_plots_dir, exist_ok=True)

    # --- 绘制除 Total Model Cost 之外的其他指标 ---
    if not other_metrics_data.empty:
        num_other_metrics = len(other_metrics_data['Metric'].unique())
        num_models = len(other_metrics_data[model_col].unique())

        fig_width_others = max(10, num_other_metrics * 2.0 + num_models * 0.05 * num_other_metrics)  # 调整宽度计算
        fig_width_others = min(fig_width_others, 30)  # 设置一个最大宽度上限，防止过宽

        plt.figure(figsize=(fig_width_others, 7))

        sns.barplot(x='Metric', y='Score', hue=model_col, data=other_metrics_data, palette=current_palette)

        plt.title('Model Performance Comparison (Key Metrics)')
        plt.ylabel('Mean Score (Higher is Better, for most)')
        plt.xlabel('Performance Metric')
        plt.xticks(rotation=35, ha='right', fontsize='small')
        plt.legend(title=model_col, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='x-small',
                   ncol=max(1, num_models // 12))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.ylim(0, 1.05)  # 大部分指标在0-1范围，稍微留点上面空间
        plt.tight_layout(rect=[0, 0, 0.80, 1])  # 给图例留更多空间

        output_path_others = os.path.join(current_plots_dir, "model_performance_comparison_others.png")
        try:
            plt.savefig(output_path_others, bbox_inches='tight')
            print(f"其他指标性能比较图已保存到: {output_path_others}")
        except Exception as e:
            print(f"保存其他指标图到 {output_path_others} 时出错: {e}")
        plt.close()
    else:
        print("Warning: 没有其他指标 (除Total Model Cost外) 的数据可供绘制。")

    # --- 单独绘制 Total Model Cost ---
    if not cost_data.empty:
        num_models_cost = len(cost_data[model_col].unique())
        fig_width_cost = max(8, num_models_cost * 0.5)  # 根据模型数量调整宽度
        fig_width_cost = min(fig_width_cost, 20)  # 上限

        plt.figure(figsize=(fig_width_cost, 6))
        sns.barplot(
            x=model_col,  # X轴仍然是模型
            y='Score',  # Y轴是分数
            data=cost_data,
            hue=model_col,  # 将X轴变量也赋给hue，以便palette生效
            palette=current_palette,
            legend=False  # 关闭图例，因为X轴已经标示了模型
            # dodge=False        # 如果条形重叠或不需要偏移，可以明确设置dodge=False
        )
        plt.title('Model Performance Comparison (Total Model Cost)')
        plt.ylabel('Mean Total Model Cost (Lower is Better)')
        plt.xlabel('Model')
        plt.xticks(rotation=45, ha='right', fontsize='small')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        output_path_cost = os.path.join(current_plots_dir, "model_performance_comparison_cost.png")
        try:
            plt.savefig(output_path_cost, bbox_inches='tight')
            print(f"模型成本比较图已保存到: {output_path_cost}")
        except Exception as e:
            print(f"保存模型成本图到 {output_path_cost} 时出错: {e}")
        plt.close()
    else:
        print("Warning: 没有 Total Model Cost 的数据可供绘制。")

    print("--- plot_model_performance_comparison (with split plots) End ---")


def plot_confusion_matrix_for_task(y_true, y_pred, model_name, split_name, class_labels=('No Bug', 'Bug')):
    # 确保标签是0和1，如果不是，CM可能会出错或不符合预期
    # 这个问题应该在评估函数或更早解决，这里假设y_true, y_pred是0/1编码
    labels_for_cm = [0, 1]  # 假设的二分类标签

    cm = confusion_matrix(y_true, y_pred, labels=labels_for_cm)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'Predicted {class_labels[0]}', f'Predicted {class_labels[1]}'],
                yticklabels=[f'Actual {class_labels[0]}', f'Actual {class_labels[1]}'])
    plt.title(f'Confusion Matrix: {model_name} on {split_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    # 清理模型名称以用于文件名
    clean_model_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', model_name)  # 替换特殊字符
    clean_split_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', split_name)
    filename = f"cm_{clean_model_name}_{clean_split_name}.png"
    output_path = os.path.join(PLOTS_DIR, filename)
    try:
        plt.savefig(output_path)
        # print(f"混淆矩阵图 '{filename}' 已保存到: {output_path}") # Can be too verbose
    except Exception as e:
        print(f"保存混淆矩阵图 '{filename}' 到 {output_path} 时出错: {e}")
    plt.close()


def plot_learning_curves_generic(results_df, model_name_pattern, varying_param_char, fixed_params_str,
                                 score_to_plot="F1-score (minority)", dataset_name="DefaultDataset"):
    param_values = []
    scores = []

    if not isinstance(results_df, pd.DataFrame) or results_df.empty or \
            'Model' not in results_df.columns or score_to_plot not in results_df.columns:
        print(f"Error: 结果DataFrame不适合绘制学习曲线 (空、缺列: Model or {score_to_plot})")
        return

    # 确保 score_to_plot 列是数值型
    try:
        results_df[score_to_plot] = pd.to_numeric(results_df[score_to_plot], errors='coerce')
    except Exception as e_conv:
        print(f"Error: 转换指标 '{score_to_plot}' 为数值型时出错: {e_conv}。无法绘制学习曲线。")
        return

    if results_df[score_to_plot].isnull().all():
        print(f"Warning: 指标 '{score_to_plot}' 在转换为数值型后全为 NaN。无法绘制学习曲线。")
        return

    for idx, row in results_df.iterrows():
        if not isinstance(row['Model'], str): continue  # 跳过非字符串模型名
        if pd.isna(row[score_to_plot]): continue  # 跳过指标为NaN的行

        match = re.search(model_name_pattern, row['Model'])
        if match:
            try:
                param_val_str = match.group(1)  # 假设第一个捕获组是变化的参数值
                # 尝试转换为浮点数或整数
                param_val = float(param_val_str) if '.' in param_val_str or 'e' in param_val_str.lower() else int(
                    param_val_str)
                param_values.append(param_val)
                scores.append(row[score_to_plot])
            except (ValueError, IndexError) as e:
                # print(f"Warning: 解析参数时出错 (Model: {row['Model']}, Pattern: {model_name_pattern}): {e}")
                continue  # 跳过无法解析的行

    if not param_values or not scores:
        print(f"没有找到用于学习曲线的数据，模式: '{model_name_pattern}', 指标: '{score_to_plot}'.")
        return

    # 按参数值排序以正确绘制曲线
    paired = sorted(zip(param_values, scores))
    param_values_sorted = [p[0] for p in paired]
    scores_sorted = [p[1] for p in paired]

    if not param_values_sorted:  # 应该不会到这里，因为上面检查了 param_values
        print("排序后没有数据可以绘制学习曲线。")
        return

    plt.figure(figsize=DEFAULT_FIG_SIZE)
    plt.plot(param_values_sorted, scores_sorted, marker='o', linestyle='-')
    plt.title(
        f'Learning Curve: {score_to_plot} vs {varying_param_char}\n(Fixed: {fixed_params_str}) Dataset: {dataset_name}')
    plt.xlabel(f'Parameter: {varying_param_char}')
    plt.ylabel(score_to_plot)
    plt.grid(True)

    unique_param_values_for_ticks = sorted(list(set(param_values_sorted)))
    if 0 < len(unique_param_values_for_ticks) < 15:  # 只在点数不多时设置特定刻度
        plt.xticks(unique_param_values_for_ticks)

    # 清理文件名中的特殊字符
    safe_score_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', score_to_plot)
    safe_dataset_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', dataset_name.split('.')[0])
    safe_varying_param = re.sub(r'[^a-zA-Z0-9_\-]', '_', varying_param_char)
    filename = f"lc_{safe_dataset_name}_{safe_varying_param}_{safe_score_name}.png"
    output_path = os.path.join(PLOTS_DIR, filename)

    try:
        plt.savefig(output_path)
        print(f"学习曲线已保存到: {output_path}")
    except Exception as e:
        print(f"保存学习曲线到 {output_path} 时出错: {e}")
    plt.close()
