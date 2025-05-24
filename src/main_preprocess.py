# src/main_preprocess.py
import pandas as pd
import numpy as np  # For np.number
import os
from src.config import (DATA_DIR_RAW, PROCESSED_DATA_DIR, TARGET_VARIABLE,
                        CORRELATION_THRESHOLD_TARGET, MULTICOLLINEARITY_THRESHOLD)
from src.data_loader import load_and_merge_data, map_target_variable
from src.preprocessing import handle_missing_values_simple, encode_features
from src.feature_selector import select_features_by_target_correlation, handle_multicollinearity
from src.visualization import plot_target_distribution, plot_feature_target_correlation, plot_correlation_heatmap


def main():
    print("--- 开始数据预处理和特征选择流程 ---")

    # 1. 加载和合并数据
    original_merged_df = load_and_merge_data(DATA_DIR_RAW)
    if original_merged_df.empty:
        print("由于数据加载问题退出。")
        return

    # 1.1 映射目标变量
    df_target_mapped = map_target_variable(original_merged_df.copy())
    if TARGET_VARIABLE not in df_target_mapped.columns or df_target_mapped[TARGET_VARIABLE].isnull().all():
        print(f"目标变量 '{TARGET_VARIABLE}' 无法正确映射或全为空。退出。")
        return

    # 移除目标变量为NaN的行 (在映射或原始数据中可能出现)
    if df_target_mapped[TARGET_VARIABLE].isnull().any():
        print(f"目标变量 '{TARGET_VARIABLE}' 包含NaN值，正在移除这些行。原始行数: {len(df_target_mapped)}")
        df_target_mapped.dropna(subset=[TARGET_VARIABLE], inplace=True)
        print(f"移除后行数: {len(df_target_mapped)}")
    try:
        df_target_mapped[TARGET_VARIABLE] = df_target_mapped[TARGET_VARIABLE].astype(int)
    except ValueError:
        print(f"无法将目标变量 '{TARGET_VARIABLE}' 转换为整数。请检查映射逻辑。")
        return

    # 1.2 可视化目标变量分布
    plot_target_distribution(df_target_mapped, TARGET_VARIABLE)

    # 2. 处理缺失值 (在编码之前处理，特别是对于对象类型)
    df_no_missing = handle_missing_values_simple(df_target_mapped.copy())

    # 3. 特征编码
    df_encoded, label_encoder_map = encode_features(df_no_missing.copy())
    if df_encoded.empty:
        print("由于特征编码问题退出。")
        return

    # 确保目标变量仍然存在且为数值型
    if TARGET_VARIABLE not in df_encoded.columns or not pd.api.types.is_numeric_dtype(df_encoded[TARGET_VARIABLE]):
        print(f"目标变量 '{TARGET_VARIABLE}' 在编码后丢失或非数值。退出。")
        df_encoded[TARGET_VARIABLE] = df_no_missing[TARGET_VARIABLE].astype(int)  # 尝试恢复
        if TARGET_VARIABLE not in df_encoded.columns or not pd.api.types.is_numeric_dtype(df_encoded[TARGET_VARIABLE]):
            print("恢复目标变量失败。")
            return

    # 准备用于相关性计算的数据框 (应全为数值型)
    # df_encoded 此时应该只包含数值型特征和已编码的目标变量
    # 如果还有非数值列，说明 encode_features 有问题
    numeric_df_for_correlation = df_encoded.select_dtypes(include=np.number)
    # 确保目标变量在内
    if TARGET_VARIABLE not in numeric_df_for_correlation.columns and TARGET_VARIABLE in df_encoded.columns:
        numeric_df_for_correlation[TARGET_VARIABLE] = df_encoded[TARGET_VARIABLE]

    if TARGET_VARIABLE not in numeric_df_for_correlation.columns:
        print(f"目标变量 '{TARGET_VARIABLE}' 不在数值型数据帧中，无法进行相关性分析。")
        return

    # 4. 基于与目标变量的相关性筛选特征
    selected_features_initial, all_correlations_with_target = select_features_by_target_correlation(
        numeric_df_for_correlation,  # 应该传入只包含数值列的df
        CORRELATION_THRESHOLD_TARGET
    )
    if not selected_features_initial or TARGET_VARIABLE not in selected_features_initial:
        print("基于目标相关性未选择任何特征或目标变量丢失。退出。")
        return

    # 可视化特征与目标的相关性
    # 从 all_correlations_with_target 中获取用于绘图的数据 (排除目标自身)
    features_for_plot = [f for f in selected_features_initial if f != TARGET_VARIABLE]
    if features_for_plot:  # 只有在选择了其他特征时才绘制
        plot_feature_target_correlation(
            all_correlations_with_target[features_for_plot],  # Series of correlations for selected features
            TARGET_VARIABLE,
            CORRELATION_THRESHOLD_TARGET
        )

    # 可视化初选特征间的相关性 (在多重共线性处理前)
    if len(features_for_plot) > 1:
        df_for_heatmap_initial = numeric_df_for_correlation[features_for_plot].copy()  # 使用原始数值数据
        plot_correlation_heatmap(df_for_heatmap_initial.corr(), "Initial_Selected_Features_Correlation_Matrix")

    # 5. 处理多重共线性
    # numeric_df_for_correlation 用于相关性值计算
    final_feature_list = handle_multicollinearity(
        numeric_df_for_correlation,
        selected_features_initial,  # 包含目标变量的列表
        MULTICOLLINEARITY_THRESHOLD,
        all_correlations_with_target  # Series: feature -> corr_with_target
    )

    # 6. 创建并保存最终的 DataFrame
    print("\n--- 最终选择的特征 (包括目标变量) ---")
    print(final_feature_list)

    # 从 df_encoded (包含所有编码后特征的DataFrame) 中选取最终特征列
    # 确保 final_feature_list 中的列都存在于 df_encoded 中
    missing_cols = [col for col in final_feature_list if col not in df_encoded.columns]
    if missing_cols:
        print(f"警告: 以下最终选择的特征在df_encoded中找不到: {missing_cols}")
        final_feature_list = [col for col in final_feature_list if col in df_encoded.columns]  # 只保留存在的

    df_final_model_input = df_encoded[final_feature_list].copy()

    print("\n用于模型输入的最终数据框 (df_final_model_input) 信息:")
    df_final_model_input.info(verbose=True, show_counts=True)  # More detailed info
    print(df_final_model_input.head())

    # 可视化最终特征集的相关性
    final_features_for_heatmap = [f for f in final_feature_list if
                                  f != TARGET_VARIABLE and f in numeric_df_for_correlation.columns]
    if len(final_features_for_heatmap) > 1:
        df_for_heatmap_final = numeric_df_for_correlation[final_features_for_heatmap].copy()
        plot_correlation_heatmap(df_for_heatmap_final.corr(), "Final_Selected_Features_Correlation_Matrix")

    # 保存处理后的数据
    output_file_name = "ALL_preprocessed.csv"  # Give a more descriptive name
    output_file_path = os.path.join(PROCESSED_DATA_DIR, output_file_name)
    try:
        df_final_model_input.to_csv(output_file_path, index=False)
        print(f"\n最终处理后的数据已保存到: {output_file_path}")
    except Exception as e:
        print(f"保存最终数据到 {output_file_path} 时出错: {e}")

    print("\n--- 数据预处理和特征选择流程完成 ---")


if __name__ == "__main__":
    # 为了测试，你可以临时设置一个环境变量来触发虚拟数据的创建
    # os.environ["test_mode_no_real_data"] = "1"
    main()
    # del os.environ["test_mode_no_real_data"] # 清理