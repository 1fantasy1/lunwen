# src/feature_selector.py
import pandas as pd
import numpy as np
from src.config import TARGET_VARIABLE


def select_features_by_target_correlation(df_numeric, threshold):
    if TARGET_VARIABLE not in df_numeric.columns:
        print(f"目标变量 '{TARGET_VARIABLE}' 不在用于相关性计算的数值型数据框中。")
        return [], pd.Series(dtype='float64')

    print(f"\n--- 基于与 '{TARGET_VARIABLE}' 的相关性筛选特征 (阈值: {threshold}) ---")
    try:
        correlations_with_target = df_numeric.corr(method='pearson')[TARGET_VARIABLE].sort_values(ascending=False)
    except Exception as e:
        print(f"计算与目标变量的相关性时出错: {e}")
        # 尝试只选择数值列
        df_numeric_cols_only = df_numeric.select_dtypes(include=np.number)
        if TARGET_VARIABLE not in df_numeric_cols_only.columns and TARGET_VARIABLE in df_numeric.columns:
            df_numeric_cols_only[TARGET_VARIABLE] = df_numeric[TARGET_VARIABLE]  # 确保目标变量在
        if TARGET_VARIABLE not in df_numeric_cols_only.columns:
            print(f"无法在数值列中找到目标变量'{TARGET_VARIABLE}'进行相关性计算。")
            return [], pd.Series(dtype='float64')
        try:
            correlations_with_target = df_numeric_cols_only.corr(method='pearson')[TARGET_VARIABLE].sort_values(
                ascending=False)
        except Exception as e2:
            print(f"再次尝试计算相关性失败: {e2}")
            return [], pd.Series(dtype='float64')

    significant_features_series = correlations_with_target[abs(correlations_with_target) > threshold]
    significant_features_series = significant_features_series.drop(TARGET_VARIABLE, errors='ignore')
    selected_feature_names = list(significant_features_series.index)

    print(f"与 '{TARGET_VARIABLE}' 的绝对相关性 > {threshold} 的特征:")
    print(significant_features_series if not significant_features_series.empty else "无")
    num_selected = len(selected_feature_names)
    print(f"基于目标相关性选择的特征数量: {num_selected}")

    final_list_for_df = selected_feature_names + [TARGET_VARIABLE]
    final_list_for_df = list(dict.fromkeys(final_list_for_df))

    return final_list_for_df, correlations_with_target


def handle_multicollinearity(df_for_corr_calc, selected_feature_names, multicollinearity_threshold,
                             correlations_with_target):
    print(f"\n--- 处理多重共线性 (阈值: {multicollinearity_threshold}) ---")
    features_for_multicol_check = [f for f in selected_feature_names if f != TARGET_VARIABLE]

    if len(features_for_multicol_check) < 2:
        print("没有足够的特征来检查多重共线性。")
        return selected_feature_names

    # 确保只对数值列进行操作
    numeric_df_for_check = df_for_corr_calc[features_for_multicol_check].select_dtypes(include=np.number)
    if numeric_df_for_check.shape[1] < 2:
        print("在所选特征中没有足够的数值型特征来检查多重共线性。")
        return selected_feature_names

    try:
        correlation_matrix_features = numeric_df_for_check.corr(method='pearson')
    except Exception as e:
        print(f"计算特征相关性矩阵时出错: {e}")
        return selected_feature_names  # 返回未处理的列表

    upper = correlation_matrix_features.where(np.triu(np.ones(correlation_matrix_features.shape), k=1).astype(bool))
    features_to_drop_multicollinearity = set()
    found_highly_correlated = False

    for column in upper.columns:
        for index in upper.index:
            correlation_value = upper.loc[index, column]
            if pd.notna(correlation_value) and abs(correlation_value) > multicollinearity_threshold:
                found_highly_correlated = True
                corr_index_target = abs(correlations_with_target.get(index, 0))
                corr_column_target = abs(correlations_with_target.get(column, 0))
                if corr_index_target >= corr_column_target:
                    if column not in features_to_drop_multicollinearity:
                        features_to_drop_multicollinearity.add(column)
                else:
                    if index not in features_to_drop_multicollinearity:
                        features_to_drop_multicollinearity.add(index)

    if not found_highly_correlated:
        print("在所选特征中未发现超过多重共线性阈值的特征对。")

    final_selected_features_after_multicol = [
        f for f in selected_feature_names if f not in features_to_drop_multicollinearity
    ]
    if TARGET_VARIABLE not in final_selected_features_after_multicol and TARGET_VARIABLE in selected_feature_names:
        final_selected_features_after_multicol.append(TARGET_VARIABLE)
    final_selected_features_after_multicol = list(dict.fromkeys(final_selected_features_after_multicol))

    print(
        f"由于多重共线性而移除的特征: {list(features_to_drop_multicollinearity) if features_to_drop_multicollinearity else '无'}")
    num_final_features = len(
        final_selected_features_after_multicol) - 1 if TARGET_VARIABLE in final_selected_features_after_multicol else len(
        final_selected_features_after_multicol)
    print(f"处理多重共线性后剩余的特征数量 (不含目标变量): {num_final_features}")
    return final_selected_features_after_multicol