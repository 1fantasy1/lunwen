# src/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.config import TARGET_VARIABLE, FILE_COLUMN_TO_DROP

def handle_missing_values_simple(df):
    print("\n--- 正在处理缺失值 (简单策略) ---")
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                # print(f"列 '{col}' (数值型) 的缺失值已用中位数 {median_val:.2f} 填充。")
            else:
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col].fillna(mode_val[0], inplace=True)
                    # print(f"列 '{col}' (对象型) 的缺失值已用众数 '{mode_val[0]}' 填充。")
                else:
                    df[col].fillna("Missing_Value", inplace=True)
                    # print(f"列 '{col}' (对象型) 的缺失值已用 'Missing_Value' 填充 (无众数)。")
    print("缺失值处理完成。")
    return df

def encode_features(df):
    if df.empty:
        return df, {}

    df_processed = df.copy()
    le_map = {}
    print("\n--- 正在编码特征 ---")

    if FILE_COLUMN_TO_DROP and FILE_COLUMN_TO_DROP in df_processed.columns:
        df_processed.drop(FILE_COLUMN_TO_DROP, axis=1, inplace=True)
        print(f"已删除列 '{FILE_COLUMN_TO_DROP}'。")

    for col in df_processed.columns:
        if col == TARGET_VARIABLE:
            continue

        if df_processed[col].dtype == 'object' or not pd.api.types.is_numeric_dtype(df_processed[col]): #更广泛地捕获非数值列
            # Version 列通常基数较高，但在此处我们假设它被保留并编码
            le = LabelEncoder()
            try:
                # 先填充对象列的NaN，防止LabelEncoder出错
                if df_processed[col].isnull().any():
                    mode_val = df_processed[col].mode()
                    fill_val = mode_val[0] if not mode_val.empty else "Missing_Encode"
                    df_processed[col].fillna(fill_val, inplace=True)

                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                le_map[col] = le
                # print(f"已将标签编码应用于对象列 '{col}'。")
            except Exception as e:
                print(f"标签编码列 '{col}' 时出错: {e}")
        elif df_processed[col].dtype == 'bool': #布尔列在 data_loader 中目标变量已处理，这里处理其他布尔特征
            df_processed[col] = df_processed[col].astype(int)
            # print(f"已将布尔列 '{col}' 转换为整数。")
    print("特征编码完成。")
    return df_processed, le_map