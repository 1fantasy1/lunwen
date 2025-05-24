# src/data_loader.py
import pandas as pd
import glob
import os
from src.config import TARGET_VARIABLE, POSITIVE_LABEL_IN_RAW_DATA, NEGATIVE_LABEL_IN_RAW_DATA

def load_and_merge_data(data_dir_path):
    print(f"从以下目录加载数据: {data_dir_path}")
    files = glob.glob(os.path.join(data_dir_path, "*.csv"))
    if not files:
        print(f"警告: 在目录 {data_dir_path} 中未找到CSV文件。")
        # 创建一个示例的虚拟数据，如果目录为空且用于测试
        if "test_mode_no_real_data" in os.environ: # 仅用于测试
            print("测试模式: 创建虚拟数据...")
            dummy_df = pd.DataFrame({
                'Feature1': range(100),
                'Feature2': [x * 0.5 for x in range(100)],
                'File': [f'file_{i}.java' for i in range(100)],
                TARGET_VARIABLE: [0]*80 + [1]*20, # 不平衡数据
                'Version': ['dummy_v1'] * 50 + ['dummy_v2'] * 50
            })
            dummy_df.to_csv(os.path.join(data_dir_path, "dummy_data_project1.csv"), index=False)
            files = glob.glob(os.path.join(data_dir_path, "*.csv"))
            if not files: return pd.DataFrame() # 还是没找到就返回空
        else:
            return pd.DataFrame()


    dfs = []
    for file_path in files:
        try:
            df = pd.read_csv(file_path)
            base_name = os.path.basename(file_path)
            file_name_without_ext = os.path.splitext(base_name)[0]
            df["Version"] = file_name_without_ext
            dfs.append(df)
            print(f"成功加载: {base_name}, 形状: {df.shape}")
        except Exception as e:
            print(f"读取 {file_path} 时出错: {e}")

    if not dfs:
        print("没有数据框被加载。")
        return pd.DataFrame()

    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"数据已合并。最终形状: {merged_df.shape}")
    return merged_df

def map_target_variable(df):
    if TARGET_VARIABLE not in df.columns:
        print(f"错误: 目标变量 '{TARGET_VARIABLE}' 未在数据框中找到。")
        return df

    if df[TARGET_VARIABLE].dtype == 'bool':
        df[TARGET_VARIABLE] = df[TARGET_VARIABLE].astype(int)
    elif not pd.api.types.is_numeric_dtype(df[TARGET_VARIABLE]) or \
         not all(val in [0, 1] for val in df[TARGET_VARIABLE].dropna().unique()):
        try:
            mapping = {POSITIVE_LABEL_IN_RAW_DATA: 1, NEGATIVE_LABEL_IN_RAW_DATA: 0}
            original_unique_values = df[TARGET_VARIABLE].unique()
            df[TARGET_VARIABLE] = df[TARGET_VARIABLE].map(mapping)
            if df[TARGET_VARIABLE].isnull().any():
                print(f"警告: 目标变量 '{TARGET_VARIABLE}' 在映射后包含NaN值。原始唯一值: {original_unique_values}")
            print(f"已将目标变量 '{TARGET_VARIABLE}' 映射到 0/1。")
        except Exception as e:
            print(f"错误: 无法将 '{TARGET_VARIABLE}' 映射到 0/1 整数: {e}。")
            # exit()
    return df