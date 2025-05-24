# src/experiment_runner.py
import pandas as pd
import numpy as np
import os
import traceback
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold  # Ensure StratifiedKFold is used
from sklearn.preprocessing import MinMaxScaler  # For CV-safe scaling
from sklearn.impute import SimpleImputer  # For CV-safe imputation
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from joblib import Parallel, delayed
# Samplers
from imblearn.over_sampling import SMOTE, ADASYN
# Metrics and models will be imported as needed or passed
from src.evaluate import evaluate_classification_model
from src.config import (RANDOM_SEED, EVAL_COST_MATRIX, N_JOBS_PARALLEL,
                        USE_CROSS_VALIDATION, N_SPLITS_CV, N_REPEATS_CV,
                        POTENTIALLY_LEAKY_FEATURES, TARGET_VARIABLE,
                        MODELS_FOR_STAT_TEST,
                        STAT_TEST_LIBS_AVAILABLE as CONFIG_STAT_TEST_FLAG)  # Import STAT_TEST_LIBS_AVAILABLE from config

# Attempt to import statistical test libraries and update global flag
STAT_TEST_LIBS_AVAILABLE_LOCAL = False  # Local flag for this module
try:
    from scipy.stats import friedmanchisquare
    import scikit_posthocs as sp

    STAT_TEST_LIBS_AVAILABLE_LOCAL = True
    if not CONFIG_STAT_TEST_FLAG:  # If config was False, update it based on successful import
        # This is a bit hacky; ideally config shouldn't be modified at runtime this way
        # For now, we'll use the local flag primarily.
        print("Note: Statistical libraries found, overriding config STAT_TEST_LIBS_AVAILABLE to True for this session.")
        # from src import config # Re-import to modify? Or pass this status back.
        # config.STAT_TEST_LIBS_AVAILABLE = True # This won't persist if config is imported elsewhere again
except ImportError:
    STAT_TEST_LIBS_AVAILABLE_LOCAL = False
    print("警告: scipy 或 scikit-posthocs 未安装。统计检验将被跳过。")


# Worker function for parallel execution
def train_evaluate_fold_task(params):
    model_name, model_blueprint, X_train, y_train, X_test, y_test, \
        split_name, dataset_basename, eval_cost_matrix, spw_val_for_xgb = params

    try:
        current_model = clone(model_blueprint)
        if model_name == "XGBoost" and spw_val_for_xgb is not None:
            current_model.set_params(scale_pos_weight=spw_val_for_xgb)

        current_model.fit(X_train, y_train)

        y_pred = current_model.predict(X_test)
        y_proba = np.array([[0.5, 0.5]] * len(y_test))  # Default
        if hasattr(current_model, "predict_proba"):
            y_proba = current_model.predict_proba(X_test)
        else:
            print(f"警告: 模型 {model_name} 没有 predict_proba 方法。AUC-PR 将受影响。")

        # Ensure y_proba has 2 columns for binary classification, even if model is sure
        model_classes_ref = getattr(current_model, 'classes_', None)
        if y_proba.ndim == 1:  # If predict_proba returns 1D array (e.g. for one class)
            # This case is common if the model only learned/predicted one class.
            # We need to reshape it to (n_samples, 1) to then convert to (n_samples, 2)
            y_proba = y_proba.reshape(-1, 1)

        if y_proba.shape[1] == 1:
            temp_probas = np.zeros((X_test.shape[0], 2))
            if model_classes_ref is not None and len(model_classes_ref) == 1:
                pred_cls = model_classes_ref[0]  # The single class the model knows
                if pred_cls == 0:  # Proba is for class 0
                    temp_probas[:, 0] = y_proba[:, 0]
                    temp_probas[:, 1] = 1.0 - y_proba[:, 0]
                else:  # Proba is for class 1
                    temp_probas[:, 1] = y_proba[:, 0]
                    temp_probas[:, 0] = 1.0 - y_proba[:, 0]
            else:  # Fallback: if classes_ is ambiguous or model predicts only one class but trained on two
                # This often means the model predicted a constant value.
                # Let's assume the proba is for the predicted class if y_pred is constant.
                unique_preds_task = np.unique(y_pred)
                if len(unique_preds_task) == 1:
                    # print(f"Model {model_name} predicted a single class: {unique_preds_task[0]}. Adjusting y_proba.")
                    if unique_preds_task[0] == 0:  # Predicted all 0
                        temp_probas[:, 0] = 1.0  # Certainty for class 0
                        temp_probas[:, 1] = 0.0
                    else:  # Predicted all 1
                        temp_probas[:, 1] = 1.0  # Certainty for class 1
                        temp_probas[:, 0] = 0.0
                else:  # Should not be reached if y_proba.shape[1] == 1 and y_pred not constant
                    # print(f"Warning: y_proba shape is (X,1) but y_pred is not constant for {model_name}. Using 0.5/0.5.")
                    temp_probas[:, 0] = 0.5;
                    temp_probas[:, 1] = 0.5
            y_proba = temp_probas

        metrics = evaluate_classification_model(y_test, y_pred, y_proba, eval_cost_matrix, minority_label=1,
                                                model_classes_ref=model_classes_ref)
        metrics['Dataset'] = dataset_basename
        metrics['Model'] = model_name
        metrics['Split'] = split_name
        return metrics
    except Exception as e:
        # print(f"!!! 错误发生在任务 {model_name} ({split_name}): {e} !!!")
        # print(traceback.format_exc()) # For debugging
        return {'Dataset': dataset_basename, 'Model': model_name, 'Split': split_name, 'Error': str(e),
                'Traceback': traceback.format_exc()}


def run_experiment(data_df, dataset_name_str, model_blueprints_dict, sampler_configs_dict):
    results_list = []

    # Prepare X and y from the input dataframe
    # Remove leaky features BEFORE splitting or any CV
    cols_to_drop_globally = [col for col in POTENTIALLY_LEAKY_FEATURES if col in data_df.columns]
    if TARGET_VARIABLE in cols_to_drop_globally:  # Should not happen
        cols_to_drop_globally.remove(TARGET_VARIABLE)

    X_full = data_df.drop(columns=[TARGET_VARIABLE] + cols_to_drop_globally, axis=1, errors='ignore')
    y_full = data_df[TARGET_VARIABLE].astype(int)  # Ensure target is int

    # Identify numeric and categorical features for ColumnTransformer
    numeric_features = X_full.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_full.select_dtypes(exclude=np.number).columns.tolist()

    print(
        f"数值型特征 ({len(numeric_features)}): {numeric_features if len(numeric_features) < 10 else str(numeric_features[:10]) + '...'}")
    print(
        f"类别型特征 ({len(categorical_features)}): {categorical_features if len(categorical_features) < 10 else str(categorical_features[:10]) + '...'}")

    # Create preprocessor for CV (impute then scale for numeric, impute for categorical)
    # Note: Categorical features here are assumed to be label-encoded *before* this stage
    # If they are raw strings, OneHotEncoder would be needed for many models (but not DTs)
    # The `main_preprocess.py` already does LabelEncoding.

    # For this setup, we assume main_preprocess.py already converted all to numeric.
    # If not, ColumnTransformer is more robust. Here, a simpler path assuming numeric input to CV.
    # Let's verify all X_full columns are numeric after main_preprocess.py
    if not all(pd.api.types.is_numeric_dtype(X_full[col]) for col in X_full.columns):
        print(
            "警告:并非所有 X_full 的列都是数值型。CV内部的预处理可能不完整。请确保 main_preprocess.py 正确处理了所有特征。")
        # Fallback: only use numeric features identified
        # X_full = X_full[numeric_features]
        # print(f"仅使用数值型特征进行CV: {X_full.columns.tolist()}")

    current_dataset_tasks = []

    if USE_CROSS_VALIDATION:
        rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS_CV, n_repeats=N_REPEATS_CV, random_state=RANDOM_SEED)
        fold_num = 0
        for train_index, test_index in rskf.split(X_full, y_full):
            fold_num += 1
            split_name_cv = f"Fold_{fold_num}"

            X_train_fold_orig, X_test_fold_orig = X_full.iloc[train_index], X_full.iloc[test_index]
            y_train_fold, y_test_fold = y_full.iloc[train_index], y_full.iloc[test_index]

            # CV-safe preprocessing: Impute and Scale
            # Create separate pipelines for numeric and categorical if they still exist
            # For this example, assuming X_full is already mostly numeric from main_preprocess.py

            # Simpler pipeline: Impute (median for all, as DTs can handle it) then Scale
            # This assumes all features in X_train_fold_orig are numeric or can be treated as such by SimpleImputer
            try:
                # More robust: handle numeric and categorical separately IF they exist with different types
                # For now, assuming `main_preprocess` made everything numeric that should be.
                # If X_full still has mixed types, a ColumnTransformer is better.

                # Let's be explicit for numeric features only for scaling
                # And assume imputation was handled before or here for numeric.

                num_pipeline = Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', MinMaxScaler())
                ])

                # Apply pipeline only to numeric features if categorical ones are present and distinct
                # If all are numeric, can apply to whole X_train_fold_orig

                X_train_processed = X_train_fold_orig.copy()
                X_test_processed = X_test_fold_orig.copy()

                # If your main_preprocess.py guarantees all features passed to experiment are numeric:
                if numeric_features:  # Check if list is not empty
                    X_train_processed[numeric_features] = num_pipeline.fit_transform(
                        X_train_fold_orig[numeric_features])
                    X_test_processed[numeric_features] = num_pipeline.transform(X_test_fold_orig[numeric_features])
                # If there are categorical features that were label encoded and are now numeric, they will be scaled.
                # This is usually fine for tree-based models.

            except Exception as e_prep:
                print(f"CV内部预处理错误 (Fold {fold_num}): {e_prep}. 跳过此Fold的任务。")
                results_list.append(
                    {'Dataset': dataset_name_str, 'Model': 'Preprocessing_Error', 'Split': split_name_cv,
                     'Error': str(e_prep), 'Traceback': traceback.format_exc()})
                continue

            # Calculate scale_pos_weight for XGBoost for this fold
            count_neg_tr = np.sum(y_train_fold == 0)
            count_pos_tr = np.sum(y_train_fold == 1)
            spw_xgb_fold = 1.0 if count_pos_tr == 0 else count_neg_tr / count_pos_tr

            # --- Tasks for models with samplers (SMOTE, ADASYN) ---
            samplers_map = {"SMOTE": SMOTE(random_state=RANDOM_SEED), "ADASYN": ADASYN(random_state=RANDOM_SEED)}
            base_clf_for_sampling = sampler_configs_dict.get("DT_Shallow_After_Sampling")  # Get the DT blueprint

            if base_clf_for_sampling:
                for sampler_name, sampler_instance in samplers_map.items():
                    model_id_sampler = f"{sampler_name} + DT_Shallow"  # Name from 06.ipynb
                    try:
                        # Ensure X_train_processed has feature names if sampler needs them (some do for ex_strategy)
                        # SMOTE/ADASYN work with numpy arrays.
                        X_resampled_fold, y_resampled_fold = sampler_instance.fit_resample(X_train_processed,
                                                                                           y_train_fold)

                        task_params_sampler = (
                            model_id_sampler, clone(base_clf_for_sampling),
                            X_resampled_fold, y_resampled_fold, X_test_processed, y_test_fold,
                            split_name_cv, dataset_name_str, EVAL_COST_MATRIX, None  # No SPW for this DT
                        )
                        current_dataset_tasks.append(delayed(train_evaluate_fold_task)(task_params_sampler))
                    except Exception as e_samp:
                        print(f"错误发生在 {model_id_sampler} 的重采样/任务准备阶段 (Fold {fold_num}): {e_samp}")
                        results_list.append(
                            {'Dataset': dataset_name_str, 'Model': model_id_sampler, 'Split': split_name_cv,
                             'Error': f"Resampling Error: {e_samp}", 'Traceback': traceback.format_exc()})

            # --- Tasks for other models (XGBoost, CASB variants, RUSBoost) ---
            for model_name, model_bp in model_blueprints_dict.items():
                spw_to_use = spw_xgb_fold if model_name == "XGBoost" else None
                task_params_model = (
                    model_name, model_bp, X_train_processed, y_train_fold, X_test_processed, y_test_fold,
                    split_name_cv, dataset_name_str, EVAL_COST_MATRIX, spw_to_use
                )
                current_dataset_tasks.append(delayed(train_evaluate_fold_task)(task_params_model))

        # End of CV loop
    else:  # Single split (not recommended for robust evaluation)
        print("警告: 正在使用单个训练/测试分割。交叉验证被禁用。")
        # Simplified split, no robust preprocessing here for single split example
        # This part would need its own preproc if used.
        # For this refactor, focus on CV path from 06.ipynb
        # X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_full, y_full, test_size=0.3, random_state=RANDOM_SEED, stratify=y_full)
        # ... then prepare tasks similar to above ...
        print("单分割的逻辑未在此重构中完全实现，请使用交叉验证。")

    # Execute all tasks for the current dataset
    if current_dataset_tasks:
        print(
            f"\n开始为数据集 {dataset_name_str} 并行执行 {len(current_dataset_tasks)} 个任务 ({N_JOBS_PARALLEL} jobs)...")
        # Note on verbose: 10 is quite high, 5 or 2 might be better for less output.
        dataset_fold_results = Parallel(n_jobs=N_JOBS_PARALLEL, verbose=5, backend="loky")(current_dataset_tasks)
        results_list.extend(dataset_fold_results)

    return pd.DataFrame(results_list)


def aggregate_cv_results(successful_results_df):
    if successful_results_df.empty or 'Split' not in successful_results_df.columns or successful_results_df[
        'Split'].nunique() < 2:
        print("没有足够的CV结果进行聚合。")
        return pd.DataFrame()

    print("\n\n--- 聚合交叉验证结果 (Mean ± Std) ---")
    metrics_for_agg = ["Accuracy", "F1-score (minority)", "Recall (minority)", "G-mean", "Balanced Accuracy", "AUC-PR",
                       "Total Model Cost"]

    # Ensure metrics columns are numeric
    for col in metrics_for_agg:
        if col in successful_results_df.columns:
            successful_results_df[col] = pd.to_numeric(successful_results_df[col], errors='coerce')

    results_df_for_agg = successful_results_df.dropna(subset=metrics_for_agg)  # Drop rows where any key metric is NaN
    if results_df_for_agg.empty:
        print("在dropna之后没有有效结果可用于CV聚合。")
        return pd.DataFrame()

    agg_functions = {metric: ['mean', 'std'] for metric in metrics_for_agg}
    agg_results = results_df_for_agg.groupby(['Dataset', 'Model']).agg(agg_functions).reset_index()

    # Flatten MultiIndex columns (e.g., ('Accuracy', 'mean') -> 'Accuracy_mean')
    agg_results.columns = ['_'.join(col).strip('_') for col in agg_results.columns.values]
    # Rename Dataset_ and Model_ if they got underscores
    agg_results.rename(columns={'Dataset_': 'Dataset', 'Model_': 'Model'}, inplace=True, errors='ignore')

    # Create 'Metric (Mean ± Std)' columns for display
    for metric_col_base in metrics_for_agg:
        mean_col_name = f'{metric_col_base}_mean'
        std_col_name = f'{metric_col_base}_std'
        display_col_name = f'{metric_col_base} (Mean ± Std)'

        if mean_col_name in agg_results.columns and std_col_name in agg_results.columns:
            # Format to 4 decimal places, handle NaNs from std (if only 1 fold for a group somehow)
            agg_results[display_col_name] = agg_results[mean_col_name].map('{:.4f}'.format) + \
                                            " ± " + \
                                            agg_results[std_col_name].apply(
                                                lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
            agg_results.drop(columns=[mean_col_name, std_col_name], inplace=True, errors='ignore')

    return agg_results


def perform_statistical_tests(successful_results_df, models_to_compare, metric_to_test="F1-score (minority)"):
    if not STAT_TEST_LIBS_AVAILABLE_LOCAL:
        print("统计检验库不可用，跳过检验。")
        return

    if successful_results_df.empty or 'Split' not in successful_results_df.columns or successful_results_df[
        'Split'].nunique() < 2:
        print("没有足够的CV结果进行统计检验。")
        return

    if metric_to_test not in successful_results_df.columns:
        print(f"指标 '{metric_to_test}' 未在结果中找到，无法进行统计检验。")
        return

    print(f"\n\n--- 统计显著性检验 (针对 {metric_to_test}) ---")

    # Ensure metric is numeric
    try:
        successful_results_df[metric_to_test] = pd.to_numeric(successful_results_df[metric_to_test])
    except ValueError:
        print(f"无法将指标 '{metric_to_test}' 转换为数值型，跳过统计检验。")
        return

    # Pivot table: Folds as index, Models as columns, Metric as values
    metric_pivot_df = successful_results_df.pivot_table(index='Split', columns='Model', values=metric_to_test)

    # Filter for models present in results and specified for comparison
    existing_models_for_stat = [m for m in models_to_compare if m in metric_pivot_df.columns]

    if len(existing_models_for_stat) < 2:
        print(
            f"结果中找到的待比较模型少于2个 ({len(existing_models_for_stat)} found: {existing_models_for_stat})。跳过统计检验。")
        return

    metric_to_compare_df = metric_pivot_df[
        existing_models_for_stat].dropna()  # Drop rows (folds) with NaNs for any model

    # Check if enough data points (folds) remain after dropna
    # N_SPLITS_CV * N_REPEATS_CV is the total number of folds expected
    # Friedman test typically needs a reasonable number of samples (folds) and groups (models)
    min_folds_for_test = max(N_SPLITS_CV * N_REPEATS_CV / 2, 3)  # Heuristic
    if metric_to_compare_df.shape[0] < min_folds_for_test:
        print(f"统计检验的数据点不足 (剩余folds: {metric_to_compare_df.shape[0]}，需要至少约 {min_folds_for_test:.0f})。")
        return

    try:
        # Friedman test requires list of arrays, one for each model's scores across folds
        friedman_data = [metric_to_compare_df[col].values for col in metric_to_compare_df.columns]
        statistic, p_value = friedmanchisquare(*friedman_data)
        print(
            f"Friedman 检验 ({metric_to_test}, 比较 {existing_models_for_stat}): 统计量={statistic:.4f}, p值={p_value:.4f}")

        if p_value < 0.05:  # Alpha level
            print("  Friedman 检验显著。正在执行 Nemenyi 事后检验:")
            # Nemenyi test expects a melted DataFrame or specific array format depending on library version
            # For scikit-posthocs, it can take the wide DataFrame directly
            nemenyi_matrix = sp.posthoc_nemenyi_friedman(metric_to_compare_df)
            print(nemenyi_matrix)
        else:
            print("  Friedman 检验不显著。模型间的差异不具有统计学意义。")

    except Exception as e:
        print(f"  执行统计检验时出错: {e}")
        print(traceback.format_exc())