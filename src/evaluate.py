# src/evaluate.py
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             precision_recall_curve, auc, balanced_accuracy_score,
                             confusion_matrix, recall_score)
from imblearn.metrics import geometric_mean_score  # Ensure imblearn is installed


def calculate_model_cost(y_true, y_pred, cost_matrix):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = 0, 0, 0, 0  # Initialize

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    elif cm.shape == (1, 1):  # Only one class predicted AND present in y_true
        # This happens if y_true and y_pred are all 0s or all 1s.
        unique_true = np.unique(y_true)
        unique_pred = np.unique(y_pred)
        if len(unique_true) == 1 and len(unique_pred) == 1 and unique_true[0] == unique_pred[0]:
            if unique_true[0] == 0:
                tn = cm[0, 0]
            else:
                tp = cm[0, 0]
        # Else, it's more complex, e.g. y_true has two classes, y_pred only one.
        # This case needs careful thought or rely on sklearn's metrics to handle implicit counts.
        # For explicit cost, we need all four cm values.
    # A more robust way to get tn, fp, fn, tp if y_pred is constant:
    if len(np.unique(y_pred)) == 1:
        pred_class = np.unique(y_pred)[0]
        if pred_class == 0:  # All predicted negative
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            fp = 0
            tp = 0
        else:  # All predicted positive
            fp = np.sum((y_true == 0) & (y_pred == 1))
            tp = np.sum((y_true == 1) & (y_pred == 1))
            tn = 0
            fn = 0
    elif cm.shape == (2, 2):  # Fallback to ravel if not constant and 2x2
        tn, fp, fn, tp = cm.ravel()
    # else: # Other shapes, means y_true might be single class - metrics below handle it.
    # For cost, this simplified version might be inexact if y_true is also single class
    # but different from y_pred single class.

    total_cost = tn * cost_matrix.get((0, 0), 0) + \
                 fp * cost_matrix.get((0, 1), 1) + \
                 fn * cost_matrix.get((1, 0), 1) + \
                 tp * cost_matrix.get((1, 1), 0)
    return total_cost


def evaluate_classification_model(y_true, y_pred, y_proba, cost_matrix_eval, minority_label=1, model_classes_ref=None):
    # Ensure y_true and y_pred are numpy arrays for consistent behavior
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Handle cases where y_pred might be constant (predicts only one class)
    # or y_true has only one class (can happen in small test folds or due to issues)
    unique_true_labels = np.unique(y_true)

    if len(unique_true_labels) < 2:
        # print(f"Warning: y_true has only {len(unique_true_labels)} unique label(s): {unique_true_labels}. Metrics may be ill-defined.")
        # For metrics like F1, recall, gmean, AUC-PR, if only one class in true, they are often 0 or undefined
        # if that class is not the positive_label.
        f1 = 0.0
        recall_min = 0.0
        gmean = 0.0  # Or 1.0 if all are correctly classified as the single true class
        auc_pr = 0.0  # Undefined if only one class
        # Accuracy and balanced accuracy might still be valid
    else:
        f1 = f1_score(y_true, y_pred, labels=model_classes_ref if model_classes_ref is not None else [0, 1],
                      pos_label=minority_label, zero_division=0)
        recall_min = recall_score(y_true, y_pred, labels=model_classes_ref if model_classes_ref is not None else [0, 1],
                                  pos_label=minority_label, zero_division=0)
        try:
            gmean = geometric_mean_score(y_true, y_pred,
                                         labels=model_classes_ref if model_classes_ref is not None else [0, 1],
                                         pos_label=minority_label, average='binary', correction=1e-6)
        except ValueError:  # Can happen if a class specified by pos_label is not present
            gmean = 0.0

    bal_acc = balanced_accuracy_score(y_true, y_pred)  # Generally robust
    acc = accuracy_score(y_true, y_pred)

    proba_minority = np.zeros(len(y_true))  # Default to zeros
    if y_proba is not None and y_proba.ndim == 2:
        if y_proba.shape[1] == 2:  # Standard binary output
            proba_minority = y_proba[:, minority_label]
        elif y_proba.shape[1] == 1 and model_classes_ref is not None and len(model_classes_ref) == 1:
            # Model predicts proba for its single class
            predicted_cls_by_proba = model_classes_ref[0]
            if predicted_cls_by_proba == minority_label:
                proba_minority = y_proba[:, 0]
            else:  # Proba is for the majority, minority proba is 1 - that
                proba_minority = 1.0 - y_proba[:, 0]
        # else: print(f"Warning: y_proba shape {y_proba.shape} or model_classes_ref {model_classes_ref} not standard for proba_minority extraction.")
    # else: print(f"Warning: y_proba is None or not a 2D array. AUC-PR might be affected.")

    # AUC-PR calculation
    if len(unique_true_labels) < 2:  # AUC-PR undefined if only one true class
        auc_pr = 0.0
    else:
        try:
            precision_pts, recall_pts, _ = precision_recall_curve(y_true, proba_minority, pos_label=minority_label)
            if len(recall_pts) > 1 and len(precision_pts) > 1:  # Need at least 2 points
                auc_pr = auc(recall_pts, precision_pts)
            else:
                auc_pr = 0.0  # Not enough points to calculate area
        except ValueError as e:  # Can happen if y_true does not contain pos_label
            # print(f"Warning: Could not calculate AUC-PR: {e}")
            auc_pr = 0.0

    model_c = calculate_model_cost(y_true, y_pred, cost_matrix_eval)

    return {
        "Accuracy": acc,
        "F1-score (minority)": f1,
        "Recall (minority)": recall_min,
        "G-mean": gmean,
        "Balanced Accuracy": bal_acc,
        "AUC-PR": auc_pr,
        "Total Model Cost": model_c
    }