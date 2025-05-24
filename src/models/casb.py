# src/models/casb.py
import numpy as np
import pandas as pd  # Only if absolutely needed, try to keep models sklearn-like
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


# from scipy.special import xlogy # Not used in the provided CASB version

class CostAdaptiveSamplerBoost(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None, n_estimators=50, learning_rate=1.0, cost_matrix=None, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.cost_matrix = cost_matrix
        self.random_state = random_state

    def _get_cost(self, y_true_val, y_pred_val):
        current_cost_matrix = self.cost_matrix
        if current_cost_matrix is None:
            current_cost_matrix = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0}  # Default SAMME-like cost
        cost = current_cost_matrix.get((y_true_val, y_pred_val))
        return cost if cost is not None else (1.0 if y_true_val != y_pred_val else 0.0)

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y, accept_sparse=False)  # accept_sparse=False by default for DT

        self._internal_estimator_blueprint = self.base_estimator if self.base_estimator is not None \
            else DecisionTreeClassifier(max_depth=3)  # Default base estimator

        self.le_ = LabelEncoder()
        y_encoded = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError(f"This classifier needs at least 2 classes; got {n_classes}")

        n_samples = X.shape[0]

        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Initialize weights
        sample_weight = np.full(n_samples, 1 / n_samples, dtype=np.float64)

        self.estimators_ = []
        self.estimator_alphas_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.n_estimators_fitted_ = 0

        for m in range(self.n_estimators):
            # Bootstrap sample
            indices = np.random.choice(np.arange(n_samples), size=n_samples, replace=True, p=sample_weight)
            X_bootstrap, y_bootstrap_encoded = X[indices], y_encoded[indices]

            estimator = clone(self._internal_estimator_blueprint)
            if hasattr(estimator, 'random_state') and self.random_state is not None:
                try:  # Some estimators might not accept random_state in this way
                    estimator.set_params(random_state=(self.random_state + m))
                except ValueError:
                    pass  # If it fails, proceed without setting it again for this clone

            estimator.fit(X_bootstrap, y_bootstrap_encoded)  # Fit on bootstrapped sample
            y_pred_train_m_encoded = estimator.predict(X)  # Predict on the *original* X for error calculation

            incorrect_mask = (y_pred_train_m_encoded != y_encoded)
            error_m = np.sum(sample_weight[incorrect_mask])

            # Alpha calculation (SAMME.R-like logic for multi-class, adapted for binary)
            if error_m <= 1e-9:  # Perfect classifier or all weights on correct samples
                alpha_m = self.learning_rate * 10  # Large positive alpha
            elif error_m >= 1.0 - (1.0 / n_classes) - 1e-9:  # Error close to random guessing or worse
                alpha_m = -self.learning_rate * 10  # Large negative alpha (or effectively zero if clipped later)
            else:
                # Original SAMME: alpha = log((1.0 - error_m) / error_m) + log(n_classes - 1)
                # Your version: alpha = learning_rate * (log((1.0 - error_m) / (error_m + 1e-9)) + log(max(n_classes - 1, 1)))
                log_term_1 = np.log((1.0 - error_m) / (error_m + 1e-9))
                log_term_2 = np.log(max(n_classes - 1,
                                        1))  # Ensure log(1) if n_classes=2, or log(0) if n_classes=1 (which is caught earlier)
                alpha_m = self.learning_rate * (log_term_1 + log_term_2)

            self.estimators_.append(estimator)
            self.estimator_alphas_[m] = alpha_m
            self.n_estimators_fitted_ = m + 1

            # Early stopping condition from your code
            if alpha_m <= 0 and m > 0 and np.all(self.estimator_alphas_[:self.n_estimators_fitted_] <= 0):
                # print(f"Stopping early at iteration {m+1} due to all non-positive alphas.")
                break
            if alpha_m <= 0 and m > 0:  # If alpha is not positive, don't update weights with it and skip to next estimator
                # print(f"Skipping weight update for non-positive alpha at iteration {m+1}.")
                continue

            # Update weights
            # Iterate through each sample to apply cost-sensitive weighting for incorrect predictions
            for i in range(n_samples):
                if incorrect_mask[i]:  # If misclassified
                    cost_val = self._get_cost(y_encoded[i], y_pred_train_m_encoded[i])
                    # Effective cost factor: use the misclassification cost, but ensure it's at least 1
                    # This means misclassifications are penalized at least as much as in standard AdaBoost,
                    # and more if the specific cost is higher.
                    effective_cost_factor = max(cost_val, 1.0) if cost_val > 0 else 1.0
                    sample_weight[i] *= np.exp(alpha_m * effective_cost_factor)
                else:  # If correctly classified
                    # For correctly classified samples, the weight update is standard
                    sample_weight[i] *= np.exp(-alpha_m)  # Standard AdaBoost weight reduction

            # Normalize weights
            sample_weight_sum = np.sum(sample_weight)
            if sample_weight_sum <= 1e-12 or np.isnan(sample_weight_sum) or np.isinf(sample_weight_sum):
                # print(f"Warning: Sample weights sum problematic ({sample_weight_sum}). Reinitializing weights.")
                sample_weight = np.full(n_samples, 1 / n_samples, dtype=np.float64)  # Reinitialize
            else:
                sample_weight /= sample_weight_sum

        # Trim estimators and alphas if early stopping occurred
        self.estimators_ = self.estimators_[:self.n_estimators_fitted_]
        self.estimator_alphas_ = self.estimator_alphas_[:self.n_estimators_fitted_]
        return self

    def predict_proba(self, X):
        check_is_fitted(self, ['estimators_', 'estimator_alphas_', 'classes_', 'le_'])
        X = check_array(X, accept_sparse=False)  # Ensure X is valid
        n_model_classes = len(self.classes_)

        # Use only estimators with positive alpha for prediction
        # (as per your original logic: "if alpha_m <= 0 and m > 0 : continue")
        # However, standard AdaBoost uses all alphas. If your alpha can be negative and meaningful, this needs care.
        # Assuming here that non-positive alphas mean the estimator is too weak or counter-productive for this stage.

        active_estimators = []
        active_alphas = []
        for i in range(self.n_estimators_fitted_):
            if self.estimator_alphas_[i] > 0:  # Consider only estimators with positive alpha for voting
                active_estimators.append(self.estimators_[i])
                active_alphas.append(self.estimator_alphas_[i])

        if not active_estimators:  # No good estimators found
            # Return uniform probabilities
            return np.full((X.shape[0], n_model_classes), 1 / n_model_classes)

        # Sum of weighted probabilities
        probas_sum = np.zeros((X.shape[0], n_model_classes))
        total_alpha = np.sum(active_alphas)

        if total_alpha == 0:  # Should not happen if active_alphas exist and are >0
            return np.full((X.shape[0], n_model_classes), 1 / n_model_classes)

        for i, estimator in enumerate(active_estimators):
            alpha_m = active_alphas[i]

            # Get probabilities from the base estimator
            proba_m_encoded = estimator.predict_proba(X)

            # Ensure proba_m_encoded has columns for all classes_ of the ensemble
            if proba_m_encoded.shape[1] != n_model_classes:
                temp_proba = np.zeros((X.shape[0], n_model_classes))
                # Base estimator's classes_ might be a subset if trained on a bootstrap
                # that didn't contain all original classes.
                est_classes_ = getattr(estimator, 'classes_', None)
                if est_classes_ is None:  # Should not happen with scikit-learn estimators
                    raise ValueError("Base estimator does not have classes_ attribute.")

                for k_idx_estimator, k_class_estimator_encoded in enumerate(est_classes_):
                    # k_class_estimator_encoded is the class label (e.g., 0 or 1 from y_encoded)
                    # We need to find its index in the *ensemble's* self.classes_
                    # This assumes estimator.classes_ are a subset of self.le_.transform(self.classes_)
                    # which should be true if y_bootstrap_encoded was derived from y_encoded.

                    # Find where this estimator's class maps in the main ensemble's classes
                    # This is tricky if the base estimator's classes are not directly the encoded 0,1,...
                    # Assuming base_estimator.classes_ are already the numerically encoded labels (0, 1, ...)
                    # that align with the columns of proba_m_encoded.
                    if int(k_class_estimator_encoded) < n_model_classes:  # Defensive check
                        temp_proba[:, int(k_class_estimator_encoded)] = proba_m_encoded[:, k_idx_estimator]
                proba_m_encoded = temp_proba

            probas_sum += alpha_m * proba_m_encoded

        # Normalize the summed probabilities
        probas_sum /= total_alpha  # Weighted average

        # Ensure probabilities sum to 1 (handle potential floating point inaccuracies)
        norm = np.sum(probas_sum, axis=1, keepdims=True)
        norm[norm == 0] = 1  # Avoid division by zero if all probas are zero (shouldn't happen with active_alphas)
        final_probas = probas_sum / norm

        return final_probas

    def predict(self, X):
        probas = self.predict_proba(X)
        encoded_predictions = np.argmax(probas, axis=1)
        return self.le_.inverse_transform(encoded_predictions)  # Return original class labels