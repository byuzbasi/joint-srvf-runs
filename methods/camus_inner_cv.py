"""Fold-local representation learning for the fixed external CAMUS split.

Only arrays from the external training set enter this module. Fold features
are built once per eligible training-row set, then reused across estimators.
The existing lambda selector is called inside each fold's training subset.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (balanced_accuracy_score, mean_squared_error,
                             roc_auc_score, silhouette_score)
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.preprocessing import StandardScaler


def prepare_folds(beta, r, embedded, row_ids, fit_representation,
                  n_folds=5, random_state=2026):
    """Learn a fresh representation from each inner training subset only.

    fit_representation receives three training arrays and three validation
    arrays. It must fit using its training arguments and only transform its
    validation arguments. The adapter reuses the notebook's original steps.
    """
    arrays = [np.asarray(x) for x in (beta, r, embedded)]
    row_ids = np.asarray(row_ids)
    n = len(row_ids)
    if any(len(x) != n for x in arrays) or len(np.unique(row_ids)) != n:
        raise ValueError('Raw arrays and unique row_ids must describe the same rows.')
    k = min(n_folds, n)
    if k < 2:
        raise ValueError('At least two inner folds are required.')
    folds = []
    for fold_id, (train, validation) in enumerate(KFold(
            n_splits=k, shuffle=True, random_state=random_state).split(row_ids)):
        X_train, X_validation, metadata = fit_representation(
            *(x[train] for x in arrays), *(x[validation] for x in arrays))
        X_train = pd.DataFrame(X_train).reset_index(drop=True)
        X_validation = pd.DataFrame(X_validation).reset_index(drop=True)
        if len(X_train) != len(train) or len(X_validation) != len(validation):
            raise ValueError('Fold representation changed the number/order of rows.')
        if list(X_train.columns) != list(X_validation.columns):
            raise ValueError('Training and validation feature columns differ.')
        if not np.all(np.isfinite(X_train)) or not np.all(np.isfinite(X_validation)):
            raise ValueError('Fold representation contains nonfinite features.')
        folds.append({'fold': fold_id, 'train': train, 'validation': validation,
                      'train_row_ids': row_ids[train].copy(),
                      'validation_row_ids': row_ids[validation].copy(),
                      'X_train': X_train, 'X_validation': X_validation,
                      'representation': metadata})
    return folds


def validate_folds(folds, n):
    if folds is None:
        raise ValueError('Supply fold-local raw-data representations; global PCA features cannot be used for inner CV.')
    seen = np.zeros(n, dtype=int)
    for fold in folds:
        tr, va = np.asarray(fold['train']), np.asarray(fold['validation'])
        if len(np.unique(tr)) != len(tr) or len(np.unique(va)) != len(va):
            raise ValueError('Duplicate inner-fold row positions.')
        if len(np.intersect1d(tr, va)) or not np.array_equal(
                np.sort(np.r_[tr, va]), np.arange(n)):
            raise ValueError('Inner folds must partition the eligible external-training rows.')
        if len(fold['X_train']) != len(tr) or len(fold['X_validation']) != len(va):
            raise ValueError('Feature rows do not match fold row positions.')
        seen[va] += 1
    if len(folds) < 2 or not np.all(seen == 1):
        raise ValueError('Each eligible row must be validated exactly once.')


def tune_supervised(spec, X_train, y_train, folds, task):
    """Keep the original grids/scores; clone every fitted candidate."""
    y = np.asarray(y_train)
    if len(X_train) != len(y):
        raise ValueError('Final training features and outcome lengths differ.')
    validate_folds(folds, len(y))
    if task not in ('regression', 'classification'):
        raise ValueError('Unknown supervised task.')
    best_score = np.inf if task == 'regression' else -np.inf
    best_params = None
    for params in ParameterGrid(spec['params'] or {}):
        scores = []
        for fold in folds:
            y_fit, y_validation = y[fold['train']], y[fold['validation']]
            if task == 'classification' and (
                    len(np.unique(y_fit)) < 2 or len(np.unique(y_validation)) < 2):
                # Preserve the original one-class-fold exclusion rule.
                continue
            model = clone(spec['model']).set_params(**params)
            model.fit(fold['X_train'], y_fit)
            if task == 'regression':
                score = np.sqrt(mean_squared_error(
                    y_validation, model.predict(fold['X_validation'])))
            elif hasattr(model, 'predict_proba'):
                score = roc_auc_score(y_validation, model.predict_proba(
                    fold['X_validation'])[:, 1])
            else:
                score = balanced_accuracy_score(
                    y_validation, model.predict(fold['X_validation']))
            scores.append(score)
        if not scores:
            continue
        score = float(np.mean(scores))
        better = score < best_score if task == 'regression' else score > best_score
        if better:
            best_params, best_score = dict(params), score
    if best_params is None:
        raise ValueError('No valid inner CV scores; final refit was not performed.')
    # These features were fitted on the entire external-training set only.
    final = clone(spec['model']).set_params(**best_params)
    final.fit(X_train, y)
    return final, best_params, best_score


def select_clusters(X_train_scaled, folds, model_specs, sizes=(2, 3, 4, 5, 6)):
    """Refit the clustering scaler on each inner training feature matrix."""
    validate_folds(folds, len(X_train_scaled))
    scaled = []
    for fold in folds:
        scaler = StandardScaler()
        training = scaler.fit_transform(fold['X_train'])
        validation = scaler.transform(fold['X_validation'])
        scaled.append((training, validation))
    results, selected = [], None
    best_score = -np.inf
    for name, spec in model_specs.items():
        for k in sizes:
            scores = []
            for training, validation in scaled:
                if len(training) <= k or len(validation) <= k:
                    continue
                model = spec['fit'](training, k)
                if not hasattr(model, 'predict'):
                    raise ValueError('Validation must use predict; fitting clusters on validation is not allowed.')
                labels = model.predict(validation)
                if len(np.unique(labels)) < 2:
                    continue
                scores.append(silhouette_score(validation, labels))
            if not scores:
                continue
            score = float(np.mean(scores))
            results.append({'model': name, 'k': k, 'cv_silhouette': score})
            if score > best_score:
                best_score, selected = score, (name, k)
    if selected is None:
        raise ValueError('No valid clustering CV scores; final refit was not performed.')
    name, k = selected
    return model_specs[name]['fit'](X_train_scaled, k), name, k, pd.DataFrame(results)


def audit_folds(folds):
    """JSON-compatible trace without persisting per-fold data matrices."""
    return [{'fold': int(f['fold']), 'n_train': len(f['train']),
             'n_validation': len(f['validation']),
             'train_row_ids': f['train_row_ids'].tolist(),
             'validation_row_ids': f['validation_row_ids'].tolist(),
             'feature_columns': list(f['X_train'].columns),
             'representation': f['representation']} for f in folds]
