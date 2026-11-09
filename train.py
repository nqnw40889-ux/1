"""Compare class imbalance treatments and freeze the policy before scoring test."""
import argparse
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from core import FEATURES, NUM, probability, validate
from evaluation import build_reports, measure
from policy import DEFAULT_MISS_COST, FALSE_ALARM_COST, cost_of, select_thresholds, threshold_curve

SEED = 42
TARGET = 'Machine failure'
MODEL_PACKAGES = ['numpy', 'pandas', 'scikit-learn', 'lightgbm', 'xgboost', 'joblib']


def load_splits(path):
    raw = pd.read_csv(path)
    raw.columns = raw.columns.str.strip()
    inputs, errors, _ = validate(raw, {})
    if errors.ne('').any():
        raise ValueError(f'CSV có lỗi đầu vào tại dòng {errors.ne("").idxmax() + 2}.')
    labels = pd.to_numeric(raw[TARGET], errors='raise')
    if not labels.isin([0, 1]).all() or labels.nunique() != 2:
        raise ValueError('Cột Machine failure phải có cả hai lớp 0 và 1.')
    x_train, x_holdout, y_train, y_holdout = train_test_split(
        inputs, labels.astype(int), test_size=.30, stratify=labels, random_state=SEED
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_holdout, y_holdout, test_size=.50, stratify=y_holdout, random_state=SEED
    )
    return {'Train': (x_train, y_train), 'Validation': (x_val, y_val), 'Test': (x_test, y_test)}


def resample_training(x_train, y_train):
    # Use standardized distances so rpm does not dominate neighbor selection.
    scaler = StandardScaler()
    scaled = x_train.copy().astype(dict.fromkeys(NUM, float))
    scaled[NUM] = scaler.fit_transform(x_train[NUM])
    sampler = SMOTENC(categorical_features=['Type'], random_state=SEED, sampling_strategy=1.0)
    sampled, labels = sampler.fit_resample(scaled, y_train)
    sampled[NUM] = scaler.inverse_transform(sampled[NUM])
    # Keep the count/time variables integral in synthetic rows.
    for name in (NUM[2], NUM[4]):
        sampled[name] = sampled[name].round().astype(int)
    return sampled, labels


def models(weighted, imbalance_ratio):
    class_weight = 'balanced' if weighted else None
    positive_weight = imbalance_ratio if weighted else 1.0
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=2000, class_weight=class_weight, random_state=SEED),
        'Random Forest': RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight=class_weight, n_jobs=2, random_state=SEED),
        'XGBoost': XGBClassifier(
            n_estimators=300, learning_rate=.05, max_depth=4, min_child_weight=1,
            subsample=.8, colsample_bytree=.8, reg_lambda=1, scale_pos_weight=positive_weight,
            objective='binary:logistic', eval_metric='logloss', tree_method='hist', n_jobs=2, random_state=SEED),
        'LightGBM': LGBMClassifier(
            n_estimators=300, learning_rate=.05, num_leaves=15, max_depth=4, min_child_samples=20,
            subsample=.8, subsample_freq=1, colsample_bytree=.8, reg_lambda=1,
            scale_pos_weight=positive_weight, verbosity=-1, n_jobs=2, random_state=SEED),
    }


def pipeline(model, algorithm):
    preprocessing = ColumnTransformer([
        ('numeric', StandardScaler() if algorithm == 'Logistic Regression' else 'passthrough', NUM),
        ('type', OneHotEncoder(categories=[['L', 'M', 'H']], drop=['L'],
                               sparse_output=False, handle_unknown='error'), ['Type']),
    ])
    return Pipeline([('preprocessing', preprocessing), ('model', model)])


def compare_models(x_train, y_train, x_val, y_val, miss_cost):
    x_resampled, y_resampled = resample_training(x_train, y_train)
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    comparison, fitted, validation_scores = [], {}, {}
    for method, x_fit, y_fit in [
        ('Original', x_train, y_train),
        ('Weighted', x_train, y_train),
        ('SMOTENC', x_resampled, y_resampled),
    ]:
        for algorithm, model in models(method == 'Weighted', ratio).items():
            name = f'{algorithm} / {method}'
            print(f'Huấn luyện {name}', flush=True)
            fitted[name] = pipeline(model, algorithm).fit(x_fit, y_fit)
            scores = probability({'pipeline': fitted[name]}, x_val)
            validation_scores[name] = scores
            cutoff = select_thresholds(threshold_curve(y_val, scores), miss_cost)['Cost']
            metrics = measure(y_val, scores, cutoff)
            comparison.append({
                'Model': name, 'Algorithm': algorithm, 'Method': method,
                'Fit normal': int((y_fit == 0).sum()), 'Fit failure': int((y_fit == 1).sum()),
                'Threshold': cutoff, **metrics,
                'Cost': cost_of(metrics, miss_cost),
                'F1 at 0.5': measure(y_val, scores, .5)['F1'],
            })
    comparison = pd.DataFrame(comparison).sort_values(['AP', 'Cost', 'Model'], ascending=[False, True, True])
    winner = comparison.iloc[0]['Model']
    return comparison, winner, fitted[winner], validation_scores[winner]


def write_policy_report(path, splits, scores_by_split, thresholds, miss_cost):
    rows = []
    for split in ('Validation', 'Test'):
        _, y = splits[split]
        p = scores_by_split[split]
        for rule, scores, cutoff in [
            ('Cost', p, thresholds['Cost']),
            ('F1', p, thresholds['F1']),
            ('0.5', p, .5),
            ('All normal', np.zeros(len(y)), .5),
        ]:
            m = measure(y, scores, cutoff)
            rows.append({'Split': split, 'Policy': rule, 'Threshold': cutoff, **m,
                         'Cost': cost_of(m, miss_cost)})
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', default='data/ai4i2020.csv', type=Path)
    parser.add_argument('--output', default='trained_run', type=Path)
    parser.add_argument('--miss-cost', default=DEFAULT_MISS_COST, type=int)
    args = parser.parse_args()
    if args.miss_cost < 1:
        parser.error('--miss-cost phải ít nhất là 1.')
    dest = args.output
    if dest.exists() and any(dest.iterdir()):
        parser.error('Chọn thư mục đầu ra mới để giữ thực nghiệm đã lưu.')
    reports = dest / 'reports'
    reports.mkdir(parents=True, exist_ok=True)
    data_dir = dest / 'data'
    data_dir.mkdir()
    splits = load_splits(args.csv)
    x_train, y_train = splits['Train']
    x_val, y_val = splits['Validation']

    comparison, name, model, scores = compare_models(x_train, y_train, x_val, y_val, args.miss_cost)
    comparison.to_csv(reports / 'validation_comparison.csv', index=False)
    thresholds = select_thresholds(threshold_curve(y_val, scores), args.miss_cost)
    metadata = {
        'dataset': 'AI4I 2020 synthetic',
        'dataset_sha256': hashlib.sha256(args.csv.read_bytes()).hexdigest(),
        'split_seed': SEED,
        'train_ranges': {col: {'min': float(x_train[col].min()), 'max': float(x_train[col].max())}
                         for col in NUM},
        'selection': 'Validation AP; minimum validation cost at fixed ratio; F1 as reference',
        'calibrated': False,
        'miss_cost': args.miss_cost,
        'false_alarm_cost': FALSE_ALARM_COST,
        'policies': thresholds,
        'python': sys.version,
        'versions': {p: version(p) for p in MODEL_PACKAGES + ['imbalanced-learn']},
    }
    bundle = {'pipeline': model, 'threshold': thresholds['Cost'], 'model_name': name,
              'feature_columns': FEATURES, 'metadata': metadata}
    # Freeze the classifier and both thresholds before scoring the test set.
    joblib.dump(bundle, dest / 'mo_hinh_demo.joblib', compress=3)
    (dest / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    table = build_reports(bundle, splits, reports)
    scores_by_split = {}
    counts = []
    for split, (x, y) in splits.items():
        name = split.lower()
        positives = int(y.sum())
        counts.append({'Split': split, 'Samples': len(x), 'Normal': len(x) - positives,
                       'Failure': positives, 'Failure rate': positives / len(x)})
        data = x.copy()
        data.insert(0, 'source_row', x.index)
        data[TARGET] = y.to_numpy()
        data.to_csv(data_dir / f'{name}_goc.csv', index=False)
        pred_path = reports / f'predictions_{name}.csv'
        pred = pd.read_csv(pred_path, float_precision='round_trip')
        scores_by_split[split] = pred['probability'].to_numpy()
        pred['threshold'] = thresholds['Cost']
        pred['prediction'] = (pred['probability'] >= thresholds['Cost']).astype(int)
        pred.to_csv(pred_path, index=False)

    pd.DataFrame(counts).to_csv(reports / 'split_counts.csv', index=False)
    write_policy_report(reports / 'policy_comparison.csv', splits, scores_by_split, thresholds, args.miss_cost)
    print(table.to_string(index=False))
    print(f'Đã lưu kết quả tại {dest}')


if __name__ == '__main__':
    main()
