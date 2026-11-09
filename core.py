from pathlib import Path
import warnings
import numpy as np
import pandas as pd

NUM = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
FEATURES = ['Type'] + NUM
LABELS = dict(zip(NUM, ['Nhiệt độ không khí (K)', 'Nhiệt độ quá trình (K)', 'Tốc độ quay (rpm)', 'Mô-men xoắn (Nm)', 'Thời gian dụng cụ (min)']))

def validate(frame, metadata):
    if frame.columns.duplicated().any():
        raise ValueError('Tên cột bị trùng.')
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError('Thiếu cột: ' + ', '.join(sorted(missing)))
    x = frame[FEATURES].copy().reset_index(drop=True)
    x['Type'] = x['Type'].astype('string').str.strip().str.upper()
    issues = [[] for _ in range(len(x))]
    outside = [[] for _ in range(len(x))]
    for i, v in enumerate(x['Type']):
        if pd.isna(v) or v not in ['L', 'M', 'H']:
            issues[i].append('Type phải là L, M hoặc H')
    for col in NUM:
        x[col] = pd.to_numeric(x[col], errors='coerce')
        bounds = metadata.get('train_ranges', {}).get(col)
        for i, v in enumerate(x[col]):
            if not np.isfinite(v):
                issues[i].append(f'{col}: thiếu hoặc không phải số hữu hạn')
            elif (v < 0 if col == NUM[-1] else v <= 0):
                issues[i].append(f'{col}: giá trị không hợp lệ')
            elif col in [NUM[2], NUM[4]] and v != np.floor(v):
                issues[i].append(f'{col}: cần số nguyên')
            elif bounds and not bounds['min'] <= v <= bounds['max']:
                outside[i].append(f'{col} ngoài [{bounds["min"]}, {bounds["max"]}]')
    # Phạm vi từng biến không kiểm tra mọi quan hệ vật lý giữa các biến.
    return x, pd.Series(['; '.join(v) for v in issues]), pd.Series(['; '.join(v) for v in outside])

def probability(bundle, x):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='X does not have valid feature names, but LGBMClassifier was fitted with feature names', category=UserWarning)
        model = bundle['pipeline']
        pos = list(model.classes_).index(1)
        return model.predict_proba(x[bundle.get('feature_columns', FEATURES)])[:, pos]

def predict_batch(bundle, frame, metadata, allow_outside=True):
    x, errors, outside = validate(frame, metadata)
    result = frame.copy().reset_index(drop=True)
    result['input_error'] = errors
    result['range_warning'] = outside
    result['probability'] = np.nan
    result['prediction'] = pd.Series(pd.NA, index=result.index, dtype='Int64')
    result['threshold'] = float(bundle['threshold'])
    ok = errors.eq('') & (outside.eq('') | allow_outside)
    if ok.any():
        p = probability(bundle, x.loc[ok])
        result.loc[ok, 'probability'] = p
        result.loc[ok, 'prediction'] = (p >= bundle['threshold']).astype(int)
    return result

def contributions(bundle, x):
    # TreeSHAP native cho LightGBM; nếu hiệu chỉnh, giải thích mô hình nền.
    pipe = bundle.get('explanation_pipeline', bundle['pipeline'])
    if not hasattr(pipe, 'steps'):
        return None
    prep, model = pipe.steps[0][1], pipe.steps[-1][1]
    if model.__class__.__name__ != 'LGBMClassifier':
        return None
    z = prep.transform(x[FEATURES])
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='X does not have valid feature names.*', category=UserWarning)
        values = np.asarray(model.predict(z, pred_contrib=True))[0]
    names = list(prep.get_feature_names_out())
    return pd.DataFrame({'Đặc trưng': names, 'Đóng góp log-odds': values[:-1]}).sort_values('Đóng góp log-odds')
