"""
BƯỚC 1: LƯU VÀ TẢI MÔ HÌNH ML
Hướng dẫn sử dụng joblib để persist (lưu) mô hình đã huấn luyện
"""

import joblib
import pickle
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

print("\n" + "="*80)
print("BƯỚC 1: LƯU VÀ TẢI MÔ HÌNH")
print("="*80)

# ============================================================================
# PHẦN 1: LƯU MÔ HÌNH (Model Serialization)
# ============================================================================

print("\n[PHẦN 1] LƯU MÔ HÌNH ĐƯỚC HUẤN LUYỆN")
print("-" * 80)

# Xác định thư mục lưu model
project_root = Path(__file__).parent.parent
models_dir = project_root / 'models'
models_dir.mkdir(exist_ok=True)

print(f"\n[1.1] Thư mục lưu model: {models_dir}")

# ============================================================================
# PHẦN 2: TẠO & HUẤN LUYỆN MÔ HÌNH (Giống trong model_training.py)
# ============================================================================

print("\n[PHẦN 2] LOAD DỮ LIỆU & HUẤN LUYỆN MÔ HÌNH")
print("-" * 80)

# Load dữ liệu
data_file = project_root / 'data' / 'customer_churn_data.csv'
df = pd.read_csv(data_file)

# Chia X, y
target_col = 'churn'
X = df.drop(columns=['customer_id', target_col])
y = df[target_col]

# Chia train/test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessing
numerical_cols = X_train.select_dtypes(include=['number']).columns.tolist()
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

X_train_processed = X_train.copy()
X_test_processed = X_test.copy()

# Label Encoding
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_train_processed[col] = le.fit_transform(X_train[col])
    X_test_processed[col] = le.transform(X_test[col])
    label_encoders[col] = le

# StandardScaling
scaler = StandardScaler()
X_train_processed[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test_processed[numerical_cols] = scaler.transform(X_test[numerical_cols])

print(f"\n[2.1] Dataset loaded: {X_train_processed.shape}")
print(f"[2.2] Preprocessing complete")

# Huấn luyện 3 mô hình
print(f"\n[2.3] Training models...")

lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_processed, y_train)
print(f"  ✓ Logistic Regression trained")

dt_model = DecisionTreeClassifier(max_depth=7, min_samples_split=10, random_state=42)
dt_model.fit(X_train_processed, y_train)
print(f"  ✓ Decision Tree trained")

rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train_processed, y_train)
print(f"  ✓ Random Forest trained")

# ============================================================================
# PHẦN 3: LƯU CÁC MODELS BẰNG JOBLIB
# ============================================================================

print("\n[PHẦN 3] LƯU MÔ HÌNH BẰNG JOBLIB")
print("-" * 80)

# 3.1 Lưu từng model
models_to_save = {
    'logistic_regression': lr_model,
    'decision_tree': dt_model,
    'random_forest': rf_model
}

saved_models = {}
for model_name, model in models_to_save.items():
    model_path = models_dir / f'{model_name}.pkl'
    joblib.dump(model, model_path)
    saved_models[model_name] = str(model_path)
    print(f"\n[3.1] Saved: {model_name}")
    print(f"  Location: {model_path}")
    print(f"  Size: {model_path.stat().st_size / 1024:.2f} KB")

# 3.2 Lưu preprocessor (Scaler & Encoders)
print(f"\n[3.2] Saving preprocessors...")

# Lưu Scaler
scaler_path = models_dir / 'scaler.pkl'
joblib.dump(scaler, scaler_path)
print(f"  ✓ Scaler: {scaler_path}")

# Lưu Label Encoders
encoders_path = models_dir / 'label_encoders.pkl'
joblib.dump(label_encoders, encoders_path)
print(f"  ✓ Label Encoders: {encoders_path}")

# 3.3 Lưu Feature Names (quan trọng!)
print(f"\n[3.3] Saving feature information...")

feature_info = {
    'feature_names': X_train.columns.tolist(),
    'numerical_cols': numerical_cols,
    'categorical_cols': categorical_cols
}

feature_info_path = models_dir / 'feature_info.json'
with open(feature_info_path, 'w') as f:
    json.dump(feature_info, f, indent=2)
print(f"  ✓ Feature Info: {feature_info_path}")

# ============================================================================
# PHẦN 4: LƯU METADATA (THÔNG TIN MÔ HÌNH)
# ============================================================================

print(f"\n[PHẦN 4] LƯU METADATA")
print("-" * 80)

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Tính metrics trên test set
metrics_dict = {}
for model_name, model in models_to_save.items():
    y_pred = model.predict(X_test_processed)
    y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
    
    metrics_dict[model_name] = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'f1_score': float(f1_score(y_test, y_pred)),
        'roc_auc': float(roc_auc_score(y_test, y_pred_proba))
    }

# Tạo metadata object
metadata = {
    'timestamp': datetime.now().isoformat(),
    'models': {
        'logistic_regression': {
            'path': saved_models['logistic_regression'],
            'type': 'LogisticRegression',
            'hyperparameters': {
                'random_state': 42,
                'max_iter': 1000
            },
            'metrics': metrics_dict['logistic_regression']
        },
        'decision_tree': {
            'path': saved_models['decision_tree'],
            'type': 'DecisionTreeClassifier',
            'hyperparameters': {
                'max_depth': 7,
                'min_samples_split': 10,
                'random_state': 42
            },
            'metrics': metrics_dict['decision_tree']
        },
        'random_forest': {
            'path': saved_models['random_forest'],
            'type': 'RandomForestClassifier',
            'hyperparameters': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 10,
                'random_state': 42
            },
            'metrics': metrics_dict['random_forest']
        }
    },
    'preprocessors': {
        'scaler': str(scaler_path),
        'label_encoders': str(encoders_path)
    },
    'feature_info': str(feature_info_path),
    'training_data': {
        'total_samples': len(df),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'churn_rate': float(y.sum() / len(y))
    }
}

# Lưu metadata vào JSON
metadata_path = models_dir / 'model_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n[4.1] Metadata saved: {metadata_path}")
print(f"\n[4.2] Model Performance Summary:")
for model_name, metrics in metrics_dict.items():
    print(f"\n  {model_name}:")
    print(f"    - Accuracy: {metrics['accuracy']:.4f}")
    print(f"    - F1 Score: {metrics['f1_score']:.4f}")
    print(f"    - ROC-AUC:  {metrics['roc_auc']:.4f}")

# ============================================================================
# PHẦN 5: TẢI MODEL (DEMO)
# ============================================================================

print("\n" + "="*80)
print("PHẦN 5: DEMO - TẢI MÔ HÌNH VÀ PREDICT")
print("="*80)

# 5.1 Tải metadata
print(f"\n[5.1] Loading metadata...")
with open(metadata_path, 'r') as f:
    loaded_metadata = json.load(f)
print(f"  ✓ Metadata loaded")
print(f"  Timestamp: {loaded_metadata['timestamp']}")

# 5.2 Tải models
print(f"\n[5.2] Loading models...")

loaded_models = {}
for model_name in ['logistic_regression', 'decision_tree', 'random_forest']:
    model_path = models_dir / f'{model_name}.pkl'
    loaded_models[model_name] = joblib.load(model_path)
    print(f"  ✓ {model_name} loaded")

# 5.3 Tải preprocessors
print(f"\n[5.3] Loading preprocessors...")

loaded_scaler = joblib.load(scaler_path)
loaded_encoders = joblib.load(encoders_path)

with open(feature_info_path, 'r') as f:
    loaded_feature_info = json.load(f)

print(f"  ✓ Scaler loaded")
print(f"  ✓ Label Encoders loaded")
print(f"  ✓ Feature Info loaded")

# 5.4 Demo prediction
print(f"\n[5.4] DEMO PREDICTION")
print(f"  Sử dụng mẫu từ test set...")

# Lấy 1 mẫu từ test set
demo_sample = X_test.iloc[0:1].copy()
print(f"\n  Sample input:")
print(demo_sample)

# Preprocessing
demo_processed = demo_sample.copy()
for col in loaded_feature_info['categorical_cols']:
    demo_processed[col] = loaded_encoders[col].transform(demo_sample[col])

demo_processed[loaded_feature_info['numerical_cols']] = loaded_scaler.transform(
    demo_sample[loaded_feature_info['numerical_cols']]
)

print(f"\n  Predictions:")
for model_name, model in loaded_models.items():
    pred = model.predict(demo_processed)[0]
    proba = model.predict_proba(demo_processed)[0]
    
    print(f"\n  {model_name}:")
    print(f"    - Prediction: {pred} ({'Stayed' if pred == 0 else 'Churned'})")
    print(f"    - Probability: Stayed={proba[0]:.2%}, Churned={proba[1]:.2%}")

print(f"\n  Ground Truth: {y_test.iloc[0]} ({'Stayed' if y_test.iloc[0] == 0 else 'Churned'})")

# ============================================================================
# PHẦN 6: BEST PRACTICES
# ============================================================================

print("\n" + "="*80)
print("PHẦN 6: BEST PRACTICES KHI LƯU/TẢI MÔ HÌNH")
print("="*80)

best_practices = """
✅ DO's (Nên làm):
  1. Luôn lưu preprocessing objects (scaler, encoders) cùng với model
  2. Lưu feature names để đảm bảo input đúng thứ tự
  3. Lưu metadata (accuracy, timestamp, hyperparameters)
  4. Dùng joblib thay vì pickle (nhanh hơn cho numpy arrays)
  5. Kiểm tra version thư viện (scikit-learn, pandas, numpy)
  6. Lưu riêng cho mỗi mô hình để dễ quản lý
  7. Tên file rõ ràng với timestamp (e.g., model_20231226_v2.pkl)

❌ DON'Ts (Không nên làm):
  1. Không lưu mô hình mà không có scaler
  2. Không xóa label encoders - cần dùng lại khi predict
  3. Không lưu model từ pickle (deprecated, bảo mật kém)
  4. Không quên feature order
  5. Không test lại model sau load trước dùng production

⚠️ Lưu ý:
  - Model size lớn? Dùng cloud storage (S3, GCS)
  - Nhiều version? Dùng version control (DVC, MLflow)
  - Production? Cần monitoring & versioning (MLOps)
  - Bảo mật? Encrypt model files trước lưu
"""

print(best_practices)

# ============================================================================
# PHẦN 7: HELPER FUNCTIONS
# ============================================================================

print("\n[PHẦN 7] HELPER FUNCTIONS - Tái sử dụng")
print("-" * 80)

# Tạo file helper.py
helper_code = '''
"""
Helper functions để lưu/tải mô hình một cách dễ dàng
"""

import joblib
import json
from pathlib import Path
from datetime import datetime

def save_model_package(model, scaler, encoders, feature_info, 
                       model_name='my_model', models_dir='models'):
    """
    Lưu mô hình + preprocessors + metadata một lúc
    
    Args:
        model: Trained model (sklearn)
        scaler: StandardScaler object
        encoders: Dict of LabelEncoders
        feature_info: Dict with feature_names, numerical_cols, categorical_cols
        model_name: Tên model (str)
        models_dir: Thư mục lưu (str or Path)
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(exist_ok=True)
    
    # Lưu model
    model_path = models_dir / f'{model_name}_model.pkl'
    joblib.dump(model, model_path)
    
    # Lưu scaler
    scaler_path = models_dir / f'{model_name}_scaler.pkl'
    joblib.dump(scaler, scaler_path)
    
    # Lưu encoders
    encoders_path = models_dir / f'{model_name}_encoders.pkl'
    joblib.dump(encoders, encoders_path)
    
    # Lưu feature info
    feature_info_path = models_dir / f'{model_name}_features.json'
    with open(feature_info_path, 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    # Tạo metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'model_name': model_name,
        'files': {
            'model': str(model_path),
            'scaler': str(scaler_path),
            'encoders': str(encoders_path),
            'features': str(feature_info_path)
        }
    }
    
    metadata_path = models_dir / f'{model_name}_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Model package saved: {model_name}")
    return metadata

def load_model_package(model_name='my_model', models_dir='models'):
    """
    Tải mô hình + preprocessors đầy đủ
    
    Args:
        model_name: Tên model (str)
        models_dir: Thư mục lưu (str or Path)
    
    Returns:
        Tuple: (model, scaler, encoders, feature_info)
    """
    models_dir = Path(models_dir)
    
    # Tải model
    model_path = models_dir / f'{model_name}_model.pkl'
    model = joblib.load(model_path)
    
    # Tải scaler
    scaler_path = models_dir / f'{model_name}_scaler.pkl'
    scaler = joblib.load(scaler_path)
    
    # Tải encoders
    encoders_path = models_dir / f'{model_name}_encoders.pkl'
    encoders = joblib.load(encoders_path)
    
    # Tải feature info
    feature_info_path = models_dir / f'{model_name}_features.json'
    with open(feature_info_path, 'r') as f:
        feature_info = json.load(f)
    
    print(f"✓ Model package loaded: {model_name}")
    return model, scaler, encoders, feature_info
'''

helper_path = project_root / 'src' / 'model_helper.py'
with open(helper_path, 'w', encoding='utf-8') as f:
    f.write(helper_code)

print(f"\n[7.1] Helper functions saved: {helper_path}")
print(f"\nUsage example:")
print(f"""
from src.model_helper import save_model_package, load_model_package

# Lưu model
save_model_package(rf_model, scaler, encoders, feature_info, 
                   model_name='churn_rf_v1', models_dir='models')

# Tải model
model, scaler, encoders, features = load_model_package(
    model_name='churn_rf_v1', models_dir='models')
""")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("✅ SUMMARY - BƯỚC 1 HOÀN TẤT")
print("="*80)

summary = f"""
📁 FILES SAVED:
  ✓ {models_dir / 'logistic_regression.pkl'} ({(models_dir / 'logistic_regression.pkl').stat().st_size / 1024:.1f} KB)
  ✓ {models_dir / 'decision_tree.pkl'} ({(models_dir / 'decision_tree.pkl').stat().st_size / 1024:.1f} KB)
  ✓ {models_dir / 'random_forest.pkl'} ({(models_dir / 'random_forest.pkl').stat().st_size / 1024:.1f} KB)
  ✓ {scaler_path}
  ✓ {encoders_path}
  ✓ {feature_info_path}
  ✓ {metadata_path}
  ✓ {helper_path}

🎯 NEXT STEP:
  Bước 2 - Build CLI Application (dùng các model đã lưu)
"""

print(summary)
