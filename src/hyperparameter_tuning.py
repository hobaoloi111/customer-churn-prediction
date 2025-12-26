"""
BƯỚC 4: GRID SEARCH CV - TỐI ƯU HYPERPARAMETERS
Tìm hyperparameters tốt nhất cho Random Forest
"""
# -*- coding: utf-8 -*-
import sys
import os
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("BƯỚC 4: GRID SEARCH CV - TỐI ƯU HYPERPARAMETERS")
print("="*80)

# ============================================================================
# BƯỚC 1: LOAD & PREPARE DATA
# ============================================================================

print("\n[BƯỚC 1] LOAD & PREPARE DATA")
print("-" * 80)

project_root = Path(__file__).parent.parent
data_file = project_root / 'data' / 'customer_churn_data.csv'

df = pd.read_csv(data_file)

# Chia X, y
target_col = 'churn'
X = df.drop(columns=['customer_id', target_col])
y = df[target_col]

# Train/test split
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

print(f"✓ Data prepared: {X_train_processed.shape}")

# ============================================================================
# BƯỚC 2: DEFINE HYPERPARAMETER GRID
# ============================================================================

print("\n[BƯỚC 2] DEFINE HYPERPARAMETER GRID")
print("-" * 80)

"""
Giải thích các hyperparameters Random Forest:

1. n_estimators: Số lượng cây quyết định trong forest
   - Càng nhiều → Dự đoán tốt hơn nhưng chậm hơn
   - Nên thử: [50, 100, 200]

2. max_depth: Độ sâu tối đa của mỗi cây
   - Hạn chế → Giảm overfitting
   - Nên thử: [5, 10, 15, 20]

3. min_samples_split: Minimum samples yêu cầu để split một node
   - Cao hơn → Cây đơn giản hơn, giảm overfitting
   - Nên thử: [5, 10, 20]

4. min_samples_leaf: Minimum samples cần có ở leaf node
   - Cao hơn → Cây đơn giản hơn
   - Nên thử: [2, 4, 8]

5. max_features: Số features xem xét khi split
   - Giảm → Tăng diversity, giảm correlation
   - Nên thử: ['sqrt', 'log2', None]
"""

param_grid = {
    'n_estimators': [50, 100, 150],      # Số cây
    'max_depth': [7, 10, 15],            # Độ sâu
    'min_samples_split': [5, 10, 20],    # Min samples để split
    'min_samples_leaf': [2, 4],          # Min samples ở leaf
    'max_features': ['sqrt', 'log2']     # Features để xem xét
}

print("\nHyperparameter Grid:")
for param, values in param_grid.items():
    print(f"  {param}: {values}")

total_combinations = 1
for values in param_grid.values():
    total_combinations *= len(values)

print(f"\nTotal combinations: {total_combinations}")
print(f"(Với 5-fold CV → {total_combinations * 5} model training)")

# ============================================================================
# BƯỚC 3: GRID SEARCH CV
# ============================================================================

print("\n[BƯỚC 3] GRID SEARCH CV")
print("-" * 80)

# Base model
base_model = RandomForestClassifier(
    random_state=42,
    n_jobs=-1  # Dùng toàn bộ CPU
)

# GridSearchCV
print("\n⏳ Tìm best hyperparameters...")
print("   (Đây là quá trình tính toán nặng, có thể mất 1-2 phút)")

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=5,                    # 5-fold cross-validation
    scoring='f1',            # Dùng F1-score làm metric chính
    n_jobs=-1,              # Parallel computing
    verbose=1               # Hiển thị progress
)

grid_search.fit(X_train_processed, y_train)

print(f"\n✅ Grid Search Complete!")

# ============================================================================
# BƯỚC 4: BEST PARAMETERS & RESULTS
# ============================================================================

print("\n[BƯỚC 4] BEST PARAMETERS & RESULTS")
print("-" * 80)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"\n🏆 BEST PARAMETERS:")
for param, value in best_params.items():
    print(f"  {param}: {value}")

print(f"\n🎯 BEST CV SCORE (F1): {best_score:.4f}")

# ============================================================================
# BƯỚC 5: COMPARE BEFORE vs AFTER
# ============================================================================

print("\n[BƯỚC 5] SO SÁNH TRƯỚC - SAU TUNING")
print("-" * 80)

# Original model
original_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1
)
original_model.fit(X_train_processed, y_train)

# Predictions
y_pred_original = original_model.predict(X_test_processed)
y_pred_best = best_model.predict(X_test_processed)

y_pred_proba_original = original_model.predict_proba(X_test_processed)[:, 1]
y_pred_proba_best = best_model.predict_proba(X_test_processed)[:, 1]

# Metrics
metrics_original = {
    'Accuracy': accuracy_score(y_test, y_pred_original),
    'F1-Score': f1_score(y_test, y_pred_original),
    'ROC-AUC': roc_auc_score(y_test, y_pred_proba_original)
}

metrics_best = {
    'Accuracy': accuracy_score(y_test, y_pred_best),
    'F1-Score': f1_score(y_test, y_pred_best),
    'ROC-AUC': roc_auc_score(y_test, y_pred_proba_best)
}

# Create comparison table
comparison_data = {
    'Metric': ['Accuracy', 'F1-Score', 'ROC-AUC'],
    'Original': [metrics_original['Accuracy'], metrics_original['F1-Score'], metrics_original['ROC-AUC']],
    'Tuned': [metrics_best['Accuracy'], metrics_best['F1-Score'], metrics_best['ROC-AUC']],
    'Improvement': [
        metrics_best['Accuracy'] - metrics_original['Accuracy'],
        metrics_best['F1-Score'] - metrics_original['F1-Score'],
        metrics_best['ROC-AUC'] - metrics_original['ROC-AUC']
    ]
}

df_comparison = pd.DataFrame(comparison_data)

print("\n📊 PERFORMANCE COMPARISON:")
print(df_comparison.to_string(index=False))

print("\n💹 IMPROVEMENT:")
for _, row in df_comparison.iterrows():
    metric = row['Metric']
    improvement = row['Improvement']
    direction = "✅" if improvement >= 0 else "❌"
    print(f"  {direction} {metric}: {improvement:+.4f} ({improvement*100:+.2f}%)")

# ============================================================================
# BƯỚC 6: DETAILED RESULTS
# ============================================================================

print("\n[BƯỚC 6] DETAILED CLASSIFICATION REPORT (TUNED MODEL)")
print("-" * 80)

print(classification_report(
    y_test, y_pred_best,
    target_names=['Stayed', 'Churned'],
    digits=4
))

# ============================================================================
# BƯỚC 7: CROSS-VALIDATION SCORES
# ============================================================================

print("\n[BƯỚC 7] CROSS-VALIDATION ANALYSIS")
print("-" * 80)

# CV scores cho best model
cv_scores = cross_val_score(
    best_model, X_train_processed, y_train,
    cv=5, scoring='f1', n_jobs=-1
)

print(f"\n5-Fold CV Scores: {[f'{score:.4f}' for score in cv_scores]}")
print(f"Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Interpretation
std = cv_scores.std()
if std < 0.05:
    print(f"✅ Model rất ổn định (std < 0.05)")
elif std < 0.10:
    print(f"🟡 Model khá ổn định (std < 0.10)")
else:
    print(f"⚠️  Model có biến động (std >= 0.10)")

# ============================================================================
# BƯỚC 8: TOP 10 HYPERPARAMETER COMBINATIONS
# ============================================================================

print("\n[BƯỚC 8] TOP 10 HYPERPARAMETER COMBINATIONS")
print("-" * 80)

# Get results from GridSearch
results_df = pd.DataFrame(grid_search.cv_results_)
results_df = results_df[['param_n_estimators', 'param_max_depth', 'param_min_samples_split',
                          'param_min_samples_leaf', 'param_max_features', 'mean_test_score']].copy()

results_df.columns = ['n_estimators', 'max_depth', 'min_samples_split',
                      'min_samples_leaf', 'max_features', 'CV F1-Score']

results_df = results_df.sort_values('CV F1-Score', ascending=False).head(10).reset_index(drop=True)

print("\n" + results_df.to_string())

# ============================================================================
# BƯỚC 9: RECOMMENDATIONS
# ============================================================================

print("\n[BƯỚC 9] RECOMMENDATIONS & BEST PRACTICES")
print("-" * 80)

recommendations = """
✅ BEST PRACTICES KHI DÙNG GRID SEARCH:

1. Scope:
   - Không nên thử quá nhiều hyperparameters (tính toán lâu)
   - Focus vào hyperparameters MỒM QUAN TRỌNG nhất
   - Có thể thử RandomizedSearchCV cho không gian lớn

2. Cross-Validation:
   - Luôn dùng CV (không chỉ train/test split)
   - 5-fold hoặc 10-fold là chuẩn
   - Stratified CV nếu dữ liệu imbalanced

3. Scoring:
   - Chọn scoring phù hợp với bài toán
   - F1 tốt cho imbalanced classification
   - AUC nếu quan tâm tới ranking

4. Computational Cost:
   - GridSearchCV tính toán nặng
   - Dùng n_jobs=-1 để parallel computing
   - Xem xét RandomizedSearchCV nếu quá lâu

5. Overfitting Check:
   - So sánh train vs CV scores
   - Nếu train >> CV → Có overfitting
   - Adjust hyperparameters để giảm complexity

6. KHÔNG nên:
   - Tune hyperparameters trên test set
   - Quá tối ưu hóa cho train set
   - Bỏ qua cross-validation
   - Dùng quá nhiều hyperparameters (curse of dimensionality)

7. Tiếp theo:
   - Xác nhận kết quả trên test set
   - Lưu best model & hyperparameters
   - Deploy & monitor performance trong production
"""

print(recommendations)

# ============================================================================
# BƯỚC 10: SAVE BEST MODEL
# ============================================================================

print("\n[BƯỚC 10] SAVE BEST MODEL")
print("-" * 80)

import joblib

models_dir = project_root / 'models'
models_dir.mkdir(exist_ok=True)

# Save best model
model_path = models_dir / 'random_forest_tuned.pkl'
joblib.dump(best_model, model_path)

print(f"\n✓ Best model saved: {model_path}")

# Save best hyperparameters
params_path = models_dir / 'best_hyperparameters.json'
import json

with open(params_path, 'w') as f:
    json.dump(best_params, f, indent=2)

print(f"✓ Best hyperparameters saved: {params_path}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("✅ BƯỚC 4 HOÀN TẤT - GRID SEARCH CV")
print("="*80)

summary = f"""
🎯 SUMMARY:

📊 Original Model:
  - Accuracy:  {metrics_original['Accuracy']:.4f}
  - F1-Score:  {metrics_original['F1-Score']:.4f}
  - ROC-AUC:   {metrics_original['ROC-AUC']:.4f}

🏆 Tuned Model (Best):
  - Accuracy:  {metrics_best['Accuracy']:.4f}
  - F1-Score:  {metrics_best['F1-Score']:.4f}
  - ROC-AUC:   {metrics_best['ROC-AUC']:.4f}

💹 Improvement:
  - Accuracy:  {(metrics_best['Accuracy'] - metrics_original['Accuracy'])*100:+.2f}%
  - F1-Score:  {(metrics_best['F1-Score'] - metrics_original['F1-Score'])*100:+.2f}%
  - ROC-AUC:   {(metrics_best['ROC-AUC'] - metrics_original['ROC-AUC'])*100:+.2f}%

🎯 Best Hyperparameters:
  - n_estimators: {best_params['n_estimators']}
  - max_depth: {best_params['max_depth']}
  - min_samples_split: {best_params['min_samples_split']}
  - min_samples_leaf: {best_params['min_samples_leaf']}
  - max_features: {best_params['max_features']}

📁 Files Saved:
  ✓ {model_path}
  ✓ {params_path}

🚀 Next Steps:
  Bước 5 - Unit Tests
"""

print(summary)
