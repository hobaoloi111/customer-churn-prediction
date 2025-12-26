"""
Complete ML Pipeline for Customer Churn Prediction
Bài toán: Binary Classification - Dự đoán khách hàng rời đi

Các bước:
1. Load dữ liệu từ CSV
2. Chia train/test split (80/20)
3. Preprocessing: Scaling + Encoding
4. Huấn luyện 3 mô hình ML
5. Đánh giá trên test set
6. Visualization kết quả
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Scikit-learn imports
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report, 
                             roc_auc_score, roc_curve)

# Cấu hình matplotlib
plt.switch_backend('Agg')
sns.set_style("whitegrid")

# ============================================================================
# BƯỚC 1: LOAD DỮ LIỆU
# ============================================================================

print("\n" + "="*80)
print("BƯỚC 1: LOAD DỮ LIỆU TỪ CSV")
print("="*80)

# Xác định đường dẫn file
project_root = Path(__file__).parent.parent
data_file = project_root / 'data' / 'customer_churn_data.csv'

print(f"\n[1.1] Đang load dữ liệu từ: {data_file}")
df = pd.read_csv(data_file)

print(f"[1.2] Dataset shape: {df.shape}")
print(f"[1.3] Cột dữ liệu:")
print(df.columns.tolist())
print(f"\n[1.4] 5 hàng đầu tiên:")
print(df.head())

# ============================================================================
# BƯỚC 2: PHÂN TÁCH FEATURES VÀ TARGET
# ============================================================================

print("\n" + "="*80)
print("BƯỚC 2: PHÂN TÁCH FEATURES VÀ TARGET")
print("="*80)

# Xác định target column
target_col = 'churn'
X = df.drop(columns=['customer_id', target_col])  # Bỏ ID và target
y = df[target_col]

print(f"\n[2.1] Target distribution:")
print(y.value_counts())
print(f"  - Churn rate: {(y.sum()/len(y))*100:.1f}%")

print(f"\n[2.2] Features sẽ dùng: {X.columns.tolist()}")
print(f"[2.3] Số features: {X.shape[1]}")

# ============================================================================
# BƯỚC 3: CHIA TRAIN/TEST SPLIT (80/20)
# ============================================================================

print("\n" + "="*80)
print("BƯỚC 3: CHIA TRAIN/TEST SPLIT (80/20)")
print("="*80)

# Chia dữ liệu - stratify để đảm bảo tỷ lệ class giống nhau
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,           # 20% test
    random_state=42,         # Reproducible
    stratify=y               # Giữ tỷ lệ churn giống
)

print(f"\n[3.1] Train set size: {X_train.shape[0]}")
print(f"[3.2] Test set size: {X_test.shape[0]}")
print(f"\n[3.3] Train set target distribution:")
print(y_train.value_counts())
print(f"\n[3.4] Test set target distribution:")
print(y_test.value_counts())

# ============================================================================
# BƯỚC 4: PREPROCESSING DỮ LIỆU
# ============================================================================

print("\n" + "="*80)
print("BƯỚC 4: PREPROCESSING DỮ LIỆU (SCALING + ENCODING)")
print("="*80)

# 4.1 Xác định cột numerical và categorical
numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

print(f"\n[4.1] Numerical columns: {numerical_cols}")
print(f"[4.2] Categorical columns: {categorical_cols}")

# 4.2 Xử lý categorical features - Label Encoding
print(f"\n[4.3] Label Encoding các categorical features:")
X_train_processed = X_train.copy()
X_test_processed = X_test.copy()

label_encoders = {}  # Lưu encoder để apply trên test
for col in categorical_cols:
    le = LabelEncoder()
    # Fit trên train set
    X_train_processed[col] = le.fit_transform(X_train[col])
    # Transform test set với encoder từ train
    X_test_processed[col] = le.transform(X_test[col])
    label_encoders[col] = le
    
    print(f"  {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# 4.3 Xử lý numerical features - StandardScaler (chuẩn hóa)
print(f"\n[4.4] Standardization (chuẩn hóa) numerical features:")
scaler = StandardScaler()
X_train_processed[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test_processed[numerical_cols] = scaler.transform(X_test[numerical_cols])

print(f"  Mean sau chuẩn hóa: {X_train_processed[numerical_cols].mean().round(3).tolist()}")
print(f"  Std sau chuẩn hóa: {X_train_processed[numerical_cols].std().round(3).tolist()}")

print(f"\n[4.5] Dữ liệu sau preprocessing:")
print(f"  X_train shape: {X_train_processed.shape}")
print(f"  X_test shape: {X_test_processed.shape}")
print(f"  5 hàng đầu tiên (X_train_processed):")
print(X_train_processed.head())

# ============================================================================
# BƯỚC 5: HUẤN LUYỆN CÁC MÔ HÌNH
# ============================================================================

print("\n" + "="*80)
print("BƯỚC 5: HUẤN LUYỆN CÁC MÔ HÌNH")
print("="*80)

# Dictionary để lưu kết quả
models = {}

# 5.1 Logistic Regression
print(f"\n[5.1] Training Logistic Regression...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_processed, y_train)
models['Logistic Regression'] = lr_model
print(f"  ✓ Training complete")

# 5.2 Decision Tree
print(f"\n[5.2] Training Decision Tree Classifier...")
dt_model = DecisionTreeClassifier(
    max_depth=7,           # Hạn chế độ sâu để tránh overfitting
    min_samples_split=10,  # Minimum samples để split
    random_state=42
)
dt_model.fit(X_train_processed, y_train)
models['Decision Tree'] = dt_model
print(f"  ✓ Training complete")

# 5.3 Random Forest
print(f"\n[5.3] Training Random Forest Classifier...")
rf_model = RandomForestClassifier(
    n_estimators=100,      # Số trees trong forest
    max_depth=10,          # Độ sâu tối đa
    min_samples_split=10,  # Minimum samples để split
    random_state=42,
    n_jobs=-1              # Sử dụng toàn bộ CPU cores
)
rf_model.fit(X_train_processed, y_train)
models['Random Forest'] = rf_model
print(f"  ✓ Training complete")

# ============================================================================
# BƯỚC 6: ĐÁNH GIÁ MÔ HÌNH TRÊN TEST SET
# ============================================================================

print("\n" + "="*80)
print("BƯỚC 6: ĐÁNH GIÁ MÔ HÌNH TRÊN TEST SET")
print("="*80)

# Dictionary để lưu results
results = {}

for model_name, model in models.items():
    print(f"\n[6.{list(models.keys()).index(model_name) + 1}] {model_name}")
    print("-" * 80)
    
    # Predictions
    y_pred = model.predict(X_test_processed)
    y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Lưu results
    results[model_name] = {
        'model': model,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc
    }
    
    # In ra metrics
    print(f"\n  📊 METRICS:")
    print(f"    Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"    Precision: {prec:.4f} (Trong số dự đoán churn, bao nhiêu đúng)")
    print(f"    Recall:    {rec:.4f} (Trong số khách hàng churn, dự đoán đúng bao nhiêu)")
    print(f"    F1 Score:  {f1:.4f} (Điểm cân bằng Precision-Recall)")
    print(f"    ROC-AUC:   {auc:.4f} (Khả năng phân biệt 2 class)")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  📈 CONFUSION MATRIX:")
    print(f"    True Negatives:  {cm[0, 0]:>3} (Dự đoán Stayed - Đúng)")
    print(f"    False Positives: {cm[0, 1]:>3} (Dự đoán Churned - Sai)")
    print(f"    False Negatives: {cm[1, 0]:>3} (Dự đoán Stayed - Sai)")
    print(f"    True Positives:  {cm[1, 1]:>3} (Dự đoán Churned - Đúng)")
    
    # Classification Report
    print(f"\n  📋 CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, 
                              target_names=['Stayed', 'Churned']))

# ============================================================================
# BƯỚC 7: SO SÁNH CÁC MÔ HÌNH
# ============================================================================

print("\n" + "="*80)
print("BƯỚC 7: SO SÁNH CÁC MÔ HÌNH")
print("="*80)

# Tạo bảng so sánh
comparison_df = pd.DataFrame({
    'Model': results.keys(),
    'Accuracy': [results[m]['accuracy'] for m in results.keys()],
    'Precision': [results[m]['precision'] for m in results.keys()],
    'Recall': [results[m]['recall'] for m in results.keys()],
    'F1 Score': [results[m]['f1'] for m in results.keys()],
    'ROC-AUC': [results[m]['auc'] for m in results.keys()]
})

print("\n📊 Bảng so sánh:")
print(comparison_df.to_string(index=False))

# Tìm mô hình tốt nhất
best_model_name = comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Model']
best_accuracy = comparison_df['Accuracy'].max()

print(f"\n🏆 Mô hình tốt nhất: {best_model_name}")
print(f"   Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")

# ============================================================================
# BƯỚC 8: VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print("BƯỚC 8: VISUALIZATION KẾT QUẢ")
print("="*80)

# Tạo thư mục để lưu hình
output_dir = project_root / 'notebooks'
output_dir.mkdir(exist_ok=True)

# 8.1 Confusion Matrix Heatmap
print(f"\n[8.1] Saving Confusion Matrix visualization...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (model_name, result) in enumerate(results.items()):
    cm = confusion_matrix(y_test, result['y_pred'])
    
    ax = axes[idx]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Stayed', 'Churned'],
                yticklabels=['Stayed', 'Churned'],
                cbar=False)
    ax.set_title(f'{model_name}\nAccuracy: {result["accuracy"]:.3f}', 
                 fontweight='bold')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig(output_dir / 'confusion_matrices.png', dpi=200, bbox_inches='tight')
print(f"  ✓ Saved: confusion_matrices.png")
plt.close()

# 8.2 Metrics Comparison Bar Plot
print(f"\n[8.2] Saving Metrics Comparison...")
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
colors = ['#3498db', '#e74c3c', '#2ecc71']

for idx, metric in enumerate(metrics):
    ax = axes[idx]
    values = comparison_df[metric].values
    models_list = comparison_df['Model'].values
    
    bars = ax.bar(models_list, values, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_title(f'{metric}', fontweight='bold', fontsize=11)
    ax.set_ylabel('Score')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    # Thêm giá trị trên cột
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Rotate x labels
    ax.tick_params(axis='x', rotation=45)

# Ẩn subplot cuối (không dùng)
axes[4].axis('off')

plt.tight_layout()
plt.savefig(output_dir / 'metrics_comparison.png', dpi=200, bbox_inches='tight')
print(f"  ✓ Saved: metrics_comparison.png")
plt.close()

# 8.3 ROC Curves
print(f"\n[8.3] Saving ROC Curves...")
fig, ax = plt.subplots(figsize=(10, 7))

colors_roc = ['#3498db', '#e74c3c', '#2ecc71']
for idx, (model_name, result) in enumerate(results.items()):
    fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
    auc_score = result['auc']
    ax.plot(fpr, tpr, linewidth=2.5, label=f'{model_name} (AUC={auc_score:.3f})',
            color=colors_roc[idx])

# Plot random classifier
ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier (AUC=0.5)')

ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title('ROC Curves - Model Comparison', fontweight='bold', fontsize=12)
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'roc_curves.png', dpi=200, bbox_inches='tight')
print(f"  ✓ Saved: roc_curves.png")
plt.close()

# 8.4 Feature Importance (cho Tree-based models)
print(f"\n[8.4] Saving Feature Importance...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Decision Tree feature importance
dt_importance = dt_model.feature_importances_
dt_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': dt_importance
}).sort_values('Importance', ascending=True)

axes[0].barh(dt_importance_df['Feature'], dt_importance_df['Importance'], 
             color='#e74c3c', edgecolor='black')
axes[0].set_title('Decision Tree - Feature Importance', fontweight='bold', fontsize=11)
axes[0].set_xlabel('Importance')
axes[0].grid(axis='x', alpha=0.3)

# Random Forest feature importance
rf_importance = rf_model.feature_importances_
rf_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_importance
}).sort_values('Importance', ascending=True)

axes[1].barh(rf_importance_df['Feature'], rf_importance_df['Importance'],
             color='#2ecc71', edgecolor='black')
axes[1].set_title('Random Forest - Feature Importance', fontweight='bold', fontsize=11)
axes[1].set_xlabel('Importance')
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'feature_importance.png', dpi=200, bbox_inches='tight')
print(f"  ✓ Saved: feature_importance.png")
plt.close()

# 8.5 Learning Curve (cho Random Forest)
print(f"\n[8.5] Saving Learning Curve...")
train_sizes, train_scores, val_scores = learning_curve(
    rf_model,
    X_train_processed, y_train,
    cv=5,
    scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(train_sizes, train_mean, label='Training Accuracy', color='#3498db', linewidth=2.5)
ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                alpha=0.2, color='#3498db')

ax.plot(train_sizes, val_mean, label='Validation Accuracy', color='#e74c3c', linewidth=2.5)
ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                alpha=0.2, color='#e74c3c')

ax.set_xlabel('Training Set Size', fontsize=11)
ax.set_ylabel('Accuracy', fontsize=11)
ax.set_title('Random Forest - Learning Curve', fontweight='bold', fontsize=12)
ax.legend(loc='best', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'learning_curve.png', dpi=200, bbox_inches='tight')
print(f"  ✓ Saved: learning_curve.png")
plt.close()

# ============================================================================
# BƯỚC 9: TÓM TẮT KẾT QUẢ
# ============================================================================

print("\n" + "="*80)
print("BƯỚC 9: TÓM TẮT KẾT QUẢ & KHUYẾN NGHỊ")
print("="*80)

print(f"""
📋 SUMMARY:
  ✓ Dataset size: {len(df)} samples
  ✓ Features: {X.shape[1]} (7 numerical + 3 categorical)
  ✓ Train/Test split: {len(X_train)}/{len(X_test)}
  ✓ Churn rate: {(y.sum()/len(y))*100:.1f}%
  
🤖 MODELS TRAINED:
  1. Logistic Regression
  2. Decision Tree Classifier
  3. Random Forest Classifier

🏆 BEST MODEL: {best_model_name}
  - Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)
  - Precision: {results[best_model_name]['precision']:.4f}
  - Recall: {results[best_model_name]['recall']:.4f}
  - F1 Score: {results[best_model_name]['f1']:.4f}

💡 RECOMMENDATIONS:
  1. Logistic Regression là baseline đơn giản, phù hợp học ban đầu
  2. Decision Tree dễ visualize và giải thích
  3. Random Forest cho kết quả tốt nhất, recommend sử dụng
  
📊 VISUALIZATIONS SAVED:
  ✓ confusion_matrices.png - So sánh confusion matrix
  ✓ metrics_comparison.png - So sánh các metrics
  ✓ roc_curves.png - ROC curves cho cả 3 mô hình
  ✓ feature_importance.png - Tầm quan trọng của features
  ✓ learning_curve.png - Learning curve của Random Forest
  
📁 Tất cả hình đã lưu tại: {output_dir}
""")

print("="*80)
print("✅ TRAINING & EVALUATION COMPLETE!")
print("="*80 + "\n")
