"""
Exploratory Data Analysis (EDA) - Phiên bản Đơn Giản
Chỉ tập trung vào phân tích và thống kê, không dùng plt.show()
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Cấu hình matplotlib để lưu file thay vì hiển thị
plt.switch_backend('Agg')
sns.set_style("whitegrid")

# ============================================================================
# BƯỚC 1: LOAD VÀ XEM THÔNG TIN CƠ BẢN
# ============================================================================

print("\n" + "="*80)
print("BƯỚC 1: LOAD VÀ XEM THÔNG TIN CƠ BẢN CỦA DATASET")
print("="*80)

project_root = Path(__file__).parent.parent
data_file = project_root / 'data' / 'customer_churn_data.csv'

print(f"\n[1.1] Đang load dữ liệu từ: {data_file}")
df = pd.read_csv(data_file)

print(f"\n[1.2] Kích thước dataset: {df.shape[0]} hàng, {df.shape[1]} cột")
print(f"\n[1.3] 5 hàng đầu tiên:")
print(df.head())

print(f"\n[1.5] Tên các cột:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")

# ============================================================================
# BƯỚC 2: KIỂM TRA MISSING VALUES VÀ OUTLIERS
# ============================================================================

print("\n" + "="*80)
print("BƯỚC 2: KIỂM TRA MISSING VALUES VÀ OUTLIERS")
print("="*80)

# Missing values
print("\n[2.1] Kiểm tra Missing Values:")
missing_counts = df.isnull().sum()
if missing_counts.sum() == 0:
    print("  ✓ Không có missing values!")
else:
    for col, count in missing_counts[missing_counts > 0].items():
        percent = (count / len(df)) * 100
        print(f"  {col}: {count} ({percent:.1f}%)")

# Outliers
print("\n[2.2] Kiểm tra Outliers (sử dụng IQR method):")
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

has_outliers = False
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    if len(outliers) > 0:
        has_outliers = True
        percent = (len(outliers) / len(df)) * 100
        print(f"  {col}: {len(outliers)} outliers ({percent:.1f}%)")

if not has_outliers:
    print("  ✓ Không có outliers đáng kể!")

# ============================================================================
# BƯỚC 3: THỐNG KÊ MÔ TẢ
# ============================================================================

print("\n" + "="*80)
print("BƯỚC 3: THỐNG KÊ MÔ TẢ - DESCRIBE & INFO")
print("="*80)

print("\n[3.1] Thông tin chi tiết từng cột:")
print(df.info())

print("\n[3.2] Thống kê mô tả các cột Numerical:")
print(df.describe().round(2))

print("\n[3.3] Thống kê cho các cột Categorical:")
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    print(f"\n  {col}:")
    print(f"    {df[col].value_counts().to_string()}")

# ============================================================================
# BƯỚC 4: PHÂN TÍCH PHÂN PHỐI
# ============================================================================

print("\n" + "="*80)
print("BƯỚC 4: PHÂN TÍCH PHÂN PHỐI CỦA CÁC FEATURES")
print("="*80)

print("\n[4.1] Phân phối Numerical Features (Skewness & Kurtosis):")
for col in numerical_cols:
    skewness = df[col].skew()
    kurtosis = df[col].kurtosis()
    print(f"  {col}:")
    print(f"    - Skewness: {skewness:>7.2f} {'(Bình thường)' if abs(skewness) < 0.5 else '(Lệch)'}")
    print(f"    - Kurtosis: {kurtosis:>7.2f}")

# ============================================================================
# BƯỚC 5: PHÂN TÍCH MỐI QUAN HỆ GIỮA FEATURES VÀ TARGET
# ============================================================================

print("\n" + "="*80)
print("BƯỚC 5: PHÂN TÍCH MỐI QUAN HỆ GIỮA FEATURES VÀ TARGET")
print("="*80)

target_col = 'churn'

# Correlation
print(f"\n[5.1] Correlation của Numerical Features với {target_col}:")
corr_series = df[numerical_cols + [target_col]].corr()[target_col]
correlation_with_target = corr_series.iloc[corr_series.abs().argsort()[::-1]]
print(correlation_with_target.round(3))

# Target Distribution
print(f"\n[5.2] Phân phối của Target ({target_col}):")
target_counts = df[target_col].value_counts()
for val in sorted(target_counts.index):
    count = target_counts[val]
    percent = (count / len(df)) * 100
    label = "Stayed" if val == 0 else "Churned"
    print(f"  {label} ({val}): {count:>3} ({percent:>5.1f}%)")

# Categorical Features vs Target
print(f"\n[5.3] Churn Rate by Categorical Features:")
for col in categorical_cols:
    if col != 'customer_id':
        print(f"\n  {col}:")
        churn_by_cat = df.groupby(col)[target_col].agg(['sum', 'count', 'mean'])
        churn_by_cat.columns = ['Churned', 'Total', 'ChurnRate']
        for category, row in churn_by_cat.iterrows():
            print(f"    - {category}: {row['ChurnRate']*100:>5.1f}% (n={int(row['Total'])})")

# ============================================================================
# BƯỚC 6: CORRELATION MATRIX
# ============================================================================

print("\n" + "="*80)
print("BƯỚC 6: CORRELATION MATRIX")
print("="*80)

df_corr = df[numerical_cols + [target_col]].copy()
correlation_matrix = df_corr.corr()

print("\n[6.1] Correlation Matrix:")
print(correlation_matrix.round(3))

print("\n[6.2] Top Correlations (lớn hơn 0.5):")
found_corr = False
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.5:
            found_corr = True
            col1 = correlation_matrix.columns[i]
            col2 = correlation_matrix.columns[j]
            corr_val = correlation_matrix.iloc[i, j]
            print(f"  {col1} ↔ {col2}: {corr_val:.3f}")

if not found_corr:
    print("  Không có correlations lớn hơn 0.5")

# ============================================================================
# BƯỚC 7: VISUALIZATION - SAVE FIGURES
# ============================================================================

print("\n" + "="*80)
print("BƯỚC 7: VISUALIZATION - SAVING FIGURES")
print("="*80)

# Tạo thư mục nếu chưa tồn tại
notebooks_dir = project_root / 'notebooks'
notebooks_dir.mkdir(exist_ok=True)

# 7.1 Distribution plots
print("\n[7.1] Saving distribution plots...")
n_features = len(numerical_cols)
n_cols = 3
n_rows = (n_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
axes = axes.flatten()

for idx, col in enumerate(numerical_cols):
    ax = axes[idx]
    ax.hist(df[col], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_title(f'Distribution of {col}', fontsize=10, fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')

for idx in range(n_features, len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig(notebooks_dir / 'eda_distributions.png', dpi=200, bbox_inches='tight')
print("  ✓ Saved: eda_distributions.png")
plt.close()

# 7.2 Target distribution
print("\n[7.2] Saving target distribution...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

target_counts.plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'])
axes[0].set_title('Churn Distribution', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Churn')
axes[0].set_ylabel('Count')
axes[0].set_xticklabels(['Stayed', 'Churned'], rotation=0)

churn_pct = (target_counts / len(df)) * 100
axes[1].pie(churn_pct, labels=['Stayed', 'Churned'], autopct='%1.1f%%',
            colors=['#2ecc71', '#e74c3c'], startangle=90)
axes[1].set_title('Churn Percentage', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(notebooks_dir / 'eda_target.png', dpi=200, bbox_inches='tight')
print("  ✓ Saved: eda_target.png")
plt.close()

# 7.3 Box plots
print("\n[7.3] Saving box plots...")
n_features = len(numerical_cols)
n_cols = 3
n_rows = (n_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
axes = axes.flatten()

for idx, col in enumerate(numerical_cols):
    ax = axes[idx]
    df.boxplot(column=col, by=target_col, ax=ax)
    ax.set_title(f'{col} by Churn', fontsize=10, fontweight='bold')
    ax.set_xlabel('Churn')
    ax.set_ylabel(col)
    ax.set_xticklabels(['Stayed', 'Churned'])

for idx in range(n_features, len(axes)):
    axes[idx].axis('off')

plt.suptitle('')
plt.tight_layout()
plt.savefig(notebooks_dir / 'eda_boxplot.png', dpi=200, bbox_inches='tight')
print("  ✓ Saved: eda_boxplot.png")
plt.close()

# 7.4 Correlation heatmap
print("\n[7.4] Saving correlation heatmap...")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Correlation Matrix', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(notebooks_dir / 'eda_correlation.png', dpi=200, bbox_inches='tight')
print("  ✓ Saved: eda_correlation.png")
plt.close()

# 7.5 Categorical features
print("\n[7.5] Saving categorical plots...")
cat_cols_plot = [col for col in categorical_cols if col != 'customer_id']

if len(cat_cols_plot) > 0:
    fig, axes = plt.subplots(1, len(cat_cols_plot), figsize=(5*len(cat_cols_plot), 4))
    if len(cat_cols_plot) == 1:
        axes = [axes]

    for idx, col in enumerate(cat_cols_plot):
        ax = axes[idx]
        value_counts = df[col].value_counts()
        value_counts.plot(kind='bar', ax=ax, color=sns.color_palette("husl", len(value_counts)))
        ax.set_title(f'Distribution of {col}', fontsize=10, fontweight='bold')
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(notebooks_dir / 'eda_categorical.png', dpi=200, bbox_inches='tight')
    print("  ✓ Saved: eda_categorical.png")
    plt.close()

# ============================================================================
# BƯỚC 8: TÓM TẮT INSIGHTS
# ============================================================================

print("\n" + "="*80)
print("BƯỚC 8: TÓM TẮT INSIGHTS CHÍNH")
print("="*80)

print("\n[8.1] Thống kê chung:")
print(f"  - Tổng số khách hàng: {len(df)}")
print(f"  - Số lượng features: {len(df.columns) - 1}")
print(f"  - Numerical features: {len(numerical_cols)}")
print(f"  - Categorical features: {len(categorical_cols) - 1}")  # Bỏ customer_id
print(f"  - Churn rate: {(df[target_col].sum()/len(df))*100:.1f}%")

print("\n[8.2] Top 3 features có correlation cao nhất với Churn:")
top_corr = abs(correlation_with_target[correlation_with_target.index != target_col]).nlargest(3)
for i, (feature, corr) in enumerate(top_corr.items(), 1):
    direction = "↑" if correlation_with_target[feature] > 0 else "↓"
    print(f"  {i}. {direction} {feature}: {correlation_with_target[feature]:.3f}")

print("\n[8.3] Data Quality:")
print(f"  ✓ Missing values: {df.isnull().sum().sum()}")
print(f"  ✓ Duplicate rows: {df.duplicated().sum()}")
min_class_pct = (df[target_col].value_counts().min() / len(df)) * 100
print(f"  ✓ Minimum class percentage: {min_class_pct:.1f}%")

print("\n" + "="*80)
print("✓ EDA HOÀN TẤT! Tất cả hình đã lưu tại thư mục 'notebooks/'")
print("="*80 + "\n")
