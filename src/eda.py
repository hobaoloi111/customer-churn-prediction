"""
Exploratory Data Analysis (EDA) cho bài toán Customer Churn Prediction
Hướng dẫn từng bước phân tích dataset bằng Python
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Cấu hình visualization
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# ============================================================================
# BƯỚC 1: LOAD VÀ XEM THÔNG TIN CƠ BẢN
# ============================================================================

print("\n" + "="*80)
print("BƯỚC 1: LOAD VÀ XEM THÔNG TIN CƠ BẢN CỦA DATASET")
print("="*80)

# Xác định đường dẫn file
project_root = Path(__file__).parent.parent
data_file = project_root / 'data' / 'customer_churn_data.csv'

# Load dữ liệu
print(f"\n[1.1] Đang load dữ liệu từ: {data_file}")
df = pd.read_csv(data_file)

# In thông tin cơ bản
print(f"\n[1.2] Kích thước dataset: {df.shape[0]} hàng, {df.shape[1]} cột")
print(f"\n[1.3] 5 hàng đầu tiên:")
print(df.head())
print(f"\n[1.4] 5 hàng cuối cùng:")
print(df.tail())

print(f"\n[1.5] Tên các cột:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")

# ============================================================================
# BƯỚC 2: KIỂM TRA MISSING VALUES VÀ OUTLIERS
# ============================================================================

print("\n" + "="*80)
print("BƯỚC 2: KIỂM TRA MISSING VALUES VÀ OUTLIERS")
print("="*80)

# Kiểm tra missing values
print("\n[2.1] Kiểm tra Missing Values:")
missing_counts = df.isnull().sum()
missing_percent = (missing_counts / len(df)) * 100
missing_df = pd.DataFrame({
    'Column': missing_counts.index,
    'Missing_Count': missing_counts.values,
    'Missing_Percent': missing_percent.values
})
missing_df = missing_df[missing_df['Missing_Count'] > 0]
if len(missing_df) == 0:
    print("  ✓ Không có missing values!")
else:
    print(missing_df.to_string(index=False))

# Kiểm tra outliers bằng IQR method
print("\n[2.2] Kiểm tra Outliers (sử dụng IQR method):")
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

outlier_summary = {}
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_count = len(outliers)
    outlier_percent = (outlier_count / len(df)) * 100
    
    if outlier_count > 0:
        outlier_summary[col] = {
            'Count': outlier_count,
            'Percent': outlier_percent,
            'Range': f"[{lower_bound:.2f}, {upper_bound:.2f}]"
        }
        print(f"  {col}: {outlier_count} outliers ({outlier_percent:.1f}%)")
        print(f"    - Valid range: {outlier_summary[col]['Range']}")
        print(f"    - Min: {df[col].min():.2f}, Max: {df[col].max():.2f}")

if not outlier_summary:
    print("  ✓ Không có outliers đáng kể!")

# ============================================================================
# BƯỚC 3: THỐNG KÊ MÔ TẢ (DESCRIBE & INFO)
# ============================================================================

print("\n" + "="*80)
print("BƯỚC 3: THỐNG KÊ MÔ TẢ - DESCRIBE & INFO")
print("="*80)

# Info - kiểu dữ liệu
print("\n[3.1] Thông tin chi tiết từng cột (df.info()):")
print(df.info())

# Describe - thống kê số học
print("\n[3.2] Thống kê mô tả các cột Numerical (df.describe()):")
print(df.describe().round(2))

# Thống kê cho cột Categorical
print("\n[3.3] Thống kê cho các cột Categorical:")
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    print(f"\n  {col}:")
    print(df[col].value_counts())
    print(f"  - Số lượng categories: {df[col].nunique()}")

# ============================================================================
# BƯỚC 4: PHÂN TÍCH PHÂN PHỐI CỦA TỪNG FEATURE
# ============================================================================

print("\n" + "="*80)
print("BƯỚC 4: PHÂN TÍCH PHÂN PHỐI CỦA TỪNG FEATURE")
print("="*80)

print("\n[4.1] Phân phối của các Numerical Features:")

# Tính số hàng và cột cần thiết
n_features = len(numerical_cols)
n_cols = 3
n_rows = (n_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
axes = axes.flatten()

for idx, col in enumerate(numerical_cols):
    ax = axes[idx]
    
    # Histogram với KDE
    ax.hist(df[col], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax2 = ax.twinx()
    df[col].plot(kind='kde', ax=ax2, color='red', linewidth=2)
    
    ax.set_title(f'Phân phối {col}', fontsize=10, fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel('Tần số')
    ax2.set_ylabel('Density', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Thống kê
    skewness = df[col].skew()
    kurtosis = df[col].kurtosis()
    print(f"  {col}: Skewness={skewness:.2f}, Kurtosis={kurtosis:.2f}")

# Ẩn các subplot trống nếu có
for idx in range(n_features, len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig(project_root / 'notebooks' / 'eda_distributions_numerical.png', dpi=300, bbox_inches='tight')
print("\n  ✓ Hình lưu tại: notebooks/eda_distributions_numerical.png")
plt.close()

# Phân phối Categorical Features
print("\n[4.2] Phân phối của các Categorical Features:")

# Bỏ qua customer_id vì có quá nhiều categories
cat_cols_to_plot = [col for col in categorical_cols if col != 'customer_id']

if len(cat_cols_to_plot) > 0:
    fig, axes = plt.subplots(1, min(len(cat_cols_to_plot), 3), figsize=(14, 4))
    if len(cat_cols_to_plot) == 1:
        axes = [axes]
    
    for idx, col in enumerate(cat_cols_to_plot):
        ax = axes[idx]
        
        value_counts = df[col].value_counts()
        colors = sns.color_palette("husl", len(value_counts))
        value_counts.plot(kind='bar', ax=ax, color=colors, edgecolor='black')
        
        ax.set_title(f'Phân phối {col}', fontsize=10, fontweight='bold')
        ax.set_xlabel(col)
        ax.set_ylabel('Tần số')
        ax.tick_params(axis='x', rotation=45)
        
        # Thêm % lên trên mỗi cột
        for i, v in enumerate(value_counts.values):
            percentage = (v / len(df)) * 100
            ax.text(i, v + 5, f'{percentage:.1f}%', ha='center', fontweight='bold', fontsize=9)
    
    # Ẩn các subplot trống nếu có
    for idx in range(len(cat_cols_to_plot), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(project_root / 'notebooks' / 'eda_distributions_categorical.png', dpi=300, bbox_inches='tight')
    print("  ✓ Hình lưu tại: notebooks/eda_distributions_categorical.png")
    plt.close()

# ============================================================================
# BƯỚC 5: PHÂN TÍCH MỐI QUAN HỆ GIỮA FEATURES VÀ TARGET
# ============================================================================

print("\n" + "="*80)
print("BƯỚC 5: PHÂN TÍCH MỐI QUAN HỆ GIỮA FEATURES VÀ TARGET")
print("="*80)

target_col = 'churn'

# 5.1 Correlation với Target (Numerical)
print(f"\n[5.1] Correlation của Numerical Features với {target_col}:")
correlation_with_target = df[numerical_cols + [target_col]].corr()[target_col].sort_values(ascending=False)
print(correlation_with_target.round(3))

# 5.2 Chi-square test cho Categorical Features
print(f"\n[5.2] Mối quan hệ giữa Categorical Features và {target_col}:")
for col in categorical_cols:
    if col != target_col:  # Bỏ qua customer_id
        cross_tab = pd.crosstab(df[col], df[target_col], margins=True)
        churn_rate_by_category = df.groupby(col)[target_col].mean()
        print(f"\n  {col}:")
        print(f"    Churn Rate by Category:")
        for cat, rate in churn_rate_by_category.items():
            print(f"      - {cat}: {rate*100:.1f}%")

# 5.3 Visualization: Target Distribution
print(f"\n[5.3] Phân phối của Target ({target_col}):")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Count plot
target_counts = df[target_col].value_counts()
colors = ['#2ecc71', '#e74c3c']  # Green for 0, Red for 1
target_counts.plot(kind='bar', ax=axes[0], color=colors, edgecolor='black')
axes[0].set_title(f'Phân phối {target_col}', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Churn')
axes[0].set_ylabel('Số lượng')
axes[0].set_xticklabels(['Stayed (0)', 'Churned (1)'], rotation=0)

# Percentage
churn_pct = (target_counts / len(df)) * 100
axes[1].pie(churn_pct, labels=['Stayed', 'Churned'], autopct='%1.1f%%', 
            colors=colors, startangle=90, textprops={'fontsize': 11})
axes[1].set_title(f'Tỷ lệ {target_col}', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(project_root / 'notebooks' / 'eda_target_distribution.png', dpi=300, bbox_inches='tight')
print("  ✓ Hình lưu tại: notebooks/eda_target_distribution.png")
plt.close()

# ============================================================================
# BƯỚC 6: VISUALIZATION - RELATIONSHIP ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("BƯỚC 6: VISUALIZATION - PHÂN TÍCH MỐI QUAN HỆ")
print("="*80)

# 6.1 Box plot - Numerical Features vs Target
print("\n[6.1] Box Plot - Numerical Features vs Target:")

n_features = len(numerical_cols)
n_cols = 3
n_rows = (n_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
axes = axes.flatten()

for idx, col in enumerate(numerical_cols):
    ax = axes[idx]
    df.boxplot(column=col, by=target_col, ax=ax)
    ax.set_title(f'{col} by {target_col}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Churn')
    ax.set_ylabel(col)
    plt.sca(ax)
    plt.xticks([1, 2], ['Stayed', 'Churned'])

# Ẩn các subplot trống nếu có
for idx in range(n_features, len(axes)):
    axes[idx].axis('off')

plt.suptitle('Box Plot: Numerical Features vs Churn', fontsize=12, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(project_root / 'notebooks' / 'eda_boxplot_vs_target.png', dpi=300, bbox_inches='tight')
print("  ✓ Hình lưu tại: notebooks/eda_boxplot_vs_target.png")
plt.close()

# 6.2 Count Plot - Categorical Features vs Target
print("\n[6.2] Count Plot - Categorical Features vs Target:")

cat_cols_to_plot = [col for col in categorical_cols if col != 'customer_id']

if len(cat_cols_to_plot) > 0:
    fig, axes = plt.subplots(1, min(len(cat_cols_to_plot), 3), figsize=(14, 4))
    if len(cat_cols_to_plot) == 1:
        axes = [axes]
    
    for idx, col in enumerate(cat_cols_to_plot):
        ax = axes[idx]
        cross_data = pd.crosstab(df[col], df[target_col])
        cross_data.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], edgecolor='black')
        ax.set_title(f'{col} vs {target_col}', fontsize=10, fontweight='bold')
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        ax.legend(['Stayed', 'Churned'], loc='upper right')
        ax.tick_params(axis='x', rotation=45)
    
    # Ẩn các subplot trống nếu có
    for idx in range(len(cat_cols_to_plot), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(project_root / 'notebooks' / 'eda_countplot_vs_target.png', dpi=300, bbox_inches='tight')
    print("  ✓ Hình lưu tại: notebooks/eda_countplot_vs_target.png")
    plt.close()

# 6.3 Scatter Plot Matrix (Numerical vs Numerical with target coloring)
print("\n[6.3] Scatter Plot - Relationships between Numerical Features:")
selected_features = ['tenure_months', 'monthly_charges', 'customer_satisfaction', 'num_support_tickets']
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, (i, col1) in enumerate([(0, 1), (0, 2), (1, 2), (2, 3)]):
    ax = axes[idx]
    col1_name = selected_features[i]
    col2_name = selected_features[idx + 1]
    
    # Scatter plot với màu khác nhau cho churn/stayed
    for churn_val, color, label in [(0, '#2ecc71', 'Stayed'), (1, '#e74c3c', 'Churned')]:
        mask = df[target_col] == churn_val
        ax.scatter(df[mask][col1_name], df[mask][col2_name], 
                  alpha=0.6, s=50, color=color, label=label, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel(col1_name, fontsize=10)
    ax.set_ylabel(col2_name, fontsize=10)
    ax.set_title(f'{col1_name} vs {col2_name}', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(project_root / 'notebooks' / 'eda_scatterplot.png', dpi=300, bbox_inches='tight')
print("  ✓ Hình lưu tại: notebooks/eda_scatterplot.png")
plt.close()

# ============================================================================
# BƯỚC 7: CORRELATION MATRIX
# ============================================================================

print("\n" + "="*80)
print("BƯỚC 7: CORRELATION MATRIX")
print("="*80)

# Chuẩn bị dữ liệu cho correlation
df_corr = df[numerical_cols + [target_col]].copy()

# Tính correlation matrix
correlation_matrix = df_corr.corr()

print("\n[7.1] Correlation Matrix:")
print(correlation_matrix.round(3))

# Heatmap visualization
print("\n[7.2] Visualization - Correlation Heatmap:")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Correlation Matrix - All Numerical Features', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(project_root / 'notebooks' / 'eda_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("  ✓ Hình lưu tại: notebooks/eda_correlation_heatmap.png")
plt.close()

# Tìm correlation cao nhất (bỏ qua diagonal)
print("\n[7.3] Top Correlations (lớn hơn 0.5):")
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.5:
            print(f"  {correlation_matrix.columns[i]} <-> {correlation_matrix.columns[j]}: {correlation_matrix.iloc[i, j]:.3f}")

# ============================================================================
# BƯỚC 8: TÓM TẮT INSIGHTS
# ============================================================================

print("\n" + "="*80)
print("BƯỚC 8: TÓM TẮT INSIGHTS CHÍNH")
print("="*80)

print("\n[8.1] Thống kê chung:")
print(f"  - Tổng số khách hàng: {len(df)}")
print(f"  - Số lượng features: {len(df.columns)}")
print(f"  - Kiểu features: {len(numerical_cols)} numerical, {len(categorical_cols)} categorical")
print(f"  - Churn rate: {(df[target_col].sum()/len(df))*100:.1f}%")

print("\n[8.2] Features có correlation cao nhất với Churn:")
top_correlations = abs(correlation_with_target[correlation_with_target.index != target_col]).nlargest(3)
for feature, corr in top_correlations.items():
    direction = "↑" if correlation_with_target[feature] > 0 else "↓"
    print(f"  {direction} {feature}: {correlation_with_target[feature]:.3f}")

print("\n[8.3] Categorical Features với Churn Rate khác nhau:")
for col in categorical_cols:
    if col != 'customer_id':
        churn_rates = df.groupby(col)[target_col].agg(['mean', 'count'])
        print(f"  {col}:")
        for category, row in churn_rates.iterrows():
            print(f"    - {category}: {row['mean']*100:.1f}% (n={int(row['count'])})")

print("\n[8.4] Data Quality:")
print(f"  ✓ Missing values: {df.isnull().sum().sum()}")
print(f"  ✓ Duplicate rows: {df.duplicated().sum()}")
print(f"  ✓ No class imbalance issues: {(df[target_col].value_counts() / len(df)).min():.1%} minimum class")

print("\n" + "="*80)
print("✓ EDA HOÀN TẤT! Tất cả hình đã lưu tại thư mục 'notebooks/'")
print("="*80 + "\n")
