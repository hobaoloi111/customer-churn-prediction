"""
Script để sinh dữ liệu giả lập cho bài toán Customer Churn Prediction
Dữ liệu phản ánh kịch bản thực tế với correlations giữa các features
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Thiết lập random seed để có thể reproduce dữ liệu
np.random.seed(42)

# ============================================================================
# BƯỚC 1: ĐỊNH NGHĨA PARAMETERS
# ============================================================================
n_samples = 450  # Số lượng khách hàng (300-500)
n_customers = n_samples

print(f"[INFO] Sinh dữ liệu cho {n_samples} khách hàng...")

# ============================================================================
# BƯỚC 2: TẠO FEATURES CƠ BẢN
# ============================================================================

# 1. Customer ID
customer_ids = [f"CUST_{i:06d}" for i in range(1, n_customers + 1)]

# 2. Tuổi (Age): 18-80, phân phối chuẩn
ages = np.random.normal(loc=45, scale=15, size=n_samples)
ages = np.clip(ages, 18, 80).astype(int)

# 3. Thời gian là khách hàng (Tenure months): 0-120
# - Tuổi cao → tenure có xu hướng cao hơn
# - Tuổi thấp → tenure thường thấp
tenure_months = np.maximum(
    np.random.poisson(lam=40, size=n_samples) + (ages - 30) * 0.3,
    0
)
tenure_months = np.minimum(tenure_months, 120).astype(int)

# 4. Loại hợp đồng (Contract type): Month-to-Month, One Year, Two Year
# - Khách lâu năm → xu hướng hợp đồng dài hạn
contract_probs = np.random.random(n_samples)
contract_type = np.where(
    tenure_months < 20,
    'Month-to-Month',
    np.where(contract_probs < 0.4, 'Month-to-Month', 
             np.where(contract_probs < 0.7, 'One Year', 'Two Year'))
)

# 5. Phí hàng tháng (Monthly charges): $10-150
# - Hợp đồng dài hạn → phí thường cao hơn (tính năng cao)
# - Khách trẻ → xu hướng phí cao hơn
base_charges = np.random.uniform(10, 150, n_samples)
contract_multiplier = np.where(
    contract_type == 'Month-to-Month', 1.0,
    np.where(contract_type == 'One Year', 1.1, 1.2)
)
monthly_charges = (base_charges * contract_multiplier).round(2)
monthly_charges = np.clip(monthly_charges, 10, 150)

# 6. Tổng tiền chi trả (Total charges): Tính dựa trên tenure + monthly charges
# - Có correlation cao với tenure_months và monthly_charges
total_charges = (monthly_charges * tenure_months * np.random.uniform(0.95, 1.05, n_samples)).round(2)

# 7. Dịch vụ Internet (Internet service): DSL, Fiber optic, No
internet_service = np.random.choice(
    ['DSL', 'Fiber optic', 'No'], 
    size=n_samples,
    p=[0.45, 0.35, 0.20]
)

# 8. Phương thức thanh toán (Payment method)
payment_methods = ['Bank transfer', 'Credit card', 'Check', 'E-check']
# E-check thường correlate với churn cao hơn
payment_method = np.random.choice(payment_methods, size=n_samples, p=[0.3, 0.3, 0.25, 0.15])

# 9. Số lượng ticket hỗ trợ (Support tickets)
# - Khách hài lòng → ít ticket
# - Khách không hài lòng → nhiều ticket
# - Phí cao → có thể nhiều vấn đề → nhiều ticket
support_tickets = np.random.poisson(lam=1.5, size=n_samples)
# Tăng tickets nếu dịch vụ là Fiber optic (chất lượng vấn đề)
support_tickets = np.where(
    internet_service == 'Fiber optic',
    support_tickets + np.random.poisson(lam=0.8, size=n_samples),
    support_tickets
)
support_tickets = np.clip(support_tickets, 0, 10).astype(int)

# 10. Mức độ hài lòng (Customer satisfaction): 1-5
# - Tenure cao → hài lòng cao hơn
# - Support tickets nhiều → hài lòng thấp
# - Contract long-term → hài lòng cao hơn (cam kết)
satisfaction_base = 3 + (tenure_months / 120) * 1.5 - (support_tickets / 10) * 1.5
satisfaction = np.clip(satisfaction_base, 1, 5).round(0).astype(int)
# Thêm một chút randomness
satisfaction += np.random.choice([-1, 0, 1], size=n_samples, p=[0.2, 0.6, 0.2])
satisfaction = np.clip(satisfaction, 1, 5).astype(int)

# ============================================================================
# BƯỚC 3: TẠO TARGET - CHURN (Dự đoán khách hàng rời đi)
# ============================================================================
"""
Logic tính xác suất churn:
- Tenure thấp → churn cao (khách mới dễ rời)
- Satisfaction thấp → churn cao
- Support tickets nhiều → churn cao (nhiều vấn đề)
- Contract Month-to-Month → churn cao (dễ hủy)
- Payment method E-check → churn cao (kém tin cậy)
"""

churn_probability = np.zeros(n_samples)

# 1. Ảnh hưởng của tenure
churn_probability += (1 - tenure_months / 120) * 0.4

# 2. Ảnh hưởng của satisfaction
churn_probability += (5 - satisfaction) / 5 * 0.3

# 3. Ảnh hưởng của support tickets
churn_probability += np.minimum(support_tickets / 10, 1.0) * 0.15

# 4. Ảnh hưởng của contract type
churn_probability += np.where(contract_type == 'Month-to-Month', 0.15, 
                              np.where(contract_type == 'One Year', 0.05, 0.02))

# 5. Ảnh hưởng của payment method
churn_probability += np.where(payment_method == 'E-check', 0.08, 0.02)

# Chuẩn hóa xác suất trong [0, 1]
churn_probability = np.clip(churn_probability, 0, 1)

# Sinh dữ liệu churn dựa trên xác suất
churn = (np.random.random(n_samples) < churn_probability).astype(int)

print(f"[INFO] Tỷ lệ churn: {churn.mean()*100:.1f}%")

# ============================================================================
# BƯỚC 4: TẠO DATAFRAME
# ============================================================================

df = pd.DataFrame({
    'customer_id': customer_ids,
    'age': ages,
    'tenure_months': tenure_months,
    'monthly_charges': monthly_charges,
    'total_charges': total_charges,
    'contract_type': contract_type,
    'internet_service': internet_service,
    'payment_method': payment_method,
    'num_support_tickets': support_tickets,
    'customer_satisfaction': satisfaction,
    'churn': churn
})

# ============================================================================
# BƯỚC 5: THỐNG KÊ DỮ LIỆU
# ============================================================================

print("\n" + "="*70)
print("THỐNG KÊ DỮ LIỆU ĐÃ SINH")
print("="*70)
print(f"\nHình dạng dataset: {df.shape}")
print(f"\nThông tin cột:")
print(df.info())
print(f"\nThống kê mô tả (Numerical Features):")
print(df.describe().round(2))
print(f"\nThống kê theo Category:")
print(f"  - Contract Type:\n{df['contract_type'].value_counts()}")
print(f"  - Internet Service:\n{df['internet_service'].value_counts()}")
print(f"  - Payment Method:\n{df['payment_method'].value_counts()}")
print(f"  - Churn Distribution:\n{df['churn'].value_counts()}")
print(f"\nCorrelation Matrix (Numerical):")
correlation = df[['age', 'tenure_months', 'monthly_charges', 'total_charges', 
                   'num_support_tickets', 'customer_satisfaction', 'churn']].corr()
print(correlation.round(3))

# ============================================================================
# BƯỚC 6: LƯU VÀO FILE CSV
# ============================================================================

# Tạo thư mục data nếu chưa tồn tại
data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(data_dir, exist_ok=True)

# Đường dẫn file
csv_path = os.path.join(data_dir, 'customer_churn_data.csv')

# Lưu file
df.to_csv(csv_path, index=False)
print(f"\n[SUCCESS] Dữ liệu đã lưu vào: {csv_path}")
print(f"[INFO] Tổng cộng {len(df)} dòng dữ liệu")

# ============================================================================
# BƯỚC 7: KIỂM ĐỊNH DỮ LIỆU
# ============================================================================

print("\n" + "="*70)
print("KIỂM ĐỊNH DỮ LIỆU")
print("="*70)

# Kiểm tra missing values
missing = df.isnull().sum()
if missing.sum() == 0:
    print("✓ Không có missing values")
else:
    print(f"⚠ Có {missing.sum()} missing values:\n{missing[missing > 0]}")

# Kiểm tra kiểu dữ liệu
print("\n✓ Kiểu dữ liệu:")
for col, dtype in df.dtypes.items():
    print(f"  - {col}: {dtype}")

# Kiểm tra khoảng giá trị
print("\n✓ Khoảng giá trị:")
print(f"  - age: {df['age'].min()}-{df['age'].max()}")
print(f"  - tenure_months: {df['tenure_months'].min()}-{df['tenure_months'].max()}")
print(f"  - monthly_charges: ${df['monthly_charges'].min():.2f}-${df['monthly_charges'].max():.2f}")
print(f"  - total_charges: ${df['total_charges'].min():.2f}-${df['total_charges'].max():.2f}")
print(f"  - num_support_tickets: {df['num_support_tickets'].min()}-{df['num_support_tickets'].max()}")
print(f"  - customer_satisfaction: {df['customer_satisfaction'].min()}-{df['customer_satisfaction'].max()}")
print(f"  - churn: {df['churn'].min()}-{df['churn'].max()}")

print("\n[SUCCESS] Quá trình sinh dữ liệu hoàn tất! 🎉")
