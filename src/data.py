"""
Module: Data Generation and Management
Mục đích: Tạo dữ liệu giả lập cho bài toán Customer Churn Prediction
Tác giả: Student
Ngày: 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CustomerDataGenerator:
    """
    Lớp để sinh dữ liệu giả lập về khách hàng.
    
    Attributes:
        n_samples (int): Số lượng bản ghi cần tạo
        random_seed (int): Seed cho random để có kết quả lặp lại được
    """
    
    def __init__(self, n_samples=400, random_seed=42):
        """
        Khởi tạo CustomerDataGenerator
        
        Args:
            n_samples (int): Số lượng khách hàng (mặc định 400)
            random_seed (int): Seed cho numpy random (mặc định 42)
        """
        self.n_samples = n_samples
        self.random_seed = random_seed
        np.random.seed(random_seed)
        logger.info(f"Khởi tạo CustomerDataGenerator với {n_samples} samples")
    
    def generate_data(self):
        """
        Sinh dữ liệu giả lập có tương quan hợp lý.
        
        Cấu trúc dữ liệu:
        - customer_id: ID khách hàng (1-400)
        - tenure_months: Thời gian sử dụng dịch vụ (tháng) - từ 1 đến 60
        - total_spend: Tổng chi tiêu (nghìn đồng) - từ 200 đến 20000
        - support_calls: Số lần liên hệ hỗ trợ (0-10)
        - service_type: Loại gói dịch vụ (Basic, Standard, Premium)
        - usage_rate: Tỷ lệ sử dụng dịch vụ (%) - từ 10 đến 100
        - churn: Nhãn (0=giữ lại khách, 1=khách rời bỏ dịch vụ)
        
        Correlation (mối liên hệ thực tế):
        - Khách sử dụng lâu (tenure cao) → Xác suất churn thấp
        - Chi tiêu cao → Xác suất churn thấp
        - Support calls nhiều → Xác suất churn cao (khách gặp vấn đề)
        - Usage rate cao → Xác suất churn thấp (khách dùng nhiều)
        - Service type Premium → Xác suất churn thấp
        
        Returns:
            pd.DataFrame: DataFrame chứa dữ liệu khách hàng
        """
        
        logger.info("Bắt đầu tạo dữ liệu...")
        
        # 1. Customer ID: từ 1 đến n_samples
        customer_id = np.arange(1, self.n_samples + 1)
        
        # 2. Tenure (tháng): phân phối lệch phải (nhiều khách dùng ngắn hạn)
        #    Sử dụng exponential distribution để tạo độ lệch
        tenure_months = np.random.exponential(scale=12, size=self.n_samples)
        tenure_months = np.clip(tenure_months, 1, 60).astype(int)  # Giới hạn 1-60 tháng
        
        # 3. Service Type: phân bố không đều (Basic phổ biến hơn)
        #    Basic: 50%, Standard: 30%, Premium: 20%
        service_type = np.random.choice(
            ['Basic', 'Standard', 'Premium'],
            size=self.n_samples,
            p=[0.5, 0.3, 0.2]
        )
        
        # 4. Total Spend: tương quan với (tenure, service_type, usage_rate)
        #    Base spend theo service type
        service_base_spend = {'Basic': 500, 'Standard': 1500, 'Premium': 3000}
        total_spend = np.array([service_base_spend[st] for st in service_type])
        
        # Thêm biến động dựa trên tenure (khách cũ thường chi tiêu hơn)
        total_spend = total_spend + (tenure_months * 100) + np.random.normal(0, 300, self.n_samples)
        total_spend = np.clip(total_spend, 100, 25000).astype(int)  # Giới hạn 100-25000
        
        # 5. Usage Rate (%): tương quan với service_type
        #    Premium users có usage rate cao hơn
        usage_rate_base = {'Basic': 30, 'Standard': 60, 'Premium': 80}
        usage_rate = np.array([usage_rate_base[st] for st in service_type])
        usage_rate = usage_rate + np.random.normal(0, 15, self.n_samples)
        usage_rate = np.clip(usage_rate, 10, 100).astype(int)  # Giới hạn 10-100%
        
        # 6. Support Calls: tương quan với usage_rate (dùng nhiều → support nhiều)
        #    Nhưng cũng có khách call support vì có vấn đề (usage rate thấp)
        support_calls = np.abs(np.random.poisson(lam=3, size=self.n_samples) - 2)
        # Điều chỉnh: khách dùng ít hoặc dùng rất nhiều sẽ call support nhiều
        support_calls = support_calls + np.random.binomial(5, (100 - usage_rate) / 100)
        support_calls = np.clip(support_calls, 0, 10).astype(int)  # Giới hạn 0-10
        
        # 7. Churn (Target): tính toán dựa trên các features
        #    Khách càng lâu, chi tiêu cao, dùng nhiều → churn risk thấp
        churn_probability = 0.5  # Base probability
        
        # Giảm churn probability cho khách sử dụng lâu
        churn_probability = churn_probability - (tenure_months / 100)  # Tối đa -0.6
        
        # Giảm churn probability cho khách Premium
        churn_probability = churn_probability - (service_type == 'Premium') * 0.15
        churn_probability = churn_probability - (service_type == 'Standard') * 0.05
        
        # Giảm churn probability cho khách chi tiêu cao
        churn_probability = churn_probability - (total_spend > 5000) * 0.1
        
        # Giảm churn probability cho khách dùng nhiều
        churn_probability = churn_probability - (usage_rate > 70) * 0.2
        
        # TĂNG churn probability cho khách call support nhiều (dấu hiệu vấn đề)
        churn_probability = churn_probability + (support_calls > 5) * 0.3
        
        # Giới hạn xác suất trong [0, 1]
        churn_probability = np.clip(churn_probability, 0, 1)
        
        # Chuyển xác suất thành nhãn 0/1
        churn = np.random.binomial(n=1, p=churn_probability, size=self.n_samples)
        
        # 8. Tạo DataFrame
        df = pd.DataFrame({
            'customer_id': customer_id,
            'tenure_months': tenure_months,
            'total_spend': total_spend,
            'support_calls': support_calls,
            'service_type': service_type,
            'usage_rate': usage_rate,
            'churn': churn
        })
        
        logger.info(f"Đã tạo {len(df)} bản ghi dữ liệu")
        logger.info(f"Tỷ lệ churn: {df['churn'].mean():.2%}")
        
        return df
    
    def save_to_csv(self, df, filepath):
        """
        Lưu DataFrame thành file CSV
        
        Args:
            df (pd.DataFrame): DataFrame cần lưu
            filepath (str): Đường dẫn file CSV
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.info(f"Đã lưu dữ liệu vào: {filepath}")
    
    @staticmethod
    def load_from_csv(filepath):
        """
        Tải dữ liệu từ file CSV
        
        Args:
            filepath (str): Đường dẫn file CSV
            
        Returns:
            pd.DataFrame: DataFrame được tải
        """
        df = pd.read_csv(filepath)
        logger.info(f"Đã tải {len(df)} bản ghi từ: {filepath}")
        return df


def get_data_info(df):
    """
    Hiển thị thông tin thống kê cơ bản về dataset
    
    Args:
        df (pd.DataFrame): DataFrame cần kiểm tra
    """
    print("\n" + "="*60)
    print("THÔNG TIN DATASET")
    print("="*60)
    
    print(f"\n📊 Kích thước: {df.shape[0]} dòng, {df.shape[1]} cột")
    
    print("\n📋 Thông tin cột:")
    print(df.info())
    
    print("\n📈 Thống kê mô tả:")
    print(df.describe())
    
    print("\n🎯 Tỷ lệ churn:")
    print(df['churn'].value_counts(normalize=True))
    
    print("\n📝 Service type distribution:")
    print(df['service_type'].value_counts())
    
    print("\n⚠️  Giá trị thiếu (Missing values):")
    print(df.isnull().sum())
    
    print("\n" + "="*60)


# Hàm chính để chạy standalone
if __name__ == "__main__":
    """
    Script chạy standalone để sinh dữ liệu
    Chạy: python src/data.py
    """
    
    # Tạo dữ liệu
    generator = CustomerDataGenerator(n_samples=400, random_seed=42)
    df = generator.generate_data()
    
    # Hiển thị thông tin
    get_data_info(df)
    
    # Lưu vào file
    data_path = Path(__file__).parent.parent / "data" / "customer_data.csv"
    generator.save_to_csv(df, str(data_path))
    
    # Hiển thị 5 dòng đầu
    print("\n📄 5 dòng đầu tiên của dataset:")
    print(df.head(10))
