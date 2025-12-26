"""
BƯỚC 2: BUILD CLI APPLICATION
Ứng dụng dòng lệnh (Command Line Interface) để predict churn
"""

import joblib
import json
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ============================================================================
# PHẦN 1: LOAD MODEL & PREPROCESSORS
# ============================================================================

class ChurnPredictionApp:
    """
    CLI Application để predict Customer Churn
    """
    
    def __init__(self, model_dir='models'):
        """
        Khởi tạo app và load model
        
        Args:
            model_dir: Thư mục chứa model files
        """
        self.model_dir = Path(model_dir)
        self.model = None
        self.scaler = None
        self.encoders = None
        self.feature_info = None
        self.is_loaded = False
        
    def load_model_package(self, model_name='random_forest'):
        """
        Tải model + preprocessors từ files
        
        Args:
            model_name: Tên model (e.g., 'random_forest', 'logistic_regression')
        """
        try:
            model_path = self.model_dir / f'{model_name}.pkl'
            scaler_path = self.model_dir / 'scaler.pkl'
            encoders_path = self.model_dir / 'label_encoders.pkl'
            feature_info_path = self.model_dir / 'feature_info.json'
            
            # Check file tồn tại
            if not all([model_path.exists(), scaler_path.exists(), 
                       encoders_path.exists(), feature_info_path.exists()]):
                print("❌ Model files not found!")
                print(f"   Expected in: {self.model_dir}")
                return False
            
            # Load model
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.encoders = joblib.load(encoders_path)
            
            with open(feature_info_path, 'r') as f:
                self.feature_info = json.load(f)
            
            self.is_loaded = True
            print(f"✅ Model loaded successfully: {model_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False
    
    def validate_input(self, data):
        """
        Validate input data
        
        Args:
            data (dict): Data từ user input
            
        Returns:
            (bool, str): (is_valid, error_message)
        """
        # Kiểm tra required features
        required_features = self.feature_info['feature_names']
        for feature in required_features:
            if feature not in data:
                return False, f"Missing feature: {feature}"
        
        # Kiểm tra kiểu dữ liệu và khoảng giá trị
        validation_rules = {
            'age': (18, 80),
            'tenure_months': (0, 120),
            'monthly_charges': (10, 150),
            'total_charges': (0, 10000),
            'num_support_tickets': (0, 10),
            'customer_satisfaction': (1, 5)
        }
        
        for feature, (min_val, max_val) in validation_rules.items():
            if feature in data:
                try:
                    val = float(data[feature])
                    if not (min_val <= val <= max_val):
                        return False, f"{feature} must be between {min_val}-{max_val}"
                except ValueError:
                    return False, f"{feature} must be a number"
        
        # Kiểm tra categorical features
        categorical_rules = {
            'contract_type': ['Month-to-Month', 'One Year', 'Two Year'],
            'internet_service': ['DSL', 'Fiber optic', 'No'],
            'payment_method': ['Bank transfer', 'Credit card', 'Check', 'E-check']
        }
        
        for feature, valid_values in categorical_rules.items():
            if feature in data:
                if data[feature] not in valid_values:
                    return False, f"{feature} must be one of: {', '.join(valid_values)}"
        
        return True, ""
    
    def preprocess_input(self, data):
        """
        Preprocess input data (encode, scale)
        
        Args:
            data (dict): Raw input data
            
        Returns:
            np.array: Processed data
        """
        df = pd.DataFrame([data])
        
        # Label encode categorical features
        for col in self.feature_info['categorical_cols']:
            if col in self.encoders:
                df[col] = self.encoders[col].transform(df[col])
        
        # Standard scale numerical features
        df[self.feature_info['numerical_cols']] = self.scaler.transform(
            df[self.feature_info['numerical_cols']]
        )
        
        return df.values
    
    def predict(self, data):
        """
        Predict churn untuk input data
        
        Args:
            data (dict): Input data từ user
            
        Returns:
            dict: Kết quả prediction {prediction, probability, confidence}
        """
        # Validate input
        is_valid, error_msg = self.validate_input(data)
        if not is_valid:
            return {'error': error_msg}
        
        # Preprocess
        X_processed = self.preprocess_input(data)
        
        # Predict
        prediction = self.model.predict(X_processed)[0]
        probabilities = self.model.predict_proba(X_processed)[0]
        
        return {
            'prediction': int(prediction),
            'prediction_text': 'Churned (Sẽ rời)' if prediction == 1 else 'Stayed (Sẽ ở lại)',
            'probability_stayed': float(probabilities[0]),
            'probability_churned': float(probabilities[1]),
            'confidence': float(max(probabilities))
        }

# ============================================================================
# PHẦN 2: MAIN CLI LOOP
# ============================================================================

def get_user_input():
    """
    Nhập dữ liệu từ user
    
    Returns:
        dict: Input data
    """
    print("\n" + "="*70)
    print("NHẬP THÔNG TIN KHÁCH HÀNG")
    print("="*70)
    
    data = {}
    
    # Numerical inputs
    print("\n📊 NUMERICAL FEATURES:")
    try:
        data['age'] = float(input("  Tuổi (18-80): "))
        data['tenure_months'] = float(input("  Thời gian là khách hàng/tháng (0-120): "))
        data['monthly_charges'] = float(input("  Phí hàng tháng/$ (10-150): "))
        data['total_charges'] = float(input("  Tổng tiền chi trả/$ (0-10000): "))
        data['num_support_tickets'] = float(input("  Số lượng ticket hỗ trợ (0-10): "))
        data['customer_satisfaction'] = float(input("  Mức độ hài lòng/5 (1-5): "))
    except ValueError:
        print("❌ Lỗi: Vui lòng nhập số hợp lệ")
        return None
    
    # Categorical inputs
    print("\n📋 CATEGORICAL FEATURES:")
    
    print("  Loại hợp đồng:")
    print("    1. Month-to-Month")
    print("    2. One Year")
    print("    3. Two Year")
    choice = input("  Chọn (1-3): ")
    contract_map = {'1': 'Month-to-Month', '2': 'One Year', '3': 'Two Year'}
    data['contract_type'] = contract_map.get(choice)
    
    print("  Dịch vụ internet:")
    print("    1. DSL")
    print("    2. Fiber optic")
    print("    3. No")
    choice = input("  Chọn (1-3): ")
    internet_map = {'1': 'DSL', '2': 'Fiber optic', '3': 'No'}
    data['internet_service'] = internet_map.get(choice)
    
    print("  Phương thức thanh toán:")
    print("    1. Bank transfer")
    print("    2. Credit card")
    print("    3. Check")
    print("    4. E-check")
    choice = input("  Chọn (1-4): ")
    payment_map = {'1': 'Bank transfer', '2': 'Credit card', '3': 'Check', '4': 'E-check'}
    data['payment_method'] = payment_map.get(choice)
    
    return data

def display_prediction(result):
    """
    Hiển thị kết quả prediction
    
    Args:
        result: Dict kết quả từ predict()
    """
    print("\n" + "="*70)
    print("KẾT QUẢ DỰ ĐOÁN")
    print("="*70)
    
    if 'error' in result:
        print(f"❌ Lỗi: {result['error']}")
        return
    
    prediction = result['prediction_text']
    prob_churned = result['probability_churned']
    confidence = result['confidence']
    
    # Display result
    if result['prediction'] == 1:
        emoji = "🔴"
        color = "Churned"
    else:
        emoji = "🟢"
        color = "Stayed"
    
    print(f"\n{emoji} Kết quả: {prediction}")
    print(f"\n📊 Xác suất:")
    print(f"  - Sẽ ở lại (Stayed): {result['probability_stayed']*100:>6.2f}%")
    print(f"  - Sẽ rời (Churned):  {prob_churned*100:>6.2f}%")
    print(f"\n💪 Độ tin cậy: {confidence*100:.2f}%")
    
    # Recommendation
    print(f"\n💡 KHUYẾN NGHỊ:")
    if result['prediction'] == 1 and prob_churned > 0.8:
        print("  ⚠️  Khách hàng này có NGUY CỎ CAO sẽ rời")
        print("      → Nên liên hệ và ưu đãi ngay lập tức")
    elif result['prediction'] == 1 and prob_churned > 0.6:
        print("  ⚠️  Khách hàng này có NGUY CỎ TRUNG BÌNH")
        print("      → Nên theo dõi và chuẩn bị biện pháp")
    else:
        print("  ✅ Khách hàng này ỔNĐỊNH")
        print("      → Duy trì dịch vụ chất lượng")

def main():
    """
    Main CLI loop
    """
    print("\n" + "="*70)
    print("🎯 CUSTOMER CHURN PREDICTION - CLI APPLICATION")
    print("="*70)
    
    # Initialize app
    app = ChurnPredictionApp(model_dir='models')
    
    # Try load model - if not exist, create dummy
    if not app.load_model_package('random_forest'):
        print("\n⚠️  Mô hình chưa được lưu")
        print("   Chạy: python src/save_load_models.py trước")
        
        # Demo mode with fake predictions
        print("\n[DEMO MODE] Sử dụng dự đoán giả lập...\n")
        app.is_loaded = False  # Mark as demo
    
    # Main loop
    while True:
        print("\n" + "="*70)
        print("MENU")
        print("="*70)
        print("1. Dự đoán churn cho khách hàng mới")
        print("2. Test với mẫu dữ liệu")
        print("3. Giải thích metrics")
        print("4. Thoát")
        
        choice = input("\nChọn (1-4): ").strip()
        
        if choice == '1':
            # User input prediction
            data = get_user_input()
            if data:
                if app.is_loaded:
                    result = app.predict(data)
                    display_prediction(result)
                else:
                    # Demo prediction
                    print("\n[DEMO] Kết quả mô phỏng:")
                    print(f"  Sẽ ở lại (Stayed): 65.00%")
                    print(f"  Sẽ rời (Churned):  35.00%")
        
        elif choice == '2':
            # Sample data
            print("\n[TEST DATA] Sử dụng mẫu dữ liệu:")
            sample_data = {
                'age': 45,
                'tenure_months': 30,
                'monthly_charges': 75.5,
                'total_charges': 2265,
                'num_support_tickets': 2,
                'customer_satisfaction': 3,
                'contract_type': 'Month-to-Month',
                'internet_service': 'Fiber optic',
                'payment_method': 'E-check'
            }
            
            if app.is_loaded:
                result = app.predict(sample_data)
                display_prediction(result)
            else:
                print("  (Demo mode - mô phỏng kết quả)")
        
        elif choice == '3':
            # Explain metrics
            print("""
💡 GIẢI THÍCH METRICS:

🔴 CHURNED (Sẽ Rời):
  - Khách hàng dự kiến sẽ hủy dịch vụ
  - Cần biện pháp can thiệp (ưu đãi, liên hệ)
  - Tầm quan trọng: CAO

🟢 STAYED (Sẽ Ở Lại):
  - Khách hàng dự kiến sẽ tiếp tục sử dụng
  - Duy trì mối quan hệ tốt
  - Tầm quan trọng: BÌNH THƯỜNG

📊 PROBABILITY (XÁC SUẤT):
  - Càng cao → Càng tin cậy dự đoán
  - > 80% → Rất tin cậy
  - 60-80% → Tin cậy trung bình
  - < 60% → Cần xem xét kỹ

💪 CONFIDENCE (ĐỘ TIN CẬY):
  - Mức độ chắc chắn của mô hình
  - Cao → Model rất tự tin
  - Thấp → Cần xem xét thêm factors khác
            """)
        
        elif choice == '4':
            print("\n👋 Tạm biệt!")
            break
        
        else:
            print("❌ Lựa chọn không hợp lệ")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Ứng dụng đã đóng")
    except Exception as e:
        print(f"\n❌ Lỗi: {str(e)}")
        sys.exit(1)
