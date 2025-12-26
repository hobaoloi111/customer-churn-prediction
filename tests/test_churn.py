"""
BƯỚC 5: UNIT TESTS
Viết test cases cho các functions chính
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# ============================================================================
# TEST: DATA LOADING
# ============================================================================

class TestDataLoading:
    """Test cases cho data loading"""
    
    def test_csv_file_exists(self):
        """Test: CSV file tồn tại"""
        data_file = Path(__file__).parent.parent / 'data' / 'customer_churn_data.csv'
        assert data_file.exists(), f"CSV file not found: {data_file}"
    
    def test_csv_load_successfully(self):
        """Test: Load CSV thành công"""
        data_file = Path(__file__).parent.parent / 'data' / 'customer_churn_data.csv'
        df = pd.read_csv(data_file)
        assert not df.empty, "CSV file is empty"
        assert len(df) > 0, "Dataset has no rows"
    
    def test_required_columns_exist(self):
        """Test: Các cột bắt buộc tồn tại"""
        data_file = Path(__file__).parent.parent / 'data' / 'customer_churn_data.csv'
        df = pd.read_csv(data_file)
        
        required_cols = ['age', 'tenure_months', 'monthly_charges', 'churn']
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"
    
    def test_no_missing_values(self):
        """Test: Không có missing values"""
        data_file = Path(__file__).parent.parent / 'data' / 'customer_churn_data.csv'
        df = pd.read_csv(data_file)
        
        assert df.isnull().sum().sum() == 0, "Found missing values in dataset"
    
    def test_target_is_binary(self):
        """Test: Target column chỉ có 2 giá trị (0, 1)"""
        data_file = Path(__file__).parent.parent / 'data' / 'customer_churn_data.csv'
        df = pd.read_csv(data_file)
        
        unique_values = df['churn'].unique()
        assert len(unique_values) == 2, f"Target not binary: {unique_values}"
        assert set(unique_values) == {0, 1}, f"Target values not {0, 1}: {unique_values}"

# ============================================================================
# TEST: DATA PREPROCESSING
# ============================================================================

class TestPreprocessing:
    """Test cases cho data preprocessing"""
    
    @pytest.fixture
    def sample_data(self):
        """Fixture: Sample data"""
        return pd.DataFrame({
            'age': [25, 35, 45],
            'tenure_months': [10, 20, 30],
            'monthly_charges': [50.0, 75.0, 100.0],
            'contract_type': ['Month-to-Month', 'One Year', 'Two Year']
        })
    
    def test_label_encoder_initialization(self, sample_data):
        """Test: LabelEncoder khởi tạo đúng"""
        le = LabelEncoder()
        encoded = le.fit_transform(sample_data['contract_type'])
        
        assert len(encoded) == len(sample_data), "Encoded size mismatch"
        assert all(isinstance(x, (int, np.integer)) for x in encoded), "Not all encoded values are integers"
    
    def test_label_encoder_inverse(self, sample_data):
        """Test: LabelEncoder có thể reverse"""
        le = LabelEncoder()
        original = sample_data['contract_type'].values
        encoded = le.fit_transform(original)
        decoded = le.inverse_transform(encoded)
        
        assert list(original) == list(decoded), "Inverse transformation failed"
    
    def test_scaler_centering(self, sample_data):
        """Test: Scaler chuẩn hóa dữ liệu"""
        scaler = StandardScaler()
        X = sample_data[['age', 'tenure_months', 'monthly_charges']].values
        X_scaled = scaler.fit_transform(X)
        
        # Check mean ≈ 0, std ≈ 1
        assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-10), "Mean not centered to 0"
        assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-10), "Std not scaled to 1"
    
    def test_scaler_consistency(self, sample_data):
        """Test: Scaler consistent khi fit once"""
        scaler = StandardScaler()
        X = sample_data[['age', 'tenure_months', 'monthly_charges']].values
        
        scaler.fit(X)
        X_scaled1 = scaler.transform(X)
        X_scaled2 = scaler.transform(X)
        
        assert np.allclose(X_scaled1, X_scaled2), "Scaler not consistent"

# ============================================================================
# TEST: VALIDATION
# ============================================================================

class TestInputValidation:
    """Test cases cho input validation"""
    
    def test_age_validation_valid(self):
        """Test: Tuổi hợp lệ (18-80)"""
        age_values = [18, 30, 45, 80]
        for age in age_values:
            assert 18 <= age <= 80, f"Age {age} out of range"
    
    def test_age_validation_invalid(self):
        """Test: Tuổi không hợp lệ"""
        invalid_ages = [17, 81, -5, 150]
        for age in invalid_ages:
            assert not (18 <= age <= 80), f"Age {age} should be invalid"
    
    def test_categorical_validation(self):
        """Test: Categorical values hợp lệ"""
        valid_contracts = ['Month-to-Month', 'One Year', 'Two Year']
        
        for contract in valid_contracts:
            assert contract in valid_contracts
    
    def test_invalid_categorical(self):
        """Test: Categorical values không hợp lệ"""
        valid_contracts = ['Month-to-Month', 'One Year', 'Two Year']
        invalid_contract = 'Three Years'
        
        assert invalid_contract not in valid_contracts

# ============================================================================
# TEST: MODEL PREDICTIONS
# ============================================================================

class TestModelPredictions:
    """Test cases cho model predictions"""
    
    @pytest.fixture
    def dummy_model(self):
        """Fixture: Dummy model cho testing"""
        from sklearn.ensemble import RandomForestClassifier
        
        # Create simple dummy data
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        
        model = RandomForestClassifier(random_state=42, n_estimators=10)
        model.fit(X, y)
        
        return model
    
    def test_prediction_shape(self, dummy_model):
        """Test: Prediction shape đúng"""
        X_test = np.random.randn(10, 5)
        y_pred = dummy_model.predict(X_test)
        
        assert len(y_pred) == len(X_test), "Prediction length mismatch"
    
    def test_prediction_values(self, dummy_model):
        """Test: Prediction values chỉ có 0 hoặc 1"""
        X_test = np.random.randn(10, 5)
        y_pred = dummy_model.predict(X_test)
        
        assert all(pred in [0, 1] for pred in y_pred), "Predictions not binary"
    
    def test_probability_shape(self, dummy_model):
        """Test: Probability shape đúng"""
        X_test = np.random.randn(10, 5)
        y_proba = dummy_model.predict_proba(X_test)
        
        assert y_proba.shape == (len(X_test), 2), "Probability shape mismatch"
    
    def test_probability_sum_to_one(self, dummy_model):
        """Test: Probability sum = 1"""
        X_test = np.random.randn(10, 5)
        y_proba = dummy_model.predict_proba(X_test)
        
        sums = y_proba.sum(axis=1)
        assert np.allclose(sums, 1.0), "Probabilities don't sum to 1"

# ============================================================================
# TEST: METRICS
# ============================================================================

class TestMetrics:
    """Test cases cho evaluation metrics"""
    
    def test_accuracy_range(self):
        """Test: Accuracy dalam range [0, 1]"""
        from sklearn.metrics import accuracy_score
        
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]
        
        acc = accuracy_score(y_true, y_pred)
        assert 0 <= acc <= 1, f"Accuracy {acc} out of range"
    
    def test_f1_score_calculation(self):
        """Test: F1 score calculation"""
        from sklearn.metrics import f1_score
        
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]
        
        f1 = f1_score(y_true, y_pred)
        assert 0 <= f1 <= 1, f"F1 score {f1} out of range"
    
    def test_confusion_matrix_shape(self):
        """Test: Confusion matrix shape"""
        from sklearn.metrics import confusion_matrix
        
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]
        
        cm = confusion_matrix(y_true, y_pred)
        assert cm.shape == (2, 2), f"Confusion matrix shape {cm.shape} not (2, 2)"

# ============================================================================
# TEST: EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Test cases cho edge cases"""
    
    def test_single_sample_prediction(self):
        """Test: Prediction cho 1 sample"""
        from sklearn.ensemble import RandomForestClassifier
        
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)
        
        X_test = np.random.randn(1, 5)
        y_pred = model.predict(X_test)
        
        assert len(y_pred) == 1, "Single sample prediction failed"
    
    def test_all_same_class(self):
        """Test: Data với tất cả cùng class"""
        from sklearn.ensemble import RandomForestClassifier
        
        X = np.random.randn(50, 5)
        y = np.ones(50)  # Tất cả class 1
        
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)
        
        y_pred = model.predict(X)
        assert all(pred == 1 for pred in y_pred), "Model should predict all 1s"
    
    def test_duplicate_samples(self):
        """Test: Data có duplicate samples"""
        from sklearn.ensemble import RandomForestClassifier
        
        X = np.array([[1, 2], [1, 2], [3, 4]])
        y = np.array([0, 0, 1])
        
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)
        
        assert model.n_estimators == 5, "Model not trained with duplicates"

# ============================================================================
# PYTEST MARKERS & ORGANIZATION
# ============================================================================

"""
Để chạy tests:

1. Cài pytest:
   pip install pytest pytest-cov

2. Chạy tất cả tests:
   pytest test_churn.py -v

3. Chạy một class tests:
   pytest test_churn.py::TestDataLoading -v

4. Chạy một test function:
   pytest test_churn.py::TestDataLoading::test_csv_file_exists -v

5. Chạy với coverage:
   pytest test_churn.py --cov=src --cov-report=html

6. Chạy tests với keyword:
   pytest test_churn.py -k "validation" -v

7. Show print statements:
   pytest test_churn.py -s
"""

if __name__ == '__main__':
    print("""
    ✅ TEST SUITE - Customer Churn Prediction
    
    Chạy tests bằng lệnh:
      pytest test_churn.py -v
    """)
