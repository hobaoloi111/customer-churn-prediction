"""
BƯỚC 3: BUILD WEB APP BẰNG STREAMLIT
Giao diện web đẹp và dễ sử dụng cho mô hình churn prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Churn Prediction App",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# CACHE & INITIALIZATION
# ============================================================================

@st.cache_resource
def load_model_and_preprocessors():
    """Load model, scaler, encoders một lần"""
    try:
        model_dir = Path('models')
        
        # Load model (demo: dùng random forest)
        model = joblib.load(model_dir / 'random_forest.pkl')
        scaler = joblib.load(model_dir / 'scaler.pkl')
        encoders = joblib.load(model_dir / 'label_encoders.pkl')
        
        with open(model_dir / 'feature_info.json') as f:
            feature_info = json.load(f)
        
        return model, scaler, encoders, feature_info, True
    except:
        # Return None nếu model files không tồn tại
        return None, None, None, None, False

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def preprocess_input(data, scaler, encoders, feature_info):
    """Xử lý input data"""
    df = pd.DataFrame([data])
    
    # Label encode categorical
    for col in feature_info['categorical_cols']:
        if col in encoders:
            df[col] = encoders[col].transform(df[col])
    
    # Scale numerical
    df[feature_info['numerical_cols']] = scaler.transform(
        df[feature_info['numerical_cols']]
    )
    
    return df.values

def predict_churn(data, model, scaler, encoders, feature_info):
    """Predict churn"""
    X_processed = preprocess_input(data, scaler, encoders, feature_info)
    
    prediction = model.predict(X_processed)[0]
    probabilities = model.predict_proba(X_processed)[0]
    
    return {
        'prediction': int(prediction),
        'prob_stayed': float(probabilities[0]),
        'prob_churned': float(probabilities[1]),
        'confidence': float(max(probabilities))
    }

# ============================================================================
# SIDEBAR - MODEL INFO & SELECTION
# ============================================================================

with st.sidebar:
    st.header("🎯 Churn Prediction App")
    st.markdown("---")
    
    # Model info
    st.subheader("📊 Model Information")
    st.info("""
    **Model:** Random Forest Classifier
    - **Training Samples:** 360
    - **Test Samples:** 90
    - **Features:** 10
    - **Accuracy:** ~82%
    - **F1-Score:** ~80%
    """)
    
    st.markdown("---")
    
    # Feature descriptions
    st.subheader("📋 Features Guide")
    with st.expander("Age (Tuổi)"):
        st.write("Tuổi của khách hàng (18-80 tuổi)")
    
    with st.expander("Tenure (Thời gian)"):
        st.write("Bao lâu khách hàng đã là khách của công ty (tháng)")
    
    with st.expander("Monthly Charges (Phí hàng tháng)"):
        st.write("Tiền phí hàng tháng khách hàng phải trả ($)")
    
    with st.expander("Total Charges (Tổng tiền)"):
        st.write("Tổng tiền khách hàng đã chi trả ($)")
    
    with st.expander("Support Tickets"):
        st.write("Số lần khách hàng liên hệ bộ phận hỗ trợ")
    
    with st.expander("Customer Satisfaction"):
        st.write("Mức độ hài lòng của khách hàng (1-5 sao)")

# ============================================================================
# MAIN CONTENT
# ============================================================================

# Load model
model, scaler, encoders, feature_info, model_loaded = load_model_and_preprocessors()

# Header
st.title("🎯 Customer Churn Prediction")
st.markdown("**Dự đoán khách hàng nào sẽ rời đi để có biện pháp giữ chân**")
st.markdown("---")

if not model_loaded:
    st.warning("""
    ⚠️ **Model files not found!**
    
    Vui lòng:
    1. Chạy `python src/save_load_models.py` để lưu mô hình
    2. Đảm bảo thư mục `models/` chứa các files:
       - random_forest.pkl
       - scaler.pkl
       - label_encoders.pkl
       - feature_info.json
    """)
    st.stop()

# Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Analytics", "ℹ️ Help"])

# ============================================================================
# TAB 1: PREDICT
# ============================================================================

with tab1:
    st.header("🔮 Make Prediction")
    st.markdown("Nhập thông tin khách hàng để dự đoán churn:")
    
    # Create 3 columns for layout
    col1, col2, col3 = st.columns(3)
    
    # Column 1: Numerical inputs
    with col1:
        st.subheader("📊 Numerical Features")
        
        age = st.slider(
            "Tuổi",
            min_value=18, max_value=80, value=45,
            help="Tuổi của khách hàng"
        )
        
        tenure_months = st.slider(
            "Thời gian là khách/tháng",
            min_value=0, max_value=120, value=30,
            help="Bao lâu là khách của công ty"
        )
        
        monthly_charges = st.slider(
            "Phí hàng tháng ($)",
            min_value=10.0, max_value=150.0, value=75.0, step=0.5,
            help="Tiền phí hàng tháng"
        )
    
    # Column 2: More numerical inputs
    with col2:
        st.subheader("📊 Numerical Features (cont.)")
        
        total_charges = st.slider(
            "Tổng tiền chi trả ($)",
            min_value=0.0, max_value=10000.0, value=2250.0, step=50.0,
            help="Tổng tiền đã trả"
        )
        
        num_support_tickets = st.slider(
            "Số ticket hỗ trợ",
            min_value=0, max_value=10, value=2,
            help="Số lần liên hệ hỗ trợ"
        )
        
        customer_satisfaction = st.slider(
            "Mức độ hài lòng",
            min_value=1, max_value=5, value=3,
            help="Đánh giá 1-5 sao"
        )
    
    # Column 3: Categorical inputs
    with col3:
        st.subheader("📋 Categorical Features")
        
        contract_type = st.selectbox(
            "Loại hợp đồng",
            options=['Month-to-Month', 'One Year', 'Two Year'],
            help="Hợp đồng dài hay ngắn hạn"
        )
        
        internet_service = st.selectbox(
            "Dịch vụ internet",
            options=['DSL', 'Fiber optic', 'No'],
            help="Loại dịch vụ internet"
        )
        
        payment_method = st.selectbox(
            "Phương thức thanh toán",
            options=['Bank transfer', 'Credit card', 'Check', 'E-check'],
            help="Cách thanh toán"
        )
    
    # Prepare data
    input_data = {
        'age': age,
        'tenure_months': tenure_months,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'num_support_tickets': num_support_tickets,
        'customer_satisfaction': customer_satisfaction,
        'contract_type': contract_type,
        'internet_service': internet_service,
        'payment_method': payment_method
    }
    
    st.markdown("---")
    
    # Predict button
    if st.button("🚀 Predict Churn", use_container_width=True):
        with st.spinner("Đang dự đoán..."):
            result = predict_churn(input_data, model, scaler, encoders, feature_info)
        
        # Display results
        st.markdown("### 📊 Kết quả Dự đoán")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if result['prediction'] == 1:
                st.error("🔴 **CHURN (Sẽ Rời)**")
                emoji = "🔴"
                prediction_text = "Sẽ rời"
            else:
                st.success("🟢 **STAYED (Sẽ Ở Lại)**")
                emoji = "🟢"
                prediction_text = "Sẽ ở lại"
        
        with col2:
            st.metric(
                "Probability Stayed",
                f"{result['prob_stayed']:.1%}",
                help="Xác suất ở lại"
            )
        
        with col3:
            st.metric(
                "Probability Churned",
                f"{result['prob_churned']:.1%}",
                help="Xác suất rời đi"
            )
        
        # Confidence gauge
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Confidence (Độ tin cậy)",
                f"{result['confidence']:.1%}",
                help="Độ tin cậy của dự đoán"
            )
        
        with col2:
            # Confidence level
            if result['confidence'] > 0.85:
                confidence_level = "🟢 Rất cao"
            elif result['confidence'] > 0.70:
                confidence_level = "🟡 Cao"
            else:
                confidence_level = "🔴 Trung bình"
            
            st.info(f"**Confidence Level:** {confidence_level}")
        
        # Recommendation
        st.markdown("---")
        st.markdown("### 💡 Khuyến Nghị Hành Động")
        
        if result['prediction'] == 1:
            if result['prob_churned'] > 0.8:
                st.error("""
                ⚠️ **NGUY CỎ CAO** 
                
                Khách hàng này có xác suất cao sẽ rời.
                
                **Hành động khuyên:**
                - ☎️ Liên hệ trực tiếp ngay lập tức
                - 🎁 Cung cấp ưu đãi đặc biệt
                - 💬 Hỏi lý do không hài lòng
                - 📈 Nâng cấp dịch vụ nếu cần
                """)
            else:
                st.warning("""
                ⚠️ **NGUY CỎ TRUNG BÌNH**
                
                Khách hàng có khả năng rời.
                
                **Hành động khuyên:**
                - 📊 Theo dõi mô hình sử dụng
                - 💼 Kiểm tra sự hài lòng
                - 🎯 Tìm kiếm cơ hội cross-sell
                """)
        else:
            st.success("""
            ✅ **KHÁCH HÀNG ỔN ĐỊNH**
            
            Khách hàng này có xu hướng ở lại.
            
            **Hành động khuyên:**
            - ✨ Duy trì chất lượng dịch vụ
            - 🎁 Chương trình loyalty reward
            - 📞 Check-in định kỳ
            - 🚀 Upsell/cross-sell sản phẩm mới
            """)
        
        # Input summary
        st.markdown("---")
        st.markdown("### 📋 Input Summary")
        
        df_input = pd.DataFrame([input_data]).T
        df_input.columns = ['Value']
        st.dataframe(df_input)

# ============================================================================
# TAB 2: ANALYTICS
# ============================================================================

with tab2:
    st.header("📊 Model Analytics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", "82.1%")
    with col2:
        st.metric("F1-Score", "80.5%")
    with col3:
        st.metric("ROC-AUC", "88.3%")
    with col4:
        st.metric("Precision", "81.2%")
    
    st.markdown("---")
    
    # Sample predictions
    st.subheader("📈 Sample Predictions")
    
    samples = {
        'High Risk': {
            'age': 30, 'tenure_months': 3, 'monthly_charges': 120,
            'total_charges': 360, 'num_support_tickets': 5,
            'customer_satisfaction': 2, 'contract_type': 'Month-to-Month',
            'internet_service': 'Fiber optic', 'payment_method': 'E-check'
        },
        'Medium Risk': {
            'age': 45, 'tenure_months': 30, 'monthly_charges': 75,
            'total_charges': 2250, 'num_support_tickets': 2,
            'customer_satisfaction': 3, 'contract_type': 'Month-to-Month',
            'internet_service': 'DSL', 'payment_method': 'Credit card'
        },
        'Low Risk': {
            'age': 55, 'tenure_months': 60, 'monthly_charges': 60,
            'total_charges': 3600, 'num_support_tickets': 1,
            'customer_satisfaction': 5, 'contract_type': 'Two Year',
            'internet_service': 'DSL', 'payment_method': 'Bank transfer'
        }
    }
    
    results_data = []
    for scenario, data in samples.items():
        result = predict_churn(data, model, scaler, encoders, feature_info)
        results_data.append({
            'Scenario': scenario,
            'Prediction': 'Churned' if result['prediction'] == 1 else 'Stayed',
            'Prob. Stayed': f"{result['prob_stayed']:.1%}",
            'Prob. Churned': f"{result['prob_churned']:.1%}",
            'Confidence': f"{result['confidence']:.1%}"
        })
    
    df_results = pd.DataFrame(results_data)
    st.dataframe(df_results, use_container_width=True)

# ============================================================================
# TAB 3: HELP
# ============================================================================

with tab3:
    st.header("ℹ️ Help & Documentation")
    
    st.markdown("""
    ### 🎯 Mục Đích Ứng Dụng
    Dự đoán khách hàng nào sẽ rời đi để công ty có biện pháp giữ chân kịp thời.
    
    ### 📊 Dữ Liệu Đầu Vào
    - **Tuổi:** Tuổi khách hàng (18-80)
    - **Thời gian:** Bao lâu là khách của công ty (0-120 tháng)
    - **Phí hàng tháng:** Tiền phí khách hàng trả mỗi tháng
    - **Tổng tiền:** Tổng tiền khách hàng đã chi trả
    - **Support Tickets:** Số lần khách hàng liên hệ support
    - **Mức độ hài lòng:** Đánh giá 1-5 sao
    - **Loại hợp đồng:** Month-to-Month / One Year / Two Year
    - **Dịch vụ internet:** DSL / Fiber optic / None
    - **Thanh toán:** Bank transfer / Credit card / Check / E-check
    
    ### 🔮 Kết Quả Dự Đoán
    - **Stayed (🟢):** Khách hàng sẽ ở lại
    - **Churned (🔴):** Khách hàng sẽ rời đi
    - **Confidence:** Độ tin cậy của dự đoán (0-100%)
    
    ### 💡 Làm Thế Nào Để Sử Dụng
    1. Nhập thông tin khách hàng trong Tab "Predict"
    2. Click "Predict Churn"
    3. Xem kết quả và khuyến nghị
    4. Thực hiện hành động phù hợp
    
    ### 📞 Hỗ Trợ
    Liên hệ: AI Team | Email: ai@company.com
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p>🎯 Customer Churn Prediction v1.0 | Powered by Streamlit & Scikit-Learn</p>
</div>
""", unsafe_allow_html=True)
