"""
BƯỚC 6: DEPLOYMENT GUIDE - STREAMLIT CLOUD
Hướng dẫn chi tiết deploy ứng dụng lên Streamlit Cloud
"""

# ============================================================================
# STEP 1: PREPARE FILES
# ============================================================================

deployment_guide = """
╔════════════════════════════════════════════════════════════════════════════╗
║ BƯỚC 6: DEPLOYMENT GUIDE - STREAMLIT CLOUD                               ║
╚════════════════════════════════════════════════════════════════════════════╝

📋 BƯỚC 1: CHUẨN BỊ FILES
════════════════════════════════════════════════════════════════════════════

Các file cần thiết:

1. 📄 requirements.txt (thư viện dependencies)
   Nội dung:
   ──────────────────────────────────────────
   streamlit==1.28.0
   pandas==2.1.0
   numpy==1.24.0
   scikit-learn==1.3.0
   joblib==1.3.0
   ──────────────────────────────────────────

2. 📄 streamlit_app.py (app chính - sử dụng tên này hoặc app.py)
   Location: project_root/streamlit_app.py (hoặc src/streamlit_app.py)

3. 📁 models/ (thư mục chứa trained models)
   ├── random_forest.pkl
   ├── scaler.pkl
   ├── label_encoders.pkl
   └── feature_info.json

4. 📄 .gitignore (bỏ qua files không cần push)
   Nội dung:
   ──────────────────────────────────────────
   .venv/
   __pycache__/
   *.pyc
   .env
   .DS_Store
   ──────────────────────────────────────────

5. 📄 README.md (mô tả project)
   Xem bên dưới


════════════════════════════════════════════════════════════════════════════
📋 BƯỚC 2: GIT & GITHUB SETUP
════════════════════════════════════════════════════════════════════════════

1️⃣ Tạo GitHub repository:
   - Đăng nhập: https://github.com
   - Click "New repository"
   - Tên: "customer-churn-prediction"
   - Description: "ML App to predict customer churn"
   - Public (để Streamlit Cloud có thể access)
   - Click "Create repository"

2️⃣ Push code lên GitHub:
   ──────────────────────────────────────────
   cd /path/to/customer_churn_prediction
   
   git init
   git add .
   git commit -m "Initial commit: Churn prediction app"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/customer-churn-prediction.git
   git push -u origin main
   ──────────────────────────────────────────

3️⃣ Verify:
   - Vào https://github.com/YOUR_USERNAME/customer-churn-prediction
   - Kiểm tra files có được push đúng không


════════════════════════════════════════════════════════════════════════════
📋 BƯỚC 3: STREAMLIT CLOUD DEPLOYMENT
════════════════════════════════════════════════════════════════════════════

1️⃣ Đăng ký Streamlit Cloud:
   - Vào: https://streamlit.io/cloud
   - Click "Sign up"
   - Đăng nhập bằng GitHub account

2️⃣ Tạo app mới:
   - Click "New app"
   - Repository: chọn "customer-churn-prediction"
   - Branch: "main"
   - File path: "src/streamlit_app.py" (hoặc "streamlit_app.py")
   - Click "Deploy"

3️⃣ Chờ deployment:
   - Streamlit tự động build & deploy
   - Có thể mất 1-2 phút
   - Kiểm tra logs nếu có error

4️⃣ Kiểm tra URL:
   - App URL: https://share.streamlit.io/YOUR_USERNAME/customer-churn-prediction/main/streamlit_app.py
   - (Streamlit tự sinh URL)


════════════════════════════════════════════════════════════════════════════
⚠️ BƯỚC 4: TROUBLESHOOTING
════════════════════════════════════════════════════════════════════════════

❌ LỖI 1: "ModuleNotFoundError: No module named 'streamlit'"
   ✅ FIX: Thêm streamlit vào requirements.txt

❌ LỖI 2: "FileNotFoundError: models/random_forest.pkl"
   ✅ FIX: Đảm bảo thư mục models/ được push lên GitHub
   Note: Nếu files quá lớn, dùng DVC (Data Version Control) hoặc upload lên cloud storage

❌ LỖI 3: App chạy chậm
   ✅ FIX: 
   - Dùng @st.cache_resource để cache models
   - Giảm model size nếu có thể
   - Tối ưu code

❌ LỖI 4: "Permission denied" khi push lên GitHub
   ✅ FIX:
   - Tạo Personal Access Token: https://github.com/settings/tokens
   - Dùng token thay vì password

❌ LỖI 5: App bị timeout khi chạy lâu
   ✅ FIX:
   - Tối ưu code chạy trong callback
   - Dùng st.spinner() để show loading


════════════════════════════════════════════════════════════════════════════
✅ BƯỚC 5: BEST PRACTICES DEPLOYMENT
════════════════════════════════════════════════════════════════════════════

1️⃣ Security:
   - KHÔNG commit credentials/API keys (dùng secrets management)
   - KHÔNG push model files quá lớn (dùng cloud storage)
   - Dùng .gitignore đúng cách

2️⃣ Performance:
   - Cache models & data với @st.cache_resource
   - Optimize imports (import trong function nếu cần)
   - Dùng lazy loading nếu cần nhiều thời gian

3️⃣ Monitoring:
   - Kiểm tra Streamlit Cloud logs định kỳ
   - Monitor app usage & performance
   - Set up alerts cho errors

4️⃣ Versioning:
   - Dùng Git tags cho releases
   - Giữ commit history rõ ràng
   - Docstring + comments chi tiết

5️⃣ CI/CD:
   - Dùng GitHub Actions để automated testing
   - Run tests trước khi merge
   - Maintain code quality


════════════════════════════════════════════════════════════════════════════
📚 BƯỚC 6: ADVANCED OPTIONS
════════════════════════════════════════════════════════════════════════════

1️⃣ Custom Domain:
   - Settings → Custom domain
   - Trỏ DNS của domain vào Streamlit Cloud
   - Ví dụ: churn.mycompany.com

2️⃣ Private Apps:
   - Settings → Sharing
   - Turn on authentication
   - Chỉ authorized users mới access

3️⃣ Secrets Management:
   - Tạo .streamlit/secrets.toml
   - Store API keys, DB passwords, etc.
   - Streamlit Cloud tự load từ settings

   Ví dụ .streamlit/secrets.toml:
   ──────────────────────────────────────────
   [database]
   user = "your_db_user"
   password = "your_db_password"
   
   [api]
   key = "your_api_key"
   ──────────────────────────────────────────

4️⃣ Schedule Runs:
   - Streamlit doesn't support background jobs
   - Sử dụng GitHub Actions + API calls

5️⃣ Scale Up:
   - Streamlit Cloud free tier có limitations
   - Nếu cần high traffic → AWS/GCP deployment


════════════════════════════════════════════════════════════════════════════
🚀 BƯỚC 7: SHARE & MONITOR
════════════════════════════════════════════════════════════════════════════

1️⃣ Share URL:
   - Copy app URL
   - Share với users
   - Ví dụ: https://share.streamlit.io/username/repo/main/app.py

2️⃣ Monitor Performance:
   - Streamlit Cloud dashboard
   - Check: Runs, Errors, Memory usage
   - Performance metrics

3️⃣ Get Feedback:
   - Add feedback widget
   - Monitor user interactions
   - Iterate based on feedback

4️⃣ Update App:
   - Git push → Streamlit auto-redeploys
   - Chỉ cần commit & push
   - Update live trong ~1 phút


════════════════════════════════════════════════════════════════════════════
✅ CHECKLIST - PRE-DEPLOYMENT
════════════════════════════════════════════════════════════════════════════

□ requirements.txt created & all dependencies listed
□ streamlit_app.py tested locally
□ No hardcoded paths (use Path, relative paths)
□ .gitignore configured correctly
□ All model files available & correct path
□ No secrets/credentials in code
□ README.md written
□ Code commented & documented
□ Local testing passed
□ GitHub repo created & initialized
□ Files pushed to GitHub
□ GitHub repo is PUBLIC
□ Streamlit Cloud account created
□ Deployment successful
□ App URL working & accessible
□ Share with stakeholders


════════════════════════════════════════════════════════════════════════════
🎉 SUCCESS - YOUR APP IS LIVE!
════════════════════════════════════════════════════════════════════════════

Các resources hữu ích:
- Streamlit Docs: https://docs.streamlit.io
- Streamlit Cloud: https://streamlit.io/cloud
- Streamlit Forum: https://discuss.streamlit.io
- GitHub: https://github.com

Enjoy! 🚀
"""

print(deployment_guide)

# ============================================================================
# CREATE REQUIREMENTS.TXT TEMPLATE
# ============================================================================

requirements_template = """streamlit==1.28.0
pandas==2.1.0
numpy==1.24.0
scikit-learn==1.3.0
joblib==1.3.0
"""

# ============================================================================
# CREATE README.MD TEMPLATE
# ============================================================================

readme_template = """# Customer Churn Prediction

## 📊 Project Description

Ứng dụng Machine Learning dự đoán khách hàng sẽ rời đi để công ty có biện pháp giữ chân kịp thời.

**Live Demo:** [https://share.streamlit.io/your-username/customer-churn-prediction/main/streamlit_app.py](https://share.streamlit.io/)

## 🎯 Features

- 🔮 Real-time churn prediction
- 📊 Customer analytics dashboard
- 💡 Actionable recommendations
- 🎨 User-friendly web interface

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **ML Framework:** scikit-learn
- **Data Processing:** pandas, numpy
- **Serialization:** joblib

## 📋 Dataset

- **Samples:** 450 customers
- **Features:** 10 (numerical + categorical)
- **Target:** Binary (Churn: Yes/No)
- **Churn Rate:** ~50%

## 🚀 Quick Start

### Local Installation

```bash
# Clone repository
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run src/streamlit_app.py
```

### Online (Streamlit Cloud)

Visit: [https://share.streamlit.io/your-username/customer-churn-prediction/main/streamlit_app.py](https://share.streamlit.io/)

## 📁 Project Structure

```
customer-churn-prediction/
├── data/
│   └── customer_churn_data.csv
├── models/
│   ├── random_forest.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   └── feature_info.json
├── src/
│   ├── streamlit_app.py
│   ├── cli_app.py
│   ├── save_load_models.py
│   └── ...
├── tests/
│   └── test_churn.py
├── notebooks/
│   └── (visualization output)
├── requirements.txt
├── README.md
└── .gitignore
```

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 82.1% |
| F1-Score | 80.5% |
| ROC-AUC | 88.3% |
| Precision | 81.2% |

## 🎯 How to Use

1. **Input Customer Info:** Nhập thông tin khách hàng
2. **Click Predict:** Click nút "Predict Churn"
3. **See Results:** Xem kết quả dự đoán & khuyến nghị
4. **Take Action:** Thực hiện hành động phù hợp

## 💡 Interpretation

- 🟢 **STAYED:** Khách hàng sẽ ở lại
- 🔴 **CHURNED:** Khách hàng sẽ rời đi
- **Confidence:** Độ tin cậy của dự đoán (0-100%)

## 🧪 Testing

```bash
# Install pytest
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📝 Features Explained

| Feature | Range | Description |
|---------|-------|-------------|
| Age | 18-80 | Tuổi khách hàng |
| Tenure | 0-120 (months) | Thời gian là khách |
| Monthly Charges | 10-150 ($) | Phí hàng tháng |
| Total Charges | 0-10000 ($) | Tổng tiền chi trả |
| Support Tickets | 0-10 | Số lần liên hệ hỗ trợ |
| Satisfaction | 1-5 | Mức độ hài lòng |
| Contract | 3 types | Loại hợp đồng |
| Internet Service | 3 types | Loại dịch vụ |
| Payment Method | 4 types | Phương thức thanh toán |

## 🔍 Model Details

**Algorithm:** Random Forest Classifier
- **Trees:** 100
- **Max Depth:** 10
- **Min Samples Split:** 10

## 📚 Files Included

- `generate_data.py` - Generate synthetic dataset
- `model_training.py` - Train & evaluate models
- `save_load_models.py` - Model serialization
- `cli_app.py` - Command-line interface
- `streamlit_app.py` - Web app (Streamlit)
- `hyperparameter_tuning.py` - GridSearchCV optimization
- `test_churn.py` - Unit tests

## 🤝 Contributing

Contributions welcome! Please fork & submit pull request.

## 📄 License

MIT License - feel free to use for educational & commercial purposes.

## 👨‍💻 Author

**Your Name**
- Email: your.email@company.com
- GitHub: [@your-username](https://github.com/your-username)

## 🙏 Acknowledgments

- Dataset inspired by Telco Customer Churn
- Built with Streamlit & scikit-learn
- Thanks to all contributors!

---

⭐ If you find this helpful, please star the repository!
"""

# ============================================================================
# CREATE GITIGNORE TEMPLATE
# ============================================================================

gitignore_template = """# Virtual Environment
venv/
env/
ENV/
.venv

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Environment variables
.env
.env.local
.env.*.local

# Streamlit
.streamlit/secrets.toml

# Data files
data/raw/
*.csv.bak

# Models (if large)
# models/*.pkl

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Testing
.pytest_cache/
.coverage
htmlcov/

# Logging
*.log
logs/

# OS
.DS_Store
Thumbs.db
"""

# ============================================================================
# WRITE FILES
# ============================================================================

from pathlib import Path

project_root = Path(__file__).parent.parent

# Write requirements.txt
with open(project_root / 'requirements.txt', 'w') as f:
    f.write(requirements_template)

print("✅ requirements.txt created")

# Write README.md
with open(project_root / 'README.md', 'w') as f:
    f.write(readme_template)

print("✅ README.md created")

# Write .gitignore
with open(project_root / '.gitignore', 'w') as f:
    f.write(gitignore_template)

print("✅ .gitignore created")

# ============================================================================
# SUMMARY
# ============================================================================

summary = f"""
{deployment_guide}

════════════════════════════════════════════════════════════════════════════
✅ FILES CREATED FOR DEPLOYMENT:
════════════════════════════════════════════════════════════════════════════

📄 requirements.txt   ✓ (thư viện dependencies)
📄 README.md          ✓ (mô tả project)
📄 .gitignore         ✓ (ignore files)

════════════════════════════════════════════════════════════════════════════
🚀 NEXT STEPS:
════════════════════════════════════════════════════════════════════════════

1. Init Git repo (nếu chưa có):
   git init

2. Add files:
   git add .

3. Commit:
   git commit -m "Add deployment files"

4. Create GitHub repo & push:
   git remote add origin https://github.com/YOUR_USERNAME/customer-churn-prediction.git
   git branch -M main
   git push -u origin main

5. Deploy lên Streamlit Cloud:
   - Vào https://streamlit.io/cloud
   - Click "New app"
   - Chọn repo & branch
   - Chọn file: src/streamlit_app.py
   - Click "Deploy"

6. Share URL:
   - Copy từ Streamlit Cloud dashboard
   - Share với stakeholders!

════════════════════════════════════════════════════════════════════════════
✅ BƯỚC 6 HOÀN TẤT - DEPLOYMENT GUIDE
════════════════════════════════════════════════════════════════════════════
"""

print(summary)
