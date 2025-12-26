"""
BƯỚC 7: ACADEMIC REPORT TEMPLATE
Hướng dẫn viết báo cáo khoa học cho project ML
"""

report_template = """
╔════════════════════════════════════════════════════════════════════════════╗
║ BƯỚC 7: ACADEMIC REPORT TEMPLATE - CUSTOMER CHURN PREDICTION              ║
╚════════════════════════════════════════════════════════════════════════════╝

════════════════════════════════════════════════════════════════════════════
📚 CẤUTRÚC BÁO CÁO KHOA HỌC
════════════════════════════════════════════════════════════════════════════

1. 📄 Title Page
2. 📋 Abstract (Tóm tắt)
3. 📑 Table of Contents
4. 📖 Introduction
5. 📚 Literature Review
6. 🔬 Methodology
7. 📊 Results
8. 💬 Discussion
9. 📌 Conclusion
10. 📚 References


════════════════════════════════════════════════════════════════════════════
🔶 1. TITLE PAGE
════════════════════════════════════════════════════════════════════════════

CUSTOMER CHURN PREDICTION USING MACHINE LEARNING

[Your Full Name]
[Your ID/Student Number]
[University Name]
[Department/Faculty]

Submitted to: [Professor Name]
Course: [Course Code - Course Name]
Date: [Today's Date]

Academic Year: [Year]
"""

abstract_template = """
════════════════════════════════════════════════════════════════════════════
🔶 2. ABSTRACT / TÓM TẮT
════════════════════════════════════════════════════════════════════════════

📝 TEMPLATE (150-250 words):
────────────────────────────────────────────────────────────────────────────

This paper presents a machine learning approach to predict customer churn in 
telecommunications industry. We developed and compared three classification 
models (Logistic Regression, Decision Tree, and Random Forest) using a dataset 
of 450 customer records with 10 features. The Random Forest model achieved the 
best performance with 82.1% accuracy, 81.2% precision, and 88.3% ROC-AUC score. 
Our findings demonstrate that factors such as tenure, monthly charges, and 
satisfaction level are key indicators of churn behavior. The proposed model can 
assist telecommunications companies in identifying at-risk customers and 
implementing targeted retention strategies. Future work will explore deep 
learning approaches and real-world deployment optimization.

Keywords: Machine Learning, Customer Churn, Classification, Random Forest, 
Telecommunications

════════════════════════════════════════════════════════════════════════════
"""

introduction_template = """
════════════════════════════════════════════════════════════════════════════
🔶 3. INTRODUCTION / MỞ ĐẦU
════════════════════════════════════════════════════════════════════════════

📝 CÁC PHẦN CHÍNH:

1️⃣ BACKGROUND (Bối cảnh chung):
   - Vấn đề kinh doanh: Tại sao dự đoán churn quan trọng?
   - Ví dụ: "Trong ngành viễn thông, khách hàng rời đi (churn) là 
     vấn đề lớn. Mỗi năm, công ty mất 5-10% khách hàng..."
   
2️⃣ PROBLEM STATEMENT (Tuyên bố vấn đề):
   - Cụ thể hóa vấn đề: "Cần phát triển mô hình dự đoán churn 
     với độ chính xác cao để công ty chủ động giữ chân khách hàng"
   
3️⃣ MOTIVATION (Động lực):
   - Tại sao lại chọn machine learning?
   - Lợi ích: "ML có thể tìm ra patterns phức tạp mà con người khó phát hiện"
   
4️⃣ OBJECTIVES (Mục tiêu):
   - Mục tiêu chính: Xây dựng mô hình dự đoán churn
   - Mục tiêu cụ thể:
     * So sánh 3 algorithms
     * Tối ưu hyperparameters
     * Đạt ≥80% accuracy
     * Deploy lên production

5️⃣ CONTRIBUTIONS (Đóng góp):
   - CWhat we did: "Chúng tôi xây dựng dataset 450 mẫu, huấn luyện 3 mô hình"
   - What's new: "So sánh chi tiết 3 algorithms với 6 evaluation metrics"

📋 EXAMPLE TEXT:

────────────────────────────────────────────────────────────────────────────
1. INTRODUCTION

1.1 Background
Customer churn (khách hàng rời đi) is a critical challenge in the 
telecommunications industry. According to industry reports, companies lose 
3-5% of customers annually to competitors. Retaining customers is 5x cheaper 
than acquiring new ones (Reichheld & Schefter, 2000). Therefore, predicting 
which customers are likely to leave is essential for business survival.

1.2 Problem Statement
Current approaches rely on manual customer segmentation and reactive response 
strategies. There is a need for a data-driven, predictive approach that can 
identify at-risk customers before they churn, enabling proactive retention 
campaigns.

1.3 Motivation
Machine learning can uncover hidden patterns in customer behavior that 
traditional methods miss. By analyzing historical customer data, ML models can 
predict future churn with high accuracy and provide actionable insights.

1.4 Objectives
This study aims to:
  1. Develop and train three classification models
  2. Compare their performance using multiple evaluation metrics
  3. Identify the most important features predicting churn
  4. Deploy the best model for real-world prediction

1.5 Contributions
Our contributions include:
  1. Creation of a comprehensive customer churn dataset
  2. Systematic comparison of three algorithms
  3. Hyperparameter optimization using GridSearchCV
  4. A production-ready web application for predictions
────────────────────────────────────────────────────────────────────────────
"""

literature_template = """
════════════════════════════════════════════════════════════════════════════
🔶 4. LITERATURE REVIEW / ĐỀ CỬU NHÂN CỨU
════════════════════════════════════════════════════════════════════════════

📝 CẤU TRÚC:

1️⃣ CUSTOMER CHURN ANALYSIS (Phân tích churn):
   - Định nghĩa churn
   - Tác động kinh doanh
   - Chiến lược giữ chân khách

2️⃣ MACHINE LEARNING FOR CLASSIFICATION:
   - Decision Trees: Cây quyết định
   - Logistic Regression: Mô hình hồi quy logistic
   - Random Forest: Rừng ngẫu nhiên

3️⃣ FEATURE ENGINEERING:
   - Lựa chọn features
   - Preprocessing
   - Feature scaling

4️⃣ MODEL EVALUATION:
   - Accuracy, Precision, Recall
   - F1-Score, ROC-AUC
   - Cross-validation

5️⃣ RELATED WORK:
   - Các research trước đây
   - Comparison với bài này

📋 EXAMPLE TEXT:

────────────────────────────────────────────────────────────────────────────
2. LITERATURE REVIEW

2.1 Customer Churn Prediction
Customer churn is defined as "the voluntary abandonment of a company's products 
or services" (Neslin et al., 2006). In telecommunications, churn rate typically 
ranges from 3-5% monthly. Chen et al. (2012) showed that predicting churn can 
reduce customer loss by 30-50%.

2.2 Classification Algorithms

2.2.1 Logistic Regression
Despite its simplicity, logistic regression remains a baseline for binary 
classification (James et al., 2013). It works well when features have linear 
relationship with target.

2.2.2 Decision Trees
Decision trees are interpretable and handle non-linear relationships. However, 
they are prone to overfitting (Breiman, 1984).

2.2.3 Random Forest
Random Forest combines multiple decision trees, reducing overfitting through 
averaging (Breiman, 2001). It typically outperforms individual trees.

2.3 Evaluation Metrics
  - Accuracy: Tỷ lệ dự đoán đúng
  - Precision: Độ chính xác của lớp dương
  - Recall: Khả năng phát hiện lớp dương
  - F1-Score: Trung bình hài hòa của Precision & Recall
  - ROC-AUC: Diện tích dưới đường cong ROC

2.4 Related Work
Smith et al. (2015) compared SVM and Random Forest on telecom churn, finding 
RF superior with 85% accuracy. Kumar et al. (2018) used deep learning and 
achieved 89% accuracy but with less interpretability.

Our work extends these studies by: (1) systematic comparison of three algorithms,
(2) comprehensive evaluation with 6 metrics, (3) production deployment.
────────────────────────────────────────────────────────────────────────────
"""

methodology_template = """
════════════════════════════════════════════════════════════════════════════
🔶 5. METHODOLOGY / PHƯƠNG PHÁP
════════════════════════════════════════════════════════════════════════════

📝 CỰC PHẦN:

1️⃣ DATA COLLECTION & PREPARATION:
   - Dataset mô tả
   - Số lượng samples
   - Features & target
   - Data quality

2️⃣ EXPLORATORY DATA ANALYSIS (EDA):
   - Distribution analysis
   - Correlation analysis
   - Missing values
   - Outliers

3️⃣ PREPROCESSING:
   - Data cleaning
   - Categorical encoding
   - Feature scaling
   - Train/test split

4️⃣ MODEL DEVELOPMENT:
   - Algorithms: LR, DT, RF
   - Hyperparameters
   - Training procedure

5️⃣ MODEL EVALUATION:
   - Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
   - Cross-validation
   - Comparison

📋 EXAMPLE TEXT:

────────────────────────────────────────────────────────────────────────────
3. METHODOLOGY

3.1 Data Collection & Preparation
We created a synthetic dataset of 450 customer records from a telecommunications
company, with 10 features and 1 target variable (Churn: Yes/No).

Dataset Characteristics:
  - Total samples: 450
  - Features: 10 (7 numerical, 3 categorical)
  - Target: Binary (0=Stayed, 1=Churned)
  - Churn rate: 50% (balanced dataset)
  - Missing values: 0
  - Outliers: 3 (mild, retained for analysis)

3.2 Exploratory Data Analysis
EDA revealed:
  - Age distribution: Normal (μ=45, σ=15)
  - Tenure shows strong negative correlation with churn (r=-0.65)
  - Satisfaction inversely related to churn
  - Monthly charges slight positive correlation with churn

3.3 Data Preprocessing
  Step 1: Separate features and target
  Step 2: Encode categorical variables (LabelEncoder)
  Step 3: Scale numerical features (StandardScaler)
  Step 4: 80/20 train/test split with stratification

3.4 Model Development
We trained three models:
  1. Logistic Regression (max_iter=1000)
  2. Decision Tree (max_depth=7)
  3. Random Forest (n_estimators=100, max_depth=10)

3.5 Model Evaluation
Performance metrics:
  - Accuracy: correct predictions / total predictions
  - Precision: TP / (TP + FP)
  - Recall: TP / (TP + FN)
  - F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
  - ROC-AUC: Area under ROC curve
  
5-fold cross-validation used to ensure robustness.
────────────────────────────────────────────────────────────────────────────
"""

results_template = """
════════════════════════════════════════════════════════════════════════════
🔶 6. RESULTS / KẾT QUẢ
════════════════════════════════════════════════════════════════════════════

📝 INCLUDE:

1️⃣ PERFORMANCE METRICS TABLE:
   Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC
   ──────────────────────────────────────────────────────────

2️⃣ VISUALIZATIONS:
   - Confusion matrices
   - ROC curves
   - Feature importance plot
   - Learning curves

3️⃣ FEATURE IMPORTANCE:
   - Top 5 most important features
   - How much each contributes

4️⃣ CROSS-VALIDATION RESULTS:
   - Mean CV score
   - Standard deviation
   - Consistency check

📋 EXAMPLE TEXT & TABLE:

────────────────────────────────────────────────────────────────────────────
4. RESULTS

4.1 Model Performance
Table 1 summarizes the performance metrics for all three models on the test set:

Table 1. Model Performance Comparison
┌──────────────────┬──────────┬───────────┬────────┬──────────┬─────────┐
│ Model            │ Accuracy │ Precision │ Recall │ F1-Score │ ROC-AUC │
├──────────────────┼──────────┼───────────┼────────┼──────────┼─────────┤
│ Logistic Regr.   │ 78.2%    │ 76.5%     │ 80.1%  │ 78.3%    │ 84.2%   │
│ Decision Tree    │ 81.0%    │ 79.8%     │ 82.5%  │ 81.1%    │ 86.5%   │
│ Random Forest    │ 82.1%    │ 81.2%     │ 83.4%  │ 82.3%    │ 88.3%   │
└──────────────────┴──────────┴───────────┴────────┴──────────┴─────────┘

Random Forest achieved the best performance with 82.1% accuracy and 88.3% 
ROC-AUC, indicating excellent discriminative ability.

4.2 Feature Importance
Figure 2 shows the top 5 features predicting churn:
  1. Tenure (25.3%) - Time as customer
  2. Monthly Charges (18.7%) - Monthly fee
  3. Satisfaction (16.5%) - Customer satisfaction
  4. Support Tickets (14.2%) - Contact frequency
  5. Total Charges (12.1%) - Cumulative charges

These features explain 86.8% of the model's predictions.

4.3 Cross-Validation Results
5-fold cross-validation on Random Forest yielded:
  - Mean CV Score: 81.5% (±2.1%)
  - Standard deviation: 2.1% (very consistent)
  - Min score: 79.2%, Max score: 84.0%

The low standard deviation indicates the model generalizes well to unseen data.

4.4 Confusion Matrix Analysis
[Confusion Matrix for Random Forest]
                    Predicted
                    Stayed  Churned
Actual  Stayed      68      12
        Churned     8       62

True Positive Rate: 88.6% (khả năng phát hiện churn)
True Negative Rate: 85.0% (khả năng phát hiện stayed)
────────────────────────────────────────────────────────────────────────────
"""

discussion_template = """
════════════════════════════════════════════════════════════════════════════
🔶 7. DISCUSSION / THẢO LUẬN
════════════════════════════════════════════════════════════════════════════

📝 PHẦN CHÍNH:

1️⃣ KEY FINDINGS:
   - Mô hình tốt nhất là gì?
   - Tại sao nó tốt hơn?

2️⃣ INTERPRETATION:
   - Ý nghĩa của kết quả
   - So sánh với literature

3️⃣ IMPLICATIONS:
   - Ứng dụng thực tế
   - Tác động kinh doanh

4️⃣ LIMITATIONS:
   - Hạn chế của nghiên cứu
   - Dataset limitations
   - Model limitations

5️⃣ FUTURE WORK:
   - Cải tiến
   - Nghiên cứu tiếp theo

📋 EXAMPLE TEXT:

────────────────────────────────────────────────────────────────────────────
5. DISCUSSION

5.1 Key Findings
Our results demonstrate that Random Forest outperforms both Logistic Regression
and Decision Tree by 3.9% and 1.1% respectively in accuracy. This aligns with 
prior research showing Random Forest's superiority in non-linear classification 
tasks (Breiman, 2001; Chen et al., 2012).

5.2 Interpretation
Tenure is the strongest predictor (25.3%), confirming industry intuition: 
"long-term customers are loyal." The second-strongest predictor, Monthly Charges
(18.7%), suggests customers dissatisfied with pricing are likely to churn. 
Satisfaction ranking third (16.5%) indicates emotional loyalty matters.

5.3 Business Implications
With 82.1% accuracy, this model can:
  1. Identify 83.4% of at-risk customers (Recall)
  2. With 81.2% certainty (Precision)
  3. Enable targeted retention campaigns
  4. Reduce acquisition cost by 30-50% (Neslin et al., 2006)

For a company with 100,000 customers churning 4% annually:
  - Without model: 4,000 lost customers
  - With model (83.4% detection): 3,336 saved = $1.67M value
  
5.4 Limitations
  1. Dataset is synthetic; real data may differ
  2. Temporal factors not considered (seasonality)
  3. External factors (competition) not included
  4. Model assumes class balance (50% churn unrealistic)
  5. No demographic diversity in data

5.5 Future Work
  1. Collect real customer data
  2. Incorporate time-series features
  3. Explore deep learning (LSTM, Neural Networks)
  4. A/B testing retention campaigns
  5. Real-time model updates
────────────────────────────────────────────────────────────────────────────
"""

conclusion_template = """
════════════════════════════════════════════════════════════════════════════
🔶 8. CONCLUSION / KẾT LUẬN
════════════════════════════════════════════════════════════════════════════

📝 CẤU TRÚC:

1️⃣ SUMMARY:
   - Tóm tắt lại những gì đã làm

2️⃣ ANSWER TO RESEARCH QUESTIONS:
   - Trả lời câu hỏi từ introduction

3️⃣ MAIN CONTRIBUTIONS:
   - Đóng góp chính

4️⃣ PRACTICAL IMPACT:
   - Tác động thực tế

5️⃣ RECOMMENDATION:
   - Khuyến cáo cho tiếp theo

📋 EXAMPLE TEXT:

────────────────────────────────────────────────────────────────────────────
6. CONCLUSION

This study developed and compared three machine learning models for customer 
churn prediction. Random Forest emerged as the best performer with 82.1% 
accuracy and 88.3% ROC-AUC score, successfully identifying 83.4% of at-risk 
customers.

Our key findings are:
  1. Random Forest outperforms simpler algorithms
  2. Tenure, charges, and satisfaction are key predictors
  3. The model generalizes well (2.1% CV standard deviation)
  4. Real-world deployment is feasible

Business Impact:
For a telecomunications company, this model can save $1.67M annually by 
enabling targeted retention campaigns. The 81.2% precision ensures marketing 
resources focus on high-risk customers.

Recommendations:
  1. Deploy the model to production (Streamlit Cloud)
  2. Monitor predictions monthly
  3. Collect real customer data for model refinement
  4. A/B test retention strategies on identified segments
  5. Explore advanced approaches (deep learning, ensemble methods)

The proposed solution demonstrates practical value for the telecommunications
industry and can be adapted to other customer-centric domains (banking, SaaS, 
retail).

════════════════════════════════════════════════════════════════════════════
"""

references_template = """
════════════════════════════════════════════════════════════════════════════
🔶 9. REFERENCES / TÀI LIỆU THAM KHẢO
════════════════════════════════════════════════════════════════════════════

Breiman, L. (1984). Classification and regression trees. Chapman and Hall.

Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.

Chen, Y., Tan, K. L., & Teh, Y. W. (2012). Customer churn prediction in 
telecommunications: A stratified sampling and calibration approach. IEEE 
Transactions on Knowledge and Data Engineering, 24(8), 1556-1568.

James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). An introduction to 
statistical learning (Vol. 112). Springer Science+Business Media.

Neslin, S. A., Gupta, S., Kamakura, W., Lu, J., & Sun, B. (2006). Defection 
detection: Measuring and understanding the predictability of customer churn. 
Journal of Marketing Research, 43(2), 204-211.

Reichheld, F. F., & Schefter, P. (2000). E-loyalty: Your secret weapon on the 
Web. Harvard Business Review, 78(4), 105-113.

Scikit-learn Developers (2023). Scikit-learn: Machine learning in Python. 
Retrieved from https://scikit-learn.org/

Smith, J., Lee, P., & Zhang, Q. (2015). Comparative analysis of machine learning
methods for customer churn prediction. Proceedings of ICML, 45, 234-245.

════════════════════════════════════════════════════════════════════════════
"""

# Print the full template
full_template = f"""{report_template}
{abstract_template}
{introduction_template}
{literature_template}
{methodology_template}
{results_template}
{discussion_template}
{conclusion_template}
{references_template}

════════════════════════════════════════════════════════════════════════════
✅ FORMATTING GUIDELINES
════════════════════════════════════════════════════════════════════════════

📝 Font & Style:
   - Font: Times New Roman or Arial
   - Size: 12pt (body), 14pt (headings)
   - Line spacing: 1.5 or 2.0
   - Margins: 1 inch (2.54cm) all sides

🔢 Numbering:
   - Sections: 1, 2, 3... (or 1.1, 1.2...)
   - Figures/Tables: Figure 1, Table 1
   - References: [1], [2] or (Author, Year)

📊 Figures & Tables:
   - Caption above/below
   - Reference in text
   - High resolution (300 dpi for print)
   - Clear labels & legends

📝 Writing Style:
   - Third person: "The model was trained..." (NOT "I trained...")
   - Past tense: "We collected data..." (NOT "We are collecting...")
   - Active voice (preferred)
   - Concise & clear language

════════════════════════════════════════════════════════════════════════════
📏 WORD COUNT GUIDELINES
════════════════════════════════════════════════════════════════════════════

Typical Structure:
   - Title: N/A
   - Abstract: 150-250 words
   - Introduction: 300-500 words
   - Literature Review: 500-800 words
   - Methodology: 400-700 words
   - Results: 300-500 words
   - Discussion: 400-700 words
   - Conclusion: 200-300 words
   - References: 10-20 sources
   ─────────────────────────────
   TOTAL: 2,500-4,500 words

════════════════════════════════════════════════════════════════════════════
✅ CHECKLIST
════════════════════════════════════════════════════════════════════════════

□ Title page with all required info
□ Abstract (150-250 words)
□ Table of contents
□ Introduction with clear objectives
□ Literature review with proper citations
□ Methodology described in detail
□ Results with tables/figures
□ Discussion interpreting findings
□ Conclusion answering research questions
□ References (15+ sources)
□ All figures/tables numbered & captioned
□ Consistent formatting (font, spacing, margins)
□ Spellcheck & grammar check
□ No plagiarism (proper citations)
□ PDF conversion test
□ Print preview (if submitting hard copy)
□ Peer review by colleague
□ Final proofread

════════════════════════════════════════════════════════════════════════════
🎉 TIPS FOR EXCELLENCE
════════════════════════════════════════════════════════════════════════════

1️⃣ Read similar papers to understand style
2️⃣ Use LaTeX for technical writing (optional)
3️⃣ Create bibliography early (BibTeX, Zotero, Mendeley)
4️⃣ Write multiple drafts & iterate
5️⃣ Get feedback from advisors/peers
6️⃣ Use tables for numerical data
7️⃣ Use figures for visual insights
8️⃣ Proofread multiple times
9️⃣ Check university guidelines
🔟 Submit early for feedback

════════════════════════════════════════════════════════════════════════════
✅ BƯỚC 7 HOÀN TẤT - ACADEMIC REPORT TEMPLATE
════════════════════════════════════════════════════════════════════════════
"""

print(full_template)
