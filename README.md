# 💰 Logistic Regression Analysis for Predicting Personal Loan Acceptance

## 🧠 Overview

This project analyzes customer demographic, financial, and behavioral data to predict personal loan acceptance using **Logistic Regression**. The study leverages a dataset of 480 customers and demonstrates a high-performing binary classification model, achieving **99% accuracy** and strong precision values (97% for accepted, 100% for rejected loans). The model identifies key predictors like securities account ownership, online banking usage, and credit card spending (CCAvg), offering actionable insights for financial institutions to improve loan targeting strategies.

---

## 🎯 Objectives

- Predict personal loan acceptance based on customer profile data.
- Evaluate the importance of financial, behavioral, and demographic features.
- Provide recommendations for better customer targeting in financial services.

---

## 📦 Dataset Description

- **Total Samples**: 480
- **Target Variable**: `Personal Loan` (Binary: 1 = Accepted, 0 = Rejected)
- **Feature Categories**:
  - **Demographic**: Age, Education, Family Size
  - **Financial**: Income, Mortgage, CCAvg
  - **Behavioral**: Online Banking, Securities Account, Credit Card Usage
  - **Dropped Features**: ID, ZIP Code

---

## 🛠 Preprocessing Steps

- ✅ Removed non-predictive columns (ID, ZIP Code)
- ✅ No missing values detected
- ✅ Applied `StandardScaler` to numerical features (e.g., Income, CCAvg)
- ✅ Used stratified train-test split (80% train, 20% test)

---

## 📈 Exploratory Data Analysis (EDA)

- Class distribution: 41% accepted, 59% rejected
- **Positive Correlations**:
  - Income and CCAvg → Higher likelihood of loan acceptance
  - Securities and online banking users → More likely to accept loans
- Box plots and correlation heatmaps were used to visualize key relationships

---

## 🤖 Model: Logistic Regression

### ⚙️ Model Configuration
- Algorithm: Logistic Regression (`scikit-learn`)
- Evaluation Metrics:
  - Accuracy
  - Precision
  - Confusion Matrix

### ✅ Model Performance
| Metric            | Result         |
|------------------|----------------|
| Accuracy          | **99%**        |
| Precision (Class 1) | 97%           |
| Precision (Class 0) | **100%**       |
| Misclassifications | Only 1         |

### 🔍 Feature Importance (Top Predictors)
| Feature             | Coefficient |
|---------------------|-------------|
| Securities Account  | +0.24       |
| Online Banking      | +0.22       |
| Credit Card Spending (CCAvg) | +0.19   |

---

## 📊 Visualizations

- 🎯 Confusion Matrix
- 📦 Box Plots: CCAvg vs Loan Acceptance
- 🔥 Correlation Heatmap
- 📉 Logistic Regression Coefficient Plot

---

## 💡 Key Insights

- High-income customers with strong credit card usage and investment behavior (e.g., securities account) are more likely to accept loans.
- Online banking adoption correlates with higher engagement and acceptance.
- Age and family size showed minimal predictive value compared to financial behavior.

---

## ⚠️ Limitations

- Dataset limited to **480 samples**—small sample size may limit generalization.
- Mild class imbalance (41% vs 59%) could affect model calibration in production environments.
- Logistic Regression assumes linearity—future studies could compare with tree-based or ensemble methods.

---

## 📌 Recommendations

- 🧑‍💼 Target high-income individuals with securities accounts and strong credit card activity.
- 💻 Promote online banking adoption as an engagement strategy.
- 🎯 Develop segmented marketing campaigns based on behavioral indicators.

---

## 📚 References

- Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research.
- Waskom, M. (2021). *Seaborn: Statistical Data Visualization*. Journal of Open Source Software.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
- APA Manual. (2020). *Publication Manual of the American Psychological Association* (7th ed.).

---

## 🧠 Author  
**Mohammed Saif Wasay**  
*Data Analytics Graduate — Northeastern University*  
*Machine Learning Enthusiast | Passionate about turning data into insights*

🔗 [Connect with me on LinkedIn](https://www.linkedin.com/in/mohammed-saif-wasay-4b3b64199/)

---
