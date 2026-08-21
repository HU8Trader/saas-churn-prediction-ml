# SaaS Customer Intelligence: Predictive Churn Modeling, Survival Analysis & LTV Forecasting

> **Author**: **Himansh Upadhyay** | **GitHub**: [@HU8Trader](https://github.com/HU8Trader)


[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn%20%26%20XGBoost-orange.svg)](https://scikit-learn.org/)
[![Lifelines](https://img.shields.io/badge/Survival-Lifelines%20(Kaplan--Meier)-green.svg)](https://lifelines.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Python Data Science and Machine Learning repository analyzing $121.9M ARR in B2B enterprise SaaS telemetry across 500 customer accounts, 5,000 subscription cycles, 25,000 feature logs, and 2,000 support cases.

---

##  Key Data Science Deliverables

```
├── data/                    # 6 Raw Relational Tables + Master Customer 360 Feature Matrix
├── notebooks/               # Master Jupyter Notebook (.ipynb)
├── src/                     # Production Modular Python DS & ML Scripts
├── models/                  # Serialized ML Pipeline (XGBoost / Random Forest) & Metadata
├── figures/                 # High-Resolution Publication Charts (Survival, Cohorts, LTV)
└── outputs/                 # Regression Statistical Logs & JSON Metrics
```

---

##  Empirical Machine Learning & Statistical Findings

### 1. Multi-Model Churn Classification (5-Fold Stratified CV)
| Model Architecture | Test ROC-AUC | Test Accuracy | Brier Score | Hyperparameters |
| :--- | :---: | :---: | :---: | :--- |
| **XGBoost Classifier** | **0.6160** | **69.0%** | **0.1947** | `n_estimators=100`, `max_depth=3`, `lr=0.05` |
| **Random Forest** | **0.6096** | **67.0%** | **0.2121** | `n_estimators=150`, `max_depth=5`, `class_weight=balanced` |
| **Logistic Regression** | **0.5868** | **55.0%** | **0.2458** | `C=1.0`, `solver=lbfgs`, `penalty=l2` |

### 2. Top Empirical Churn Feature Drivers (Gini Importance)
1. **`error_rate_per_100_events` (7.39%)**: Product reliability failures directly accelerate customer departure.
2. **`avg_first_response_min` (6.89%)**: Support SLA responsiveness is the #2 leading indicator of retention.
3. **`tenure_months` (6.77%)**: Accounts past month 6 demonstrate significantly higher survival probabilities.
4. **`avg_session_duration_mins` (6.49%)**: Feature session depth reflects core workflow integration.
5. **`total_mrr` (5.63%)**: High-value accounts demonstrate distinct expansion/churn dynamics.

---

##  Visualizations & Artifacts

### Kaplan-Meier Customer Survival Curve & MoM Cohort Retention Heatmap
- **Median Customer Tenure**: `5.4 months` (Mean: `7.3 months`).
- **Triangular Retention Heatmap**: Month-over-month retention matrix from Month 0 to Month 23.
- **Cox Proportional Hazards**: Hazard ratios evaluating support latency and error rate multipliers ($p = 0.0016$).

---

##  Quick Start & Reproduction

```bash
# 1. Clone this repository
git clone https://github.com/your-username/saas-customer-intelligence-ml.git
cd saas-customer-intelligence-ml

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run full feature engineering and ML training
python src/feature_engineering.py
python src/churn_prediction_pipeline.py
python src/survival_and_cohort_analysis.py
python src/ltv_and_revenue_forecasting.py
python src/statistical_driver_analysis.py

# 4. Launch Master Jupyter Notebook
jupyter notebook notebooks/saas_executive_intelligence_analysis.ipynb
```
