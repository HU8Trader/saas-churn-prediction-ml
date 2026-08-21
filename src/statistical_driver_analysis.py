"""
Statistical Driver & Regression Analysis Pipeline
Performs OLS Multiple Regression for CSAT drivers, ANOVA tests for Churn Refunds,
Chi-Square tests for Plan Tier Churn, and correlation heatmaps.
"""

import pandas as pd
import numpy as np
import os
import json
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIGS_DIR = os.path.join(BASE_DIR, "figures")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

def run_statistical_driver_analysis():
    print("=" * 70, flush=True)
    print("SAAS STATISTICAL DRIVER & REGRESSION ANALYSIS PIPELINE", flush=True)
    print("=" * 70, flush=True)

    # 1. Load Data
    df_c360 = pd.read_csv(os.path.join(DATA_DIR, "customer_360_analytical.csv"))
    df_tickets = pd.read_csv(os.path.join(DATA_DIR, "fact_support_tickets.csv"))
    df_churn = pd.read_csv(os.path.join(DATA_DIR, "fact_churn_events.csv"))

    # 2. OLS Multiple Regression for Support CSAT Drivers
    print("\n" + "-" * 70, flush=True)
    print("1. OLS MULTIPLE REGRESSION: DRIVERS OF CUSTOMER SATISFACTION (CSAT)")
    print("-" * 70, flush=True)

    # Clean tickets with valid satisfaction scores (>0)
    ticket_ols_df = df_tickets[df_tickets['satisfaction_score'] > 0].copy()
    ticket_ols_df['is_escalated'] = ticket_ols_df['escalation_flag'].astype(int)
    ticket_ols_df['is_urgent'] = (ticket_ols_df['priority'] == 'Urgent').astype(int)
    ticket_ols_df['is_high'] = (ticket_ols_df['priority'] == 'High').astype(int)

    ols_formula = "satisfaction_score ~ first_response_time_minutes + resolution_time_hours + is_escalated + is_urgent + is_high"
    ols_model = smf.ols(ols_formula, data=ticket_ols_df).fit()

    ols_summary_str = str(ols_model.summary())
    print(ols_summary_str, flush=True)

    with open(os.path.join(OUT_DIR, "ols_csat_regression_summary.txt"), 'w') as f:
        f.write(ols_summary_str)

    # 3. One-Way ANOVA: Refund Amounts across Churn Reason Codes
    print("\n" + "-" * 70, flush=True)
    print("2. ONE-WAY ANOVA: REFUND AMOUNTS BY CHURN REASON CODE")
    print("-" * 70, flush=True)

    reasons = df_churn['reason_code'].unique()
    groups = [df_churn[df_churn['reason_code'] == r]['refund_amount_usd'].dropna() for r in reasons]
    f_stat, p_val_anova = stats.f_oneway(*groups)

    print(f"ANOVA F-statistic: {f_stat:.4f} | p-value: {p_val_anova:.4f}", flush=True)
    reason_stats = df_churn.groupby('reason_code')['refund_amount_usd'].agg(['count', 'mean', 'median', 'std', 'sum']).round(2)
    print(reason_stats.to_string(), flush=True)

    # 4. Chi-Square Test: Plan Tier vs Churn Flag
    print("\n" + "-" * 70, flush=True)
    print("3. CHI-SQUARE TEST: INDEPENDENCE BETWEEN PLAN TIER AND CHURN")
    print("-" * 70, flush=True)

    contingency_table = pd.crosstab(df_c360['plan_tier'], df_c360['churn_flag'])
    chi2, p_val_chi2, dof, expected = stats.chi2_contingency(contingency_table)

    print(f"Chi-Square Statistic: {chi2:.4f} | degrees of freedom: {dof} | p-value: {p_val_chi2:.4f}", flush=True)
    print("Observed Frequencies:\n", contingency_table, flush=True)

    # 5. Multivariate Correlation Matrix & Heatmap
    corr_cols = [
        'tenure_months', 'total_mrr', 'seats', 'error_rate_per_100_events',
        'total_duration_hours', 'avg_session_duration_mins', 'total_tickets',
        'avg_csat', 'avg_first_response_min', 'avg_resolution_hours', 'churn_flag'
    ]
    corr_matrix = df_c360[corr_cols].corr()

    corr_matrix.round(3).to_csv(os.path.join(DATA_DIR, "correlation_matrix.csv"))

    plt.figure(figsize=(12, 9))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        cbar_kws={'label': 'Pearson Correlation Coefficient r'}
    )
    plt.title('SaaS Multivariate Operational & Churn Correlation Matrix', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()

    heatmap_path = os.path.join(FIGS_DIR, "statistical_driver_heatmap.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"\nCorrelation heatmap saved to: {heatmap_path}", flush=True)

    # 6. Export JSON Summary
    summary = {
        'ols_csat_model': {
            'r_squared': round(float(ols_model.rsquared), 4),
            'r_squared_adj': round(float(ols_model.rsquared_adj), 4),
            'f_statistic': round(float(ols_model.fvalue), 4),
            'f_pvalue': float(ols_model.f_pvalue),
            'significant_features': [param for param, p in ols_model.pvalues.items() if p < 0.05]
        },
        'anova_churn_refunds': {
            'f_statistic': round(float(f_stat), 4),
            'p_value': round(float(p_val_anova), 4),
            'statistically_significant': bool(p_val_anova < 0.05)
        },
        'chisquare_tier_churn': {
            'chi2_statistic': round(float(chi2), 4),
            'p_value': round(float(p_val_chi2), 4),
            'dof': int(dof),
            'statistically_significant': bool(p_val_chi2 < 0.05)
        }
    }

    with open(os.path.join(OUT_DIR, "statistical_drivers_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\nStatistical Driver & Regression Analysis completed successfully!", flush=True)

if __name__ == "__main__":
    run_statistical_driver_analysis()
