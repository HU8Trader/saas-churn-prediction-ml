"""
Cohort & Survival Analysis Pipeline
Fits Kaplan-Meier Survival Curves, Cox Proportional Hazards regression,
and builds Month-over-Month Triangular Cohort Retention Matrix.
"""

import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIGS_DIR = os.path.join(BASE_DIR, "figures")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

def run_survival_cohort_analysis():
    print("=" * 70, flush=True)
    print("SAAS COHORT & SURVIVAL ANALYSIS PIPELINE", flush=True)
    print("=" * 70, flush=True)

    # 1. Load Data
    df = pd.read_csv(os.path.join(DATA_DIR, "customer_360_analytical.csv"))
    df_subs = pd.read_csv(os.path.join(DATA_DIR, "fact_subscriptions.csv"))
    df_accounts = pd.read_csv(os.path.join(DATA_DIR, "dim_accounts.csv"))

    # Duration: tenure_months, Event: churn_flag
    T = df['tenure_months']
    E = df['churn_flag']

    print(f"Total cohort size: {len(df)} accounts | Churn events observed: {E.sum()} ({(E.sum()/len(df))*100:.1f}%)", flush=True)
    print(f"Median customer tenure: {T.median():.1f} months (Mean: {T.mean():.1f} months)", flush=True)

    # 2. Overall Kaplan-Meier Survival Curve
    kmf_overall = KaplanMeierFitter()
    kmf_overall.fit(T, event_observed=E, label='All Accounts')

    # 3. Stratified Survival Analysis by Plan Tier
    kmf_enterprise = KaplanMeierFitter()
    kmf_pro = KaplanMeierFitter()
    kmf_basic = KaplanMeierFitter()

    ent_mask = df['plan_tier'] == 'Enterprise'
    pro_mask = df['plan_tier'] == 'Pro'
    basic_mask = df['plan_tier'] == 'Basic'

    kmf_enterprise.fit(T[ent_mask], E[ent_mask], label='Enterprise Tier')
    kmf_pro.fit(T[pro_mask], E[pro_mask], label='Pro Tier')
    kmf_basic.fit(T[basic_mask], E[basic_mask], label='Basic Tier')

    # Log-rank test between Enterprise and Basic
    lr_tier = logrank_test(T[ent_mask], T[basic_mask], E[ent_mask], E[basic_mask])
    print(f"\nLog-Rank Test (Enterprise vs Basic): p-value = {lr_tier.p_value:.4f} (test_stat: {lr_tier.test_statistic:.3f})", flush=True)

    # 4. Generate Survival Plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Overall KM Curve
    kmf_overall.plot_survival_function(ax=axes[0], color='#F59E0B', lw=2.5)
    axes[0].set_title('Overall Customer Survival Curve (Kaplan-Meier)', fontsize=13, fontweight='bold', pad=12)
    axes[0].set_xlabel('Tenure (Months)', fontsize=11)
    axes[0].set_ylabel('Customer Retention Probability S(t)', fontsize=11)
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Plot 2: Tier Breakdown
    kmf_enterprise.plot_survival_function(ax=axes[1], color='#10B981', lw=2)
    kmf_pro.plot_survival_function(ax=axes[1], color='#3B82F6', lw=2)
    kmf_basic.plot_survival_function(ax=axes[1], color='#EF4444', lw=2)
    axes[1].set_title('Survival Curves Stratified by Plan Tier', fontsize=13, fontweight='bold', pad=12)
    axes[1].set_xlabel('Tenure (Months)', fontsize=11)
    axes[1].set_ylabel('Retention Probability S(t)', fontsize=11)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(FIGS_DIR, "survival_curves.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Survival curve figures saved to: {plot_path}", flush=True)

    # 5. Cox Proportional Hazards Model
    cox_cols = [
        'tenure_months', 'churn_flag', 'seats', 'is_trial',
        'error_rate_per_100_events', 'avg_first_response_min', 'avg_csat',
        'annual_billing_ratio', 'net_expansion_score'
    ]
    df_cox = df[cox_cols].dropna()

    cph = CoxPHFitter()
    cph.fit(df_cox, duration_col='tenure_months', event_col='churn_flag')

    print("\n" + "-" * 70, flush=True)
    print("COX PROPORTIONAL HAZARDS REGRESSION (Hazard Ratios)")
    print("-" * 70, flush=True)
    cox_summary = cph.summary[['coef', 'exp(coef)', 'se(coef)', 'p']].copy()
    cox_summary.columns = ['Coefficient', 'Hazard_Ratio (exp)', 'Std_Error', 'p_value']
    print(cox_summary.to_string(), flush=True)

    cox_summary.to_csv(os.path.join(DATA_DIR, "cox_hazard_ratios.csv"))

    # 6. Triangular Month-over-Month Cohort Retention Matrix
    print("\n" + "-" * 70, flush=True)
    print("CALCULATING MONTH-OVER-MONTH COHORT RETENTION MATRIX")
    print("-" * 70, flush=True)

    # Assign signup cohort (YYYY-MM)
    df_accounts['signup_cohort'] = pd.to_datetime(df_accounts['signup_date']).dt.to_period('M')
    df_subs['sub_start_month'] = pd.to_datetime(df_subs['start_date']).dt.to_period('M')

    merged = df_subs.merge(df_accounts[['account_id', 'signup_cohort']], on='account_id')
    
    # Calculate cohort index (months since signup)
    merged['cohort_index'] = (merged['sub_start_month'].dt.year - merged['signup_cohort'].dt.year) * 12 + \
                             (merged['sub_start_month'].dt.month - merged['signup_cohort'].dt.month)
    
    # Filter non-negative indices
    merged = merged[merged['cohort_index'] >= 0]

    # Group by cohort and index
    cohort_data = merged.groupby(['signup_cohort', 'cohort_index'])['account_id'].nunique().reset_index()
    cohort_pivot = cohort_data.pivot(index='signup_cohort', columns='cohort_index', values='account_id')

    # Initial cohort size
    cohort_sizes = cohort_pivot.iloc[:, 0]
    retention_matrix = cohort_pivot.divide(cohort_sizes, axis=0) * 100.0

    cohort_out_path = os.path.join(DATA_DIR, "cohort_retention_matrix.csv")
    retention_matrix.round(1).to_csv(cohort_out_path)
    print(f"Cohort retention matrix saved to: {cohort_out_path}", flush=True)

    # 7. Plot Cohort Heatmap
    plt.figure(figsize=(14, 8))
    sns.heatmap(
        retention_matrix.round(1),
        annot=True,
        fmt=".0f",
        cmap="YlOrBr",
        vmin=0,
        vmax=100,
        cbar_kws={'label': 'Customer Retention Rate (%)'}
    )
    plt.title('MoM Customer Cohort Retention Heatmap (%)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Billing Months Since Signup (Cohort Index)', fontsize=11)
    plt.ylabel('Signup Cohort (Month)', fontsize=11)
    plt.tight_layout()
    
    heatmap_path = os.path.join(FIGS_DIR, "cohort_retention_heatmap.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"Cohort heatmap saved to: {heatmap_path}", flush=True)

    # 8. Export Summary JSON
    summary_data = {
        'total_accounts': len(df),
        'churned_accounts': int(E.sum()),
        'retention_rate_pct': round((1 - E.mean()) * 100, 2),
        'median_tenure_months': float(T.median()),
        'mean_tenure_months': round(float(T.mean()), 2),
        'logrank_enterprise_vs_basic_pvalue': round(float(lr_tier.p_value), 4),
        'top_hazard_factors': cox_summary.sort_values(by='Hazard_Ratio (exp)', ascending=False).to_dict(orient='index')
    }

    with open(os.path.join(OUT_DIR, "survival_metrics.json"), 'w') as f:
        json.dump(summary_data, f, indent=2)

    print("\nSurvival & Cohort Analysis completed successfully!", flush=True)

if __name__ == "__main__":
    run_survival_cohort_analysis()
