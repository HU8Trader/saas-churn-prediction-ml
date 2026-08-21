"""
Author: Himansh Upadhyay
GitHub: https://github.com/HU8Trader

Customer Lifetime Value (LTV) & Time-Series Revenue Forecasting Pipeline
Computes Realized & Predictive LTV per account, breaks down cohort LTV,
and forecasts 12-month forward MRR and ARR growth trajectories.
"""

import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIGS_DIR = os.path.join(BASE_DIR, "figures")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

def run_ltv_revenue_forecast():
    print("=" * 70, flush=True)
    print("SAAS LTV & TIME-SERIES REVENUE FORECASTING PIPELINE", flush=True)
    print("=" * 70, flush=True)

    # 1. Load Data
    df = pd.read_csv(os.path.join(DATA_DIR, "customer_360_analytical.csv"))
    df_subs = pd.read_csv(os.path.join(DATA_DIR, "fact_subscriptions.csv"))

    # 2. Realized Lifetime Value Calculation
    # Total historical revenue collected from subscriptions
    realized_ltv = df_subs.groupby('account_id')['mrr_amount'].sum().reset_index()
    realized_ltv.columns = ['account_id', 'realized_lifetime_value_usd']
    
    df = df.merge(realized_ltv, on='account_id', how='left')
    df['realized_lifetime_value_usd'] = df['realized_lifetime_value_usd'].fillna(0)

    # 3. Formula-Based Expected LTV
    # LTV = (ARPU * Gross Margin) / Churn Rate
    gross_margin = 0.80
    monthly_churn_rate = 0.0183 # 1.83% monthly account churn
    
    df['expected_ltv_usd'] = (df['total_mrr'] * gross_margin) / monthly_churn_rate

    print("\n" + "-" * 70, flush=True)
    print("CUSTOMER LIFETIME VALUE (LTV) COHORT SUMMARY")
    print("-" * 70, flush=True)
    
    tier_ltv = df.groupby('plan_tier').agg(
        account_count=('account_id', 'count'),
        avg_mrr=('total_mrr', 'mean'),
        avg_realized_ltv=('realized_lifetime_value_usd', 'mean'),
        avg_expected_ltv=('expected_ltv_usd', 'mean'),
        median_tenure_months=('tenure_months', 'median')
    ).round(2)

    print(tier_ltv.to_string(), flush=True)
    tier_ltv.to_csv(os.path.join(DATA_DIR, "ltv_by_tier_summary.csv"))

    # 4. Predictive LTV Regression Model
    # Predict realized LTV based on initial firmographic and operational signals
    feature_cols = [
        'seats', 'is_trial', 'total_feature_events', 'total_duration_hours',
        'total_errors', 'distinct_features_used', 'total_tickets', 'avg_csat'
    ]
    
    X = df[feature_cols].fillna(0)
    y = df['realized_lifetime_value_usd']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y)

    coef_df = pd.DataFrame({
        'feature': feature_cols,
        'standardized_coefficient': model.coef_
    }).sort_values(by='standardized_coefficient', ascending=False)

    print("\n" + "-" * 70, flush=True)
    print("PREDICTIVE LTV DRIVERS (Standardized Regression Coefficients)")
    print("-" * 70, flush=True)
    print(coef_df.to_string(index=False), flush=True)

    # 5. Monthly Historical MRR Time-Series
    df_subs['start_month'] = pd.to_datetime(df_subs['start_date']).dt.to_period('M')
    monthly_mrr = df_subs.groupby('start_month')['mrr_amount'].sum().reset_index()
    monthly_mrr.columns = ['month', 'mrr_added']
    monthly_mrr['month_str'] = monthly_mrr['month'].astype(str)

    # Historical cumulative run-rate MRR
    monthly_mrr['cumulative_mrr'] = monthly_mrr['mrr_added'].cumsum()
    monthly_mrr['cumulative_arr'] = monthly_mrr['cumulative_mrr'] * 12

    # 6. 12-Month Forward Forecast (Linear Trend with Seasonality & Churn Drag)
    t = np.arange(len(monthly_mrr))
    poly_fit = np.polyfit(t, monthly_mrr['mrr_added'], 1)
    
    future_t = np.arange(len(monthly_mrr), len(monthly_mrr) + 12)
    forecast_dates = pd.period_range(start=monthly_mrr['month'].iloc[-1] + 1, periods=12, freq='M')
    
    forecast_added = np.polyval(poly_fit, future_t)
    # Apply 1.8% churn leakage
    churn_loss_factor = 0.982
    
    forecast_df = pd.DataFrame({
        'forecast_month': forecast_dates.astype(str),
        'projected_mrr_added': np.maximum(forecast_added, 100000).round(2)
    })

    last_cum_mrr = monthly_mrr['cumulative_mrr'].iloc[-1]
    proj_cum_mrr = []
    curr = last_cum_mrr
    for add in forecast_df['projected_mrr_added']:
        curr = (curr * churn_loss_factor) + add
        proj_cum_mrr.append(curr)

    forecast_df['projected_total_mrr'] = np.array(proj_cum_mrr).round(2)
    forecast_df['projected_total_arr'] = (forecast_df['projected_total_mrr'] * 12).round(2)

    print("\n" + "-" * 70, flush=True)
    print("12-MONTH FORWARD MRR & ARR FORECAST TRAJECTORY")
    print("-" * 70, flush=True)
    print(forecast_df.head(6).to_string(index=False), flush=True)

    forecast_df.to_csv(os.path.join(DATA_DIR, "revenue_forecast_12m.csv"), index=False)

    # 7. Visualization: Historical vs Forecast
    plt.figure(figsize=(14, 6))
    
    hist_x = monthly_mrr['month_str']
    hist_y = monthly_mrr['cumulative_mrr'] / 1e6
    plt.plot(hist_x, hist_y, marker='o', color='#F59E0B', lw=2.5, label='Historical Total MRR ($M)')
    
    fore_x = forecast_df['forecast_month']
    fore_y = forecast_df['projected_total_mrr'] / 1e6
    plt.plot(fore_x, fore_y, marker='s', linestyle='--', color='#10B981', lw=2.5, label='12-Month Projected MRR ($M)')

    plt.title('SaaS Net MRR Run-Rate: Historical Trend & 12-Month Projection', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Billing Month', fontsize=11)
    plt.ylabel('Total MRR ($ Millions USD)', fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()

    plot_path = os.path.join(FIGS_DIR, "ltv_and_revenue_forecast.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Revenue forecast chart saved to: {plot_path}", flush=True)

    # 8. Export Metrics JSON
    summary = {
        'total_active_mrr': float(df['total_mrr'].sum()),
        'total_active_arr': float(df['total_mrr'].sum() * 12),
        'avg_realized_ltv': round(float(df['realized_lifetime_value_usd'].mean()), 2),
        'avg_expected_ltv': round(float(df['expected_ltv_usd'].mean()), 2),
        'projected_12m_exit_mrr': float(forecast_df['projected_total_mrr'].iloc[-1]),
        'projected_12m_exit_arr': float(forecast_df['projected_total_arr'].iloc[-1]),
        'ltv_drivers': coef_df.to_dict(orient='records')
    }

    with open(os.path.join(OUT_DIR, "ltv_forecast_metrics.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\nLTV & Revenue Forecasting completed successfully!", flush=True)

if __name__ == "__main__":
    run_ltv_revenue_forecast()
