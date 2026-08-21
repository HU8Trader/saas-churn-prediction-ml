"""
Feature Engineering & Customer 360 Analytical Dataset Generator
Consolidates dim_accounts, fact_subscriptions, fact_churn_events,
fact_feature_usage, and fact_support_tickets into a rich analytical feature matrix.
"""

import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIGS_DIR = os.path.join(BASE_DIR, "figures")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

def build_customer_360():
    print("Loading raw SaaS datasets...")
    df_accounts = pd.read_csv(os.path.join(DATA_DIR, "dim_accounts.csv"))
    df_subs = pd.read_csv(os.path.join(DATA_DIR, "fact_subscriptions.csv"))
    df_churn = pd.read_csv(os.path.join(DATA_DIR, "fact_churn_events.csv"))
    df_usage = pd.read_csv(os.path.join(DATA_DIR, "fact_feature_usage.csv"))
    df_tickets = pd.read_csv(os.path.join(DATA_DIR, "fact_support_tickets.csv"))
    df_date = pd.read_csv(os.path.join(DATA_DIR, "dim_date.csv"))

    print(f"Loaded: {len(df_accounts)} accounts, {len(df_subs)} subscriptions, {len(df_churn)} churn events, {len(df_usage)} feature logs, {len(df_tickets)} tickets.")

    # 1. Process Subscription Features per Account
    subs_agg = df_subs.groupby('account_id').agg(
        total_mrr=('mrr_amount', 'sum'),
        avg_sub_mrr=('mrr_amount', 'mean'),
        max_sub_mrr=('mrr_amount', 'max'),
        total_subscriptions_count=('subscription_id', 'count'),
        upgrade_count=('upgrade_flag', 'sum'),
        downgrade_count=('downgrade_flag', 'sum'),
        annual_billing_count=('billing_frequency', lambda x: (x == 'annual').sum()),
        monthly_billing_count=('billing_frequency', lambda x: (x == 'monthly').sum()),
        first_sub_date=('start_date', 'min'),
        last_sub_date=('start_date', 'max')
    ).reset_index()

    subs_agg['net_expansion_score'] = subs_agg['upgrade_count'] - subs_agg['downgrade_count']
    subs_agg['annual_billing_ratio'] = subs_agg['annual_billing_count'] / subs_agg['total_subscriptions_count']

    # Map subscription_id to account_id in feature usage
    df_usage = df_usage.merge(df_subs[['subscription_id', 'account_id']], on='subscription_id', how='left')

    # 2. Process Feature Usage Features per Account
    usage_agg = df_usage.groupby('account_id').agg(
        total_feature_events=('usage_id', 'count'),
        total_duration_hours=('usage_duration_secs', lambda x: x.sum() / 3600.0),
        avg_session_duration_mins=('usage_duration_secs', lambda x: x.mean() / 60.0),
        total_errors=('error_count', 'sum'),
        distinct_features_used=('feature_name', 'nunique'),
        beta_feature_events=('is_beta_feature', lambda x: (x == True).sum())
    ).reset_index()

    usage_agg['error_rate_per_100_events'] = (usage_agg['total_errors'] / np.maximum(usage_agg['total_feature_events'], 1)) * 100.0
    usage_agg['beta_feature_usage_ratio'] = usage_agg['beta_feature_events'] / np.maximum(usage_agg['total_feature_events'], 1)

    # 3. Process Support Ticket Features per Account
    ticket_agg = df_tickets.groupby('account_id').agg(
        total_tickets=('ticket_id', 'count'),
        avg_csat=('satisfaction_score', lambda x: x[x > 0].mean() if len(x[x > 0]) > 0 else np.nan),
        min_csat=('satisfaction_score', lambda x: x[x > 0].min() if len(x[x > 0]) > 0 else np.nan),
        avg_first_response_min=('first_response_time_minutes', 'mean'),
        avg_resolution_hours=('resolution_time_hours', 'mean'),
        total_escalations=('escalation_flag', lambda x: (x == True).sum()),
        urgent_tickets=('priority', lambda x: (x == 'Urgent').sum()),
        high_tickets=('priority', lambda x: (x == 'High').sum())
    ).reset_index()

    ticket_agg['escalation_rate_pct'] = (ticket_agg['total_escalations'] / np.maximum(ticket_agg['total_tickets'], 1)) * 100.0
    ticket_agg['urgent_ticket_ratio'] = (ticket_agg['urgent_tickets'] + ticket_agg['high_tickets']) / np.maximum(ticket_agg['total_tickets'], 1)

    # 4. Process Churn Details per Account
    churn_info = df_churn.groupby('account_id').agg(
        churn_reason=('reason_code', 'first'),
        refund_amount_usd=('refund_amount_usd', 'sum'),
        is_reactivation=('is_reactivation', 'any'),
        churn_date=('churn_date', 'first')
    ).reset_index()

    # 5. Merge all into Master Customer 360 Table
    c360 = df_accounts.copy()
    c360 = c360.merge(subs_agg, on='account_id', how='left')
    c360 = c360.merge(usage_agg, on='account_id', how='left')
    c360 = c360.merge(ticket_agg, on='account_id', how='left')
    c360 = c360.merge(churn_info, on='account_id', how='left')

    # Fill NaNs for accounts with 0 usage / 0 tickets
    c360['total_mrr'] = c360['total_mrr'].fillna(0)
    c360['mrr_per_seat'] = c360['total_mrr'] / np.maximum(c360['seats'], 1)
    c360['total_subscriptions_count'] = c360['total_subscriptions_count'].fillna(0).astype(int)
    c360['upgrade_count'] = c360['upgrade_count'].fillna(0).astype(int)
    c360['downgrade_count'] = c360['downgrade_count'].fillna(0).astype(int)
    c360['net_expansion_score'] = c360['net_expansion_score'].fillna(0)
    c360['annual_billing_ratio'] = c360['annual_billing_ratio'].fillna(0)

    c360['total_feature_events'] = c360['total_feature_events'].fillna(0).astype(int)
    c360['total_duration_hours'] = c360['total_duration_hours'].fillna(0)
    c360['avg_session_duration_mins'] = c360['avg_session_duration_mins'].fillna(0)
    c360['total_errors'] = c360['total_errors'].fillna(0).astype(int)
    c360['error_rate_per_100_events'] = c360['error_rate_per_100_events'].fillna(0)
    c360['distinct_features_used'] = c360['distinct_features_used'].fillna(0).astype(int)
    c360['beta_feature_usage_ratio'] = c360['beta_feature_usage_ratio'].fillna(0)

    # Global CSAT median fill for accounts with 0 rated tickets
    global_csat_mean = df_tickets[df_tickets['satisfaction_score'] > 0]['satisfaction_score'].mean()
    c360['avg_csat'] = c360['avg_csat'].fillna(global_csat_mean)
    c360['total_tickets'] = c360['total_tickets'].fillna(0).astype(int)
    c360['avg_first_response_min'] = c360['avg_first_response_min'].fillna(df_tickets['first_response_time_minutes'].mean())
    c360['avg_resolution_hours'] = c360['avg_resolution_hours'].fillna(df_tickets['resolution_time_hours'].mean())
    c360['total_escalations'] = c360['total_escalations'].fillna(0).astype(int)
    c360['escalation_rate_pct'] = c360['escalation_rate_pct'].fillna(0)
    c360['urgent_tickets'] = c360['urgent_tickets'].fillna(0).astype(int)
    c360['urgent_ticket_ratio'] = c360['urgent_ticket_ratio'].fillna(0)

    c360['refund_amount_usd'] = c360['refund_amount_usd'].fillna(0)
    c360['churn_flag'] = c360['churn_flag'].astype(int)
    c360['is_trial'] = c360['is_trial'].astype(int)

    # Calculate Tenure in days (from signup_date to 2024-12-31 or churn_date)
    ref_date = pd.to_datetime('2024-12-31')
    c360['signup_dt'] = pd.to_datetime(c360['signup_date'])
    c360['churn_dt'] = pd.to_datetime(c360['churn_date'])
    
    end_dates = np.where(c360['churn_dt'].notna(), c360['churn_dt'], ref_date)
    c360['tenure_days'] = (pd.to_datetime(end_dates) - c360['signup_dt']).dt.days.clip(lower=1)
    c360['tenure_months'] = (c360['tenure_days'] / 30.4375).round(1)

    # Output CSV
    out_path = os.path.join(DATA_DIR, "customer_360_analytical.csv")
    c360.to_csv(out_path, index=False)
    print(f"Customer 360 Analytical Dataset created successfully: {out_path} ({len(c360)} rows, {len(c360.columns)} features)")
    return c360

if __name__ == "__main__":
    build_customer_360()
