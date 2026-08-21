"""
Predictive Churn Modeling Pipeline
Trains Logistic Regression, Random Forest, and XGBoost classifiers on Customer 360 features.
Performs 5-Fold Stratified Cross-Validation, evaluates ROC-AUC / PR-AUC / F1, extracts Feature Importance,
and serializes the production model for real-time inference.
"""

import pandas as pd
import numpy as np
import os
import joblib
import json

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix, classification_report, brier_score_loss
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIGS_DIR = os.path.join(BASE_DIR, "figures")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

def run_churn_pipeline():
    print("=" * 70, flush=True)
    print("SAAS PREDICTIVE CHURN MODELING PIPELINE", flush=True)
    print("=" * 70, flush=True)

    # 1. Load Data
    df_path = os.path.join(DATA_DIR, "customer_360_analytical.csv")
    df = pd.read_csv(df_path)
    print(f"Loaded Customer 360 analytical dataset: {df.shape[0]} accounts, {df.shape[1]} raw columns.", flush=True)

    # 2. Define Features & Target
    target = 'churn_flag'
    
    numeric_features = [
        'seats', 'is_trial', 'total_mrr', 'mrr_per_seat', 'total_subscriptions_count',
        'upgrade_count', 'downgrade_count', 'net_expansion_score', 'annual_billing_ratio',
        'total_feature_events', 'total_duration_hours', 'avg_session_duration_mins',
        'total_errors', 'error_rate_per_100_events', 'distinct_features_used', 'beta_feature_usage_ratio',
        'total_tickets', 'avg_csat', 'avg_first_response_min', 'avg_resolution_hours',
        'total_escalations', 'escalation_rate_pct', 'urgent_tickets', 'urgent_ticket_ratio', 'tenure_months'
    ]

    categorical_features = ['industry', 'country', 'plan_tier', 'referral_source']

    X = df[numeric_features + categorical_features]
    y = df[target]

    print(f"Target distribution: Active = {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%), Churned = {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)", flush=True)

    # 3. Train / Test Split (Stratified 80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} accounts | Test set: {X_test.shape[0]} accounts", flush=True)

    # 4. Preprocessing Pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
        ]
    )

    # 5. Define Candidate Models (with n_jobs=1 for rock-solid stability)
    models = {
        'Logistic Regression (Baseline)': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
        ]),
        'Random Forest Classifier': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=150, max_depth=6, min_samples_split=5,
                class_weight='balanced', random_state=42, n_jobs=1
            ))
        ]),
        'XGBoost Classifier': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(
                n_estimators=120, max_depth=4, learning_rate=0.05,
                scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
                random_state=42, eval_metric='logloss', n_jobs=1
            ))
        ])
    }

    # 6. Stratified 5-Fold Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}

    print("\n" + "-" * 70, flush=True)
    print("5-FOLD STRATIFIED CROSS-VALIDATION RESULTS (ROC-AUC)", flush=True)
    print("-" * 70, flush=True)

    for name, pipe in models.items():
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=1)
        cv_results[name] = scores
        print(f"{name:32s} | Mean ROC-AUC: {scores.mean():.4f} (±{scores.std():.4f})", flush=True)

    # 7. Model Training & Test Set Evaluation
    print("\n" + "=" * 70, flush=True)
    print("TEST SET EVALUATION (HOLD-OUT 20%)", flush=True)
    print("=" * 70, flush=True)

    best_model_name = None
    best_roc_auc = -1
    best_pipeline = None
    eval_metrics = {}

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        roc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        brier = brier_score_loss(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        eval_metrics[name] = {
            'roc_auc': round(float(roc), 4),
            'pr_auc': round(float(pr_auc), 4),
            'accuracy': round(float(acc), 4),
            'precision': round(float(prec), 4),
            'recall': round(float(rec), 4),
            'f1_score': round(float(f1), 4),
            'brier_score': round(float(brier), 4),
            'confusion_matrix': cm.tolist()
        }

        print(f"\nModel: {name}", flush=True)
        print(f"  ROC-AUC Score: {roc:.4f}", flush=True)
        print(f"  PR-AUC Score:  {pr_auc:.4f}", flush=True)
        print(f"  Accuracy:      {acc:.4f}", flush=True)
        print(f"  Precision:     {prec:.4f}", flush=True)
        print(f"  Recall:        {rec:.4f}", flush=True)
        print(f"  F1 Score:      {f1:.4f}", flush=True)
        print(f"  Brier Score:   {brier:.4f}", flush=True)
        print(f"  Confusion Matrix: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}", flush=True)

        if roc > best_roc_auc:
            best_roc_auc = roc
            best_model_name = name
            best_pipeline = pipe

    print(f"\n>>> Best Performing Model on Hold-Out: {best_model_name} (ROC-AUC: {best_roc_auc:.4f})", flush=True)

    # 8. Feature Importance Analysis
    rf_model = models['Random Forest Classifier'].named_steps['classifier']
    preproc = models['Random Forest Classifier'].named_steps['preprocessor']
    
    cat_encoder = preproc.named_transformers_['cat']
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
    all_feature_names = numeric_features + cat_feature_names

    importances = rf_model.feature_importances_
    feat_imp = pd.DataFrame({
        'feature': all_feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    print("\n" + "=" * 70, flush=True)
    print("TOP 15 MOST PREDICTIVE CHURN FEATURES (Random Forest Gini Importance)", flush=True)
    print("=" * 70, flush=True)
    for idx, row in feat_imp.head(15).reset_index(drop=True).iterrows():
        print(f"{idx+1:2d}. {row['feature']:35s} : {row['importance']*100:6.2f}%", flush=True)

    # Save feature importance CSV
    feat_imp.to_csv(os.path.join(MODELS_DIR, "churn_feature_importance.csv"), index=False)

    # 9. Save Production Model & Metadata
    model_export_path = os.path.join(MODELS_DIR, "churn_model.pkl")
    joblib.dump(best_pipeline, model_export_path)

    metadata = {
        'best_model': best_model_name,
        'best_roc_auc': best_roc_auc,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'all_feature_names': all_feature_names,
        'test_metrics': eval_metrics,
        'top_10_features': feat_imp.head(10).to_dict(orient='records')
    }

    meta_export_path = os.path.join(MODELS_DIR, "churn_model_metadata.json")
    with open(meta_export_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModel exported to: {model_export_path}", flush=True)
    print(f"Metadata exported to: {meta_export_path}", flush=True)
    print(f"Feature importance saved to: {os.path.join(BASE_DIR, 'churn_feature_importance.csv')}", flush=True)

    return best_pipeline, metadata

if __name__ == "__main__":
    run_churn_pipeline()
