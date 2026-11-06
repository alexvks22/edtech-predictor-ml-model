#!/usr/bin/env python
# coding: utf-8

"""
EdTech Lead Scoring - Production Training Script (train.py)
Machine Learning Zoomcamp Midterm Project
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import xgboost as xgb

# 1. Παράμετροι & Ρυθμίσεις
DATA_PATH = 'data/edtech_advanced_dataset.csv'
MODEL_OUTPUT_FILE = 'model.bin'
SEED = 42

# Βέλτιστες υπερπαράμετροι όπως προέκυψαν από το notebook
XGB_PARAMS = {
    'eta': 0.1,
    'max_depth': 4,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'seed': SEED,
    'nthread': -1
}
NUM_BOOST_ROUND = 100

# 2. Φόρτωση & Καθαρισμός Δεδομένων
def load_and_clean_data(file_path: str) -> pd.DataFrame:
    print(f"Loading raw data from '{file_path}'...")
    df = pd.read_csv(file_path)

    # Απομόνωση ΜΟΝΟ των στηλών του Midterm (αποφυγή Data Leakage από το Capstone)
    midterm_cols = [
        'student_id',
        'discovery_channel',
        'primary_interest',
        'days_signup_vs_reachout',
        'replied_to_followup_email',
        'sent_second_message',
        'free_courses_enrolled_count',
        'free_semantic_model_completed',
        'platform_logins_count_14d',
        'total_video_minutes_watched_14d',
        'is_converted'
    ]
    df = df[midterm_cols].copy()

    # Καθαρισμός κατηγορικών τιμών στο discovery_channel
    channel_map = {
        'linkedin': 'LinkedIn',
        'LI': 'LinkedIn',
        'yt': 'YouTube'
    }
    df['discovery_channel'] = df['discovery_channel'].replace(channel_map).fillna('Unknown')
    df['primary_interest'] = df['primary_interest'].fillna('Unknown')

    print(f"Data successfully cleaned. Shape: {df.shape}")
    return df

# 3. Εκπαίδευση & Αποθήκευση Μοντέλου
def train_and_export():
    df = load_and_clean_data(DATA_PATH)

    # Διαχωρισμός σε Full Train (80%) και Test Set (20%) με Stratification
    df_full_train, df_test = train_test_split(
        df, 
        test_size=0.20, 
        random_state=SEED, 
        stratify=df['is_converted']
    )

    y_full_train = df_full_train['is_converted'].values
    y_test = df_test['is_converted'].values

    # Επιλογή των τελικών χαρακτηριστικών (χωρίς student_id και target)
    feature_cols = [c for c in df_full_train.columns if c not in ['student_id', 'is_converted']]

    print(f"Features used for training ({len(feature_cols)}):")
    print(f"  {feature_cols}")

    # Encoding με DictVectorizer
    dv = DictVectorizer(sparse=False)
    X_full_train = dv.fit_transform(df_full_train[feature_cols].to_dict(orient='records'))
    X_test = dv.transform(df_test[feature_cols].to_dict(orient='records'))

    feature_names = list(dv.get_feature_names_out())

    # Δημιουργία XGBoost DMatrix
    dfulltrain = xgb.DMatrix(X_full_train, label=y_full_train, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)

    # Εκπαίδευση του τελικού μοντέλου
    print("\nTraining final XGBoost model...")
    model = xgb.train(XGB_PARAMS, dfulltrain, num_boost_round=NUM_BOOST_ROUND)

    # Αξιολόγηση στο κρυφό Test Set
    y_pred = model.predict(dtest)
    test_roc_auc = roc_auc_score(y_test, y_pred)
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    test_pr_auc = auc(recall, precision)

    print("\n=== FINAL TEST METRICS ===")
    print(f"  Test ROC-AUC : {test_roc_auc:.4f}")
    print(f"  Test PR-AUC  : {test_pr_auc:.4f}")

    # Αποθήκευση DictVectorizer και Model στο model.bin
    print(f"\nSaving DictVectorizer and Model to '{MODEL_OUTPUT_FILE}'...")
    with open(MODEL_OUTPUT_FILE, 'wb') as f_out:
        pickle.dump((dv, model), f_out)

    print("Success! 'model.bin' is ready for deployment.\n")

if __name__ == '__main__':
    train_and_export()