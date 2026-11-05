import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Fix seed for reproducibility
np.random.seed(42)
n_samples = 10000

print("Generating synthetic EdTech dataset...")

# 1. Basic Demographics & Identifiers
student_ids = [f"STU_{10000 + i}" for i in range(n_samples)]

# Signup dates spread over the last 2 years
start_date = datetime(2024, 1, 1)
signup_dates = [start_date + timedelta(days=int(np.random.uniform(0, 700)), 
                                        hours=int(np.random.uniform(0, 23))) for _ in range(n_samples)]

ages = np.random.normal(loc=31, scale=8, size=n_samples).astype(int)
ages = np.clip(ages, 18, 65)
# Inject Outliers in Age (-1 or 99)
age_outlier_idx = np.random.choice(n_samples, size=50, replace=False)
ages[age_outlier_idx] = np.random.choice([-1, 99], size=50)

# Gender with missing values (NaN)
genders = np.random.choice(['Male', 'Female', 'Other', np.nan], size=n_samples, p=[0.45, 0.38, 0.02, 0.15])

# Inconsistent Lead Sources
lead_sources_raw = ['Facebook', 'facebook', 'FB', 'fb_ads', 'Google_Ads', 'google', 'Google', 'Referral', 'Organic', 'LinkedIn', 'linkedin']
lead_sources = np.random.choice(lead_sources_raw, size=n_samples, p=[0.1, 0.08, 0.05, 0.07, 0.2, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05])

# Primary Interest with NaNs
interests = np.random.choice(['Data Science', 'Web Development', 'UI/UX Design', 'Digital Marketing', np.nan], 
                             size=n_samples, p=[0.35, 0.30, 0.12, 0.13, 0.10])

# Engagement features (Free previews)
free_previews = np.random.poisson(lam=3, size=n_samples)
# Inject Bot Outliers (extremely high views)
bot_idx = np.random.choice(n_samples, size=20, replace=False)
free_previews[bot_idx] = np.random.randint(250, 600, size=20)

# 2. Target Generation (Imbalanced Lead Conversion ~14%)
# Conversion probability depends on engagement and source
conversion_prob = (
    0.05 
    + 0.04 * free_previews 
    + np.where(pd.Series(lead_sources).isin(['Referral', 'LinkedIn', 'linkedin']), 0.12, 0)
    + np.where(pd.Series(interests) == 'Data Science', 0.08, 0)
)
# Sigmoid normalization to probability range [0, 1]
conversion_prob = 1 / (1 + np.exp(- (conversion_prob - 2)))
is_converted = np.random.binomial(1, p=np.clip(conversion_prob, 0.02, 0.85))

# 3. Purchases & Dates
first_purchase_dates = []
purchase_amounts = []

for i in range(n_samples):
    if is_converted[i] == 1:
        # Purchased within 1-14 days
        days_to_buy = int(np.random.exponential(scale=4)) + 1
        purchase_date = signup_dates[i] + timedelta(days=days_to_buy)
        first_purchase_dates.append(purchase_date.strftime('%Y-%m-%d'))
        # Purchase amount in EUR
        amount = np.random.choice([49.0, 99.0, 149.0, 299.0], p=[0.2, 0.5, 0.2, 0.1])
        purchase_amounts.append(amount)
    else:
        first_purchase_dates.append(np.nan)
        purchase_amounts.append(0.0)

# 4. Capstone Features (Course Completion & Churn ~26%)
completion_rates = []
is_churned = []

for i in range(n_samples):
    if is_converted[i] == 1:
        rate = round(float(np.random.beta(a=2, b=3)), 2)
        completion_rates.append(rate)
        # Higher churn if completion rate is low
        churn_p = 0.65 if rate < 0.30 else 0.12
        is_churned.append(np.random.binomial(1, churn_p))
    else:
        completion_rates.append(0.0)
        is_churned.append(1)  # Non-converted leads are considered churned/lost

# Create DataFrame
df = pd.DataFrame({
    'student_id': student_ids,
    'signup_date': [d.strftime('%Y-%m-%d %H:%M:%S') if np.random.rand() > 0.1 else d.strftime('%d/%m/%Y') for d in signup_dates], # Mixed date formats
    'age': ages,
    'gender': genders,
    'lead_source': lead_sources,
    'primary_interest': interests,
    'free_preview_views': free_previews,
    'first_purchase_date': first_purchase_dates,
    'purchase_amount_eur': purchase_amounts,
    'is_converted': is_converted,
    'course_completion_rate': completion_rates,
    'is_churned': is_churned
})

# Save to CSV
df.to_csv('edtech_students_raw.csv', index=False)
print("Dataset 'edtech_students_raw.csv' created successfully with 10,000 rows!")