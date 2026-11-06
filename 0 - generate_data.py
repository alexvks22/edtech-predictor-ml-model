import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(2026)
n_samples = 2000

print("Generating Advanced High-Touch EdTech Dataset...")

# 1. Identifiers & Demographics
student_ids = [f"STU_{202600 + i}" for i in range(n_samples)]
countries = np.random.choice(['GR', 'CY', 'UK', 'DE', 'US'], size=n_samples, p=[0.75, 0.12, 0.05, 0.05, 0.03]) # Noise
channels_raw = ['LinkedIn', 'linkedin', 'LI', 'YouTube', 'yt', 'Google', 'Newsletter', 'Referral', np.nan]
discovery_channels = np.random.choice(channels_raw, size=n_samples, p=[0.30, 0.15, 0.05, 0.20, 0.05, 0.10, 0.10, 0.02, 0.03])
primary_interests = np.random.choice(['Data Analytics', 'Analytics Engineering', 'Power BI', np.nan], size=n_samples, p=[0.40, 0.35, 0.18, 0.07])

# 2. Timing (Reach-out vs Sign-up)
# Days signup vs reachout: Negative = signup first, Positive = reachout first
days_signup_vs_reachout = np.random.normal(loc=-2, scale=10, size=n_samples).round(1)

# 3. Midterm Engagement Features (First 14 days after reach-out)
replied_to_followup = np.random.binomial(1, p=0.22, size=n_samples)
sent_second_message = np.where(replied_to_followup == 1, np.random.binomial(1, 0.65, size=n_samples), np.random.binomial(1, 0.05, size=n_samples))
free_courses_enrolled = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.40, 0.35, 0.18, 0.07])
free_semantic_completed = np.where(free_courses_enrolled >= 1, np.random.binomial(1, 0.15, size=n_samples), 0)
logins_14d = np.random.poisson(lam=4, size=n_samples)
video_mins_14d = np.random.exponential(scale=45, size=n_samples).round(1)

# 4. Target 1: Midterm Conversion (Strongly driven by followup reply & second message)
conv_score = (
    -2.5 
    + 2.8 * replied_to_followup 
    + 1.5 * sent_second_message 
    + 1.2 * free_semantic_completed 
    + 0.05 * logins_14d
)
conv_prob = 1 / (1 + np.exp(-conv_score))
is_converted = np.random.binomial(1, p=np.clip(conv_prob, 0.01, 0.92))

# 5. Capstone / Post-Purchase Features (Applied to converted students)
enrollment_types = []
sessions_booked = []
projects_submitted = []
reported_ai_usage = []
passed_ai_check = []
months_enrolled = []
is_stalled = []

for i in range(n_samples):
    if is_converted[i] == 1:
        # 30% Full Track Program, 70% Single Course
        is_program = np.random.rand() < 0.30
        enrollment_types.append('Full Track Program' if is_program else 'Single Course')
        
        months = int(np.random.uniform(1, 24))
        months_enrolled.append(months)
        
        if is_program:
            # 1-on-1 sessions booked (0 to 6)
            s_booked = int(np.random.choice([0, 1, 2, 3, 4, 5, 6], p=[0.35, 0.20, 0.15, 0.10, 0.10, 0.05, 0.05]))
            sessions_booked.append(s_booked)
            projs = int(np.random.choice([0, 1, 2, 3, 4, 5], p=[0.45, 0.25, 0.15, 0.08, 0.05, 0.02])) # Only 2% finish all
            projects_submitted.append(projs)
            
            ai_used = np.random.binomial(1, 0.60)
            reported_ai_usage.append(ai_used)
            ai_pass = np.random.binomial(1, 0.75) if ai_used == 1 else 1
            passed_ai_check.append(ai_pass)
            
            # High stall probability even with support (reflecting reality)
            stall_p = 0.88 if (s_booked < 2 or projs < 2) else 0.45
            is_stalled.append(np.random.binomial(1, stall_p))
        else:
            sessions_booked.append(0)
            projs = int(np.random.choice([0, 1, 2], p=[0.40, 0.40, 0.20]))
            projects_submitted.append(projs)
            reported_ai_usage.append(np.random.binomial(1, 0.40))
            passed_ai_check.append(1)
            is_stalled.append(1 if projs < 1 and months > 3 else 0)
    else:
        enrollment_types.append('None')
        months_enrolled.append(0)
        sessions_booked.append(0)
        projects_submitted.append(0)
        reported_ai_usage.append(0)
        passed_ai_check.append(0)
        is_stalled.append(np.nan)

# Create DataFrame
df = pd.DataFrame({
    'student_id': student_ids,
    'user_country_code': countries, # Noise feature
    'discovery_channel': discovery_channels,
    'primary_interest': primary_interests,
    'days_signup_vs_reachout': days_signup_vs_reachout,
    'replied_to_followup_email': replied_to_followup,
    'sent_second_message': sent_second_message,
    'free_courses_enrolled_count': free_courses_enrolled,
    'free_semantic_model_completed': free_semantic_completed,
    'platform_logins_count_14d': logins_14d,
    'total_video_minutes_watched_14d': video_mins_14d,
    'is_converted': is_converted, # TARGET 1 (Midterm)
    'enrollment_type': enrollment_types,
    'months_since_enrollment': months_enrolled,
    'sessions_30min_booked': sessions_booked,
    'projects_submitted_count': projects_submitted,
    'reported_ai_tool_usage': reported_ai_usage,
    'passed_ai_comprehension_check': passed_ai_check,
    'is_stalled_or_abandoned': is_stalled # TARGET 2 (Capstone)
})

df.to_csv('edtech_advanced_dataset.csv', index=False)
print("Dataset 'edtech_advanced_dataset.csv' created successfully with 2,000 rows!")