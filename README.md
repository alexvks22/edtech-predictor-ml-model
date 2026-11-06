# 🎓 EdTech High-Touch Lead Scoring Engine

> **Midterm Project** for the [DataTalks.Club Machine Learning Zoomcamp (2026 Cohort)](https://github.com/DataTalksClub/machine-learning-zoomcamp)  
> **Author:** Alexandra  
> **Role:** Senior BI & Analytics Engineer | Applied ML Practitioner  

---

## 📌 Executive Summary & Business Context

In specialized education and high-touch mentorship tracks, customer conversion behaves fundamentally differently from low-touch, self-serve SaaS products:

* **Cold registrations rarely convert:** Historical funnel metrics indicate that passive free signups converting without direct 1-on-1 contact convert at a baseline below $3\%$.
* **Conversations drive revenue:** Conversions happen after personalized discovery touchpoints (meeting requests, 1-on-1 strategy sessions, and clarifying email exchanges).
* **The Bandwidth Bottleneck:** Sales and consultation time is strictly finite. Reaching out manually to every registered lead exhausts operational capacity on disengaged users. Conversely, failing to respond rapidly to high-intent prospective students leads to lost opportunities.

### Business Objective
Design, benchmark, containerize, and deploy an end-to-end Machine Learning web service to predict the probability of a prospect converting (`is_converted = 1`) within a **14-day observation window following their initial reach-out touchpoint**.

Operational triage thresholds:
1. **High Intent ($p \ge 0.70$):** Immediate calendar booking priority for personal consultation.
2. **Warm ($0.30 \le p < 0.70$):** Automated nurture workflow focused on hands-on labs and semantic modeling workshops.
3. **Cold ($p < 0.30$):** Retained in general asynchronous newsletter broadcasts without dedicating manual hours.

---

## 🛡️ Problem Framing & Leakage Prevention Strategy

A primary structural risk in transactional ML models is **Data Leakage** (evaluating models on features populated only after buyer intent or payment has occurred).

```
                      [ POINT ZERO (t = 0): INITIAL REACH-OUT ]
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
┌───────────────────────┐                             ┌───────────────────────┐
│   PRE-REACHOUT DATA   │                             │  14-DAY WINDOW METRICS│
│  (Source & Fit)       │                             │  (Behavior & Signals) │
└───────────────────────┘                             └───────────────────────┘
  • Discovery Channel                                   • Follow-up Email Replied
  • Primary Interest Area                               • Second Message Sent
  • Days Signup vs Contact                              • Free Courses Enrolled
                                                        • Free Semantic Model Done
                                                        • 14-day Platform Logins
                                                        • 14-day Video Minutes
                                                        │
                                                        ▼
                                             [ TARGET: IS_CONVERTED (0/1) ]
```

1. **Point Zero Definition ($t=0$):** Timelines are anchored to the date of first direct contact/message rather than registration.
2. **Strict Feature Isolation:** Columns generated after course commitment (`enrollment_type`, `months_since_enrollment`, `sessions_30min_booked`, `projects_submitted_count`, `reported_ai_tool_usage`, `passed_ai_comprehension_check`, `is_stalled_or_abandoned`) were excluded prior to modeling.

---

## 📊 Dataset & Feature Dictionary

The service is trained on `edtech_advanced_dataset.csv` (2,000 records, 26.75% conversion rate).

| Feature Name | Data Type | Role | Business Description |
| :--- | :--- | :--- | :--- |
| `discovery_channel` | Categorical | Input Feature | Initial referral channel (`LinkedIn`, `YouTube`, `Google`, `Newsletter`, `Referral`, `Unknown`). |
| `primary_interest` | Categorical | Input Feature | Target track domain (`Data Analytics`, `Analytics Engineering`, `Power BI`, `Unknown`). |
| `days_signup_vs_reachout` | Continuous (Float) | Input Feature | Elapsed days between signup and initial message (negative = signup preceded reachout). |
| `replied_to_followup_email`| Binary (0/1) | Input Feature | Whether the prospect responded to the first consultation follow-up email. |
| `sent_second_message` | Binary (0/1) | Input Feature | Whether the prospect sent a follow-up or clarifying inquiry. |
| `free_courses_enrolled_count` | Discrete (0 to 3) | Input Feature | Quantity of free preview courses added to profile. |
| `free_semantic_model_completed` | Binary (0/1) | Input Feature | Finished the hands-on Power BI Semantic Model preview module. |
| `platform_logins_count_14d` | Integer | Input Feature | Platform sessions logged during the 14-day window. |
| `total_video_minutes_watched_14d` | Continuous (Float) | Input Feature | Total video consumption in minutes during the 14-day window. |
| `user_country_code` | Categorical | **Discarded Noise** | Country identifier (`GR`, `CY`, `DE`, `US`, `UK`), removed via Mutual Information. |
| **`is_converted`** | Binary (0/1) | **Target Variable** | Paid enrollment completed within the 14-day window. |

---

## 🔬 Exploratory Data Analysis & Hypotheses

During exploration (`notebook_clean.ipynb`):

1. **Email Response Reciprocity:**  
   Proactive responses to follow-up emails (`replied_to_followup_email == 1`) drove an **81.59% conversion rate**, compared to just **11.78%** among non-repliers.
2. **Follow-up Persistence:**  
   Users who sent a second message converted at **80.38%** versus **14.70%** for single-touch users.
3. **Information Theory & Noise Detection:**  
   Calculating Mutual Information scores confirmed that `user_country_code` yielded negligible mutual information ($\text{MI} \approx 0.000$), validating its exclusion from production models to prevent overfitting and unwarranted variance.

---

## 🧪 Experimentation Matrix

Because the conversion rate shows moderate class imbalance (~26.75%), models were evaluated on both **ROC-AUC** and **Precision-Recall AUC (PR-AUC)** across a Stratified 60% Train / 20% Validation / 20% Test split.

| Experiment ID | Model Architecture | Feature Selection | Validation ROC-AUC | Validation PR-AUC | Key Finding |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **EXP-01** | Logistic Regression (Full) | All 10 features (with country) | `0.8527` | `0.7743` | Linear benchmark; cannot capture feature interactions. |
| **EXP-02** | Logistic Regression (Filtered) | 9 features (country dropped) | `0.8535` | `0.7745` | Confirming noise removal stabilizes the model without loss. |
| **EXP-03** | Decision Tree (`max_depth=4`) | Filtered | `0.8307` | `0.7345` | High variance; single trees struggle on threshold boundaries. |
| **EXP-04** | Random Forest (`n=100, depth=4`) | Filtered | **`0.8579`** | `0.7834` | Bagging ensemble smoothed decision surfaces across features. |
| **EXP-05** | **XGBoost (`eta=0.1, depth=4`)** | Filtered | `0.8469` | **`0.7923`** | **Selected Winner.** Highest PR-AUC, prioritizing positive cases. |

### Final Out-of-Sample Performance
Retrained on Full Train (Train + Validation, 1,600 observations) and evaluated on the held-out Test Set (400 observations):
* **Final Test ROC-AUC:** **`0.8379`**
* **Validation vs. Test Delta:** $< 0.01$, proving strong out-of-sample generalization with minimal overfitting.

---

## 🛠️ Repository File Structure

```text
.
├── data/
│   ├── 0 - generate_data.py              # Notebook to generate reproducable data that respect real-world situation
│   ├── edtech_advanced_dataset.csv       # Raw dataset
│   └── edtech_midterm_clean.csv          # Processed dataset
├── static/
│   ├── index.html                        # User friendly UI of model
├── Dockerfile                            # Container definition for Uvicorn on port 9696
├── 1 - notebook_clean.ipynb              # Complete EDA, hygiene, experimentation
├── 2 - prepare_data_and_train_clean.py   # Tuning and full retraining
├── train.py                              # Production retraining script (exports model.bin)
├── predict.py                            # FastAPI web service exposing /predict and /health
├── model.bin                             # Serialized DictVectorizer and XGBoost booster
├── requirements.txt                      # Locked Python dependencies
└── README.md                             # Project documentation and engineering log
```

---

## 🚀 Execution & Verification Guide

### 1. Local Environment Execution

1. **Clone repository and setup environment:**
   ```bash
   git clone https://github.com/alexvks22/edtech-predictor-ml-model.git
   cd edtech-predictor-ml-model
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Run training pipeline (generates `model.bin`):**
   ```bash
   python train.py
   ```

3. **Start the API service:**
   ```bash
   uvicorn predict:app --host 0.0.0.0 --port 9696 --reload
   ```

4. **Interactive Documentation:**
   Open your browser at `http://localhost:9696/docs` to test via Swagger UI.

---

### 2. Docker Container Execution

1. **Build image:**
   ```bash
   docker build -t edtech-predictor-ml-model:v1 .
   ```

2. **Run container:**
   ```bash
   docker run -it -p 9696:9696 --rm edtech-predictor-ml-model:v1
   ```

---

### 3. API Request Verification (`curl`)

Send a test request via terminal:

```bash
curl -X 'POST' \
  'http://localhost:9696/predict' \
  -H 'Content-Type: application/json' \
  -d '{
    "discovery_channel": "LinkedIn",
    "primary_interest": "Analytics Engineering",
    "days_signup_vs_reachout": -1.5,
    "replied_to_followup_email": 1,
    "sent_second_message": 1,
    "free_courses_enrolled_count": 2,
    "free_semantic_model_completed": 1,
    "platform_logins_count_14d": 5,
    "total_video_minutes_watched_14d": 60.0
  }'
```

**Expected JSON Response:**
```json
{
  "conversion_probability": 0.9696,
  "is_converted": true
}
```

---

## ☁️ Live Cloud Deployment

* **Live Base URL:** `https://edtech-predictor-ml-model.onrender.com`
* **Health Check:** `https://edtech-predictor-ml-model.onrender.com/health`
* **Interactive API Documentation (Swagger):** `https://edtech-predictor-ml-model.onrender.com/docs`

---

## 📋 Evaluation Criteria Verification
- [x] **Problem Formulation:** High-touch reach-out framing defined with operational triage actions.
- [x] **EDA & Data Hygiene:** Addressed missing values, category unification, and statistical noise elimination.
- [x] **Model Comparison:** Evaluated 5 structured experiments across linear, tree, and gradient boosting families.
- [x] **Metric Justification:** Benchmarked ROC-AUC and PR-AUC to address class imbalance.
- [x] **Training Script:** Autonomous `train.py` exports `model.bin`.
- [x] **Web Serving:** Implemented with FastAPI and validated via Pydantic (`predict.py`).
- [x] **Containerization:** Clean, multi-stage `Dockerfile` listening on port 9696.
- [x] **Cloud Verification:** Production endpoints deployed with public Swagger accessibility.