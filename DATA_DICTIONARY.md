# Telecom CVM Intelligence - Data Dictionary

This document provides a detailed mapping and description of every dataset and feature column used within the Telecom CVM (Customer Value Management) Intelligence platform.

---

## Section 1: IBM Telco Churn Dataset (Raw)

This dataset contains raw customer profile and subscription information from the IBM Telco Churn dataset.

| Field Name | Type | Description | Example Values |
| :--- | :--- | :--- | :--- |
| `customerID` | string | Unique customer identifier | `"7590-VHVEG"`, `"5575-GNVDE"` |
| `gender` | string | Customer gender | `"Male"`, `"Female"` |
| `SeniorCitizen` | int | Indicates if the customer is a senior citizen (1) or not (0) | `0`, `1` |
| `Partner` | string | Indicates if the customer has a partner | `"Yes"`, `"No"` |
| `Dependents` | string | Indicates if the customer has dependents | `"Yes"`, `"No"` |
| `tenure` | int | Number of months the customer has stayed with the company (0-72) | `1`, `34`, `72` |
| `PhoneService` | string | Indicates if the customer has subscribed to a phone service | `"Yes"`, `"No"` |
| `MultipleLines` | string | Indicates if the customer has multiple phone lines | `"Yes"`, `"No"`, `"No phone service"` |
| `InternetService` | string | Type of internet service provider the customer subscribes to | `"DSL"`, `"Fiber optic"`, `"No"` |
| `OnlineSecurity` | string | Indicates if the customer has online security add-on service | `"Yes"`, `"No"`, `"No internet service"` |
| `OnlineBackup` | string | Indicates if the customer has online backup add-on service | `"Yes"`, `"No"`, `"No internet service"` |
| `DeviceProtection` | string | Indicates if the customer has device protection add-on service | `"Yes"`, `"No"`, `"No internet service"` |
| `TechSupport` | string | Indicates if the customer has technical support add-on service | `"Yes"`, `"No"`, `"No internet service"` |
| `StreamingTV` | string | Indicates if the customer has streaming TV service | `"Yes"`, `"No"`, `"No internet service"` |
| `StreamingMovies` | string | Indicates if the customer has streaming movies service | `"Yes"`, `"No"`, `"No internet service"` |
| `Contract` | string | The contract term of the customer | `"Month-to-month"`, `"One year"`, `"Two year"` |
| `PaperlessBilling` | string | Indicates if the customer has paperless billing enabled | `"Yes"`, `"No"` |
| `PaymentMethod` | string | The customer's payment method | `"Electronic check"`, `"Mailed check"`, `"Bank transfer (automatic)"`, `"Credit card (automatic)"` |
| `MonthlyCharges` | float | The monthly amount charged to the customer in USD | `29.85`, `56.95`, `104.85` |
| `TotalCharges` | string | The total amount charged to the customer (contains spaces, requires cleaning) | `"29.85"`, `"1889.5"`, `" "` |
| `Churn` | string | Target variable indicating whether the customer churned | `"Yes"`, `"No"` |

---

## Section 2: Synthetic Customer Events Dataset (Generated)

This dataset captures granular customer transactions and event logs simulated over time by the event generator (`generator.py`).

| Field Name | Type | Description | Example Values |
| :--- | :--- | :--- | :--- |
| `customer_id` | string | Unique customer identifier, matches `customerID` in the IBM dataset | `"7590-VHVEG"`, `"5575-GNVDE"` |
| `event_date` | date | Date on which the event occurred | `2026-07-01`, `2026-07-03` |
| `event_type` | string | Type of activity recorded in the event | `"RECHARGE"`, `"DATA_SESSION"`, `"CALL"`, `"SMS"` |
| `recharge_amount` | float | Amount recharged in local currency (null if the event is not a recharge) | `500.0`, `1000.0`, `null` |
| `data_mb_used` | float | Volume of data consumed in megabytes (null if the event is not a data session) | `150.5`, `340.2`, `null` |
| `call_duration_minutes` | float | Duration of the phone call in minutes (null if the event is not a call) | `4.2`, `12.5`, `null` |
| `channel` | string | Interaction channel through which the event occurred | `"USSD"`, `"APP"`, `"AGENT"`, `"WEB"` |
| `offer_presented` | string | Marketing offer presented to the customer during the session (null if none) | `"Double Data 5GB"`, `"Discount 10%"`, `null` |
| `offer_accepted` | boolean | Indicates whether the customer accepted the presented offer (null if no offer presented) | `true`, `false`, `null` |

---

## Section 3: Engineered Features

These features are calculated or predicted by the preprocessing and modeling pipelines (`preprocessor.py`, `behavioural.py`, and machine learning models). They capture customer behavior trends, risk profiles, segments, and propensities.

| Field Name | Type | Description | Example Values |
| :--- | :--- | :--- | :--- |
| `days_since_last_recharge` | int | Number of days elapsed since the customer's last recharge event (recency) | `5`, `14`, `30` |
| `recharge_count_30d` | int | Total count of recharge events in the last 30 days (frequency) | `1`, `3`, `8` |
| `total_recharge_30d` | float | Cumulative monetary value of recharges in the last 30 days | `500.0`, `1500.0`, `4500.0` |
| `avg_recharge_amount` | float | Arithmetic mean value of recharge events | `250.0`, `500.0`, `1000.0` |
| `recharge_trend` | float | Statistical slope of recharge amounts over time (negative indicates declining spend) | `-0.25`, `0.05`, `1.20` |
| `data_usage_mb_30d` | float | Cumulative data consumption in megabytes over the last 30 days | `1024.0`, `4500.0`, `15360.0` |
| `data_usage_trend` | float | Statistical slope of data usage volume over time | `-12.5`, `0.0`, `45.2` |
| `arpu` | float | Average Revenue Per User (monthly spend metric) | `29.85`, `75.50`, `115.0` |
| `tenure_months` | int | Calculated customer lifespan duration in months | `6`, `24`, `72` |
| `churn_risk_score` | float | Estimated model-predicted probability that the customer will churn (0 to 1) | `0.12`, `0.45`, `0.87` |
| `clv_90d` | float | Predicted customer lifetime value (revenue) over the next 90 days | `90.0`, `225.0`, `600.0` |
| `clv_tier` | string | Customer segmentation tier based on predicted lifetime value | `"Platinum"`, `"Gold"`, `"Silver"`, `"Bronze"` |
| `rfm_segment` | string | Customer segment label derived from Recency, Frequency, and Monetary values | `"Champions"`, `"About to Sleep"`, `"At Risk"` |
| `ml_segment` | string | Customer cluster label produced by the K-Means clustering algorithm | `"Cluster_0"`, `"Cluster_1"`, `"Cluster_5"` |
| `propensity_bundle` | float | Model-estimated probability of the customer purchasing a service bundle | `0.15`, `0.68`, `0.92` |
| `propensity_topup` | float | Model-estimated probability of the customer responding to a top-up offer | `0.08`, `0.45`, `0.77` |
| `propensity_reactivation` | float | Model-estimated probability of reactivating a dormant customer | `0.02`, `0.15`, `0.58` |
