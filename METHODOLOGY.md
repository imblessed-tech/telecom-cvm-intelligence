# CVM Intelligence Platform - Methodology & Analytical Foundations

This document provides a comprehensive breakdown of the methodology, statistical concepts, data engineering approaches, and machine learning models implemented in the Customer Value Management (CVM) Intelligence Platform.

---

## 1. GA4-to-CVM Feature Mapping

In modern digital analytics, Google Analytics 4 (GA4) event streams and Telecom Customer Value Management (CVM) transactional database tables share an identical clickstream-to-event structure. Both schemas track user actions over time, capture event-specific attributes, record timestamps, and aggregate raw logs into customer-centric behavioral profiles.

The table below maps the structural equivalents between GA4 digital event parameters and Telecom CVM database equivalents:

| Google Analytics 4 (GA4) Element | Telecom CVM Counterpart | Business Analytics Context |
| :--- | :--- | :--- |
| `session_start` | USSD Session Start / App Launch | Initiating contact or access channel |
| `page_view` | Menu View / Balance Inquiry | Content browsing or service consideration |
| `purchase` | Recharge Event / Bundle Purchase | Direct monetization conversion |
| `user_id` / `client_id` | MSISDN / Customer ID | Unique subscriber identity key |
| `scroll` / `click` | Outgoing Call / SMS Sent | Feature interaction and active usage |
| `traffic_source` (utm_source) | Acquisition Channel (Dealer/Online) | Marketing source of subscriber enrollment |
| `user_properties` | Demographic Flags (gender, contract) | Static or slow-moving customer attributes |
| `custom_dimensions` | Device Type / Network Generation | Device attributes (2G/3G/4G/5G, OS) |
| `event_params` (value, item_name) | Recharge Amount / Data Volume MB | Quantitative event parameters |
| `active_users` (1d/7d/30d active) | MAU / 30-Day Active Subscriber | Monthly active status confirmation |

### Transferability of GA4 Analytics Experience to CVM Engineering

Experience working with GA4 event-level exports (via BigQuery) transfers directly to Telecom CVM feature engineering. Both systems require parsing raw event logs, extracting time-series intervals, and rolling up transaction details into a structured feature vector per customer. In GA4, analyst queries calculate session counts, average session durations, count event trigger rates, and track conversion funnels. In the CVM space, this translates to calculating recharge frequencies, average recharge values, data session counts, data volumes consumed, and tracking service usage trends.

Furthermore, the statistical challenge of modeling customer churn and product propensity is identical across both domains. The GA4 concept of a "lapsed user" (a user who has not triggered an event in 14 days) corresponds exactly to the CVM definition of "dormancy" (no recharge in 20 days). Optimizing funnel conversion thresholds, mapping user flow drop-offs, and grouping behavioral patterns using unsupervised clustering models are highly transferable skills that align Google Analytics methodologies with the goals of telecom marketing teams.

---

## 2. Dataset

The platform is designed around the widely recognized **IBM Telco Churn Dataset**, containing demographic and billing profiles for $7,043$ subscribers. 
- **Demographics**: Gender, Senior Citizen status, Partner, and Dependents.
- **Contract & Billing**: Tenure (months), Contract type (Month-to-month, One year, Two year), Payment Method, Monthly Charges, and Total Charges.
- **Imbalance**: Approximately $26.5\%$ of the customer base has churned (`Churn = Yes`), representing a standard class imbalance of $3:1$ active-to-churned subscribers.

### Event Log Supplementation
Because the raw IBM dataset only provides static profiles, we supplemented it with a generated **Synthetic Customer Events Log** containing over $70,000$ transactional logs spanning $90$ days. This transactional dataset simulates real-world subscriber events:
- **RECHARGE**: Top-up transactions containing `recharge_amount` and `event_date`.
- **DATA_SESSION**: Mobile internet sessions recording `data_mb_used`.
- **COMPLAINT**: Customer care complaints logging customer dissatisfaction.
- **APP_LOGIN**: Logins to the self-care portal, tracking self-service digital adoption.

By combining the demographic profiles with aggregated transactional behavior, we created a comprehensive feature store capable of feed-forward modeling across churn, segmentation, propensity, and customer lifetime value.

---

## 3. Churn Model — XGBoost

For predicting binary customer churn (`churn_label` = 0 or 1), the platform implements an **XGBoost (Extreme Gradient Boosting)** Classifier.

### Model Selection
XGBoost is selected over alternative models like Logistic Regression and Random Forests due to:
- **Non-linear Relations**: XGBoost handles complex, non-linear feature interactions natively without requiring manual interaction terms.
- **Mixed Data Types**: Naturally handles a mix of binary categorical flags, numerical features, and highly correlated variables.
- **Sparsity**: Efficiently processes sparse datasets resulting from one-hot encoding without scaling requirements.

### Addressing Class Imbalance with `scale_pos_weight`
To prevent the model from learning a trivial predictor that favors the majority class (active customers), we configure the `scale_pos_weight` parameter:
$$\text{scale\_pos\_weight} = \frac{N_{\text{negative}}}{N_{\text{positive}}}$$
This penalizes misclassifications on the minority churn class proportionally, forcing the tree booster to prioritize high-risk signals.

### Early Stopping & Generalization
We leverage early stopping by passing a validation split during model training. If the validation loss (LogLoss) does not improve for $10$ consecutive boosting rounds, training stops. This prevents overfitting the trees to noise in the training split.

### Threshold Optimization on F1-Score
In a highly imbalanced setting, raw accuracy is a misleading metric. A model that predicts "no churn" for all customers achieves $73.5\%$ accuracy but is commercially useless. Instead, the platform dynamically calculates prediction thresholds by optimizing the **F1-Score** (harmonic mean of Precision and Recall) on a separate validation partition:
$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
This balances the cost of false positives (unnecessary campaign retention spending) against false negatives (undetected churners who leave the network).

### Business Interpretability with SHAP
Machine learning models are only valuable to the business if they are actionable. We fit a **SHAP (SHapley Additive exPlanations)** TreeExplainer to the model. SHAP uses game-theory payoffs to assign an importance contribution value to each feature for every prediction. The API exposes these SHAP values to return the top 5 risk drivers for any customer, allowing customer service agents to see *why* a customer is flagged as high-risk (e.g., high billing charges combined with recent low data usage) and offer targeted incentives.

---

## 4. Segmentation — K-Means

Unsupervised customer segmentation is implemented using **K-Means Clustering** applied to all engineered customer features.

### Selection of K=6
We select $K=6$ clusters. This is a common industry standard for CVM campaigns: it provides sufficient granularity to separate high-value, dormant, and price-sensitive groups without overcomplicating campaign management for marketing teams.

### Cluster Validation
We validate the optimal cluster count using two metrics:
1. **Elbow Method (WCSS)**: Plotting Within-Cluster Sum of Squares to find the point where adding more clusters yields diminishing returns.
2. **Silhouette Coefficient**: Evaluating cluster cohesion and separation, ensuring that samples are closer to their own cluster center than neighboring ones.

### Cluster Profiles to Heuristic Labels
Once the model is fitted, the platform computes profile averages per cluster. These centroids map directly to CVM campaign archetypes:
- **Champions**: Highest spend, frequent rechargers, high data usage.
- **Price Sensitive**: High recharge frequency but very low recharge amounts.
- **Loyal But Declining**: Long tenure, but show declining recharge trends.
- **Rising Stars**: Low tenure but high data usage growth.
- **Dormant**: Zero active sessions in the last 20+ days.
- **At Risk**: High churn risk score and declining activity.

### Principal Component Analysis (PCA)
Because the feature space is high-dimensional (24+ dimensions), it cannot be plotted directly. We fit **PCA (Principal Component Analysis)** to reduce dimensions to 2 components. PCA is used **only** for visualizing the cluster boundaries in the dashboard (`segment_clusters.png`). The K-Means model itself is trained on the full high-dimensional feature space to preserve behavioral nuances.

---

## 5. Propensity Models

Propensity modeling predicts the probability that a customer will respond positively to specific campaign offer categories.

### Architecture: Three Separate Binary Classifiers
Rather than fitting a single multi-class model (which assumes mutual exclusivity of offers), we train **three separate binary RandomForest classifiers**. This allows us to score propensity for different campaigns simultaneously:
1. **Bundle Upgrade**: Targets data-heavy users who have growth potential.
2. **Voice Top-up**: Targets frequent callers who recharge with low denominations.
3. **Reactivation**: Targets dormant users who have been inactive.

### Event-Driven Label Engineering
Because historical campaign response logs are absent, we engineer labels from behavioral event histories:
- **Bundle Upsell Label**: $1$ if the customer is a high-volume data user, has a positive data usage trend, and a historical conversion rate $> 15\%$; else $0$.
- **Voice Top-up Label**: $1$ if recharge frequency exceeds the median, average top-up is low ($< ₦500$), and recharges occur frequently; else $0$.
- **Reactivation Label**: $1$ if inactivity period exceeds 20 days but historical 90-day spend is above the bottom $25\%$ quantile; else $0$.

### Precision/Recall Tradeoff
When deploying these scores, we configure `PROPENSITY_THRESHOLD = 0.50`. 
- **Higher Threshold**: Targets a smaller, high-probability customer group. This maximizes response precision and keeps campaign costs low, but limits total conversions.
- **Lower Threshold**: Casts a wider net, capturing more total responders (recall) at the expense of higher campaign costs and lower conversion rates.

---

## 6. CLV Prediction

**Customer Lifetime Value (CLV)** models predict the future financial value of a customer, enabling budget allocation decisions.

### Log-Transformation of Revenue Target
Telco recharge spend distributions are highly right-skewed (many low-spend users and a few high-value outliers). To stabilize variance and prevent large recharges from distorting the regressor, we apply a log-transformation:
$$\text{clv\_log} = \ln(\text{clv\_90d} + 1)$$
Predictions from the Gradient Boosting Regressor are back-transformed using the exponential function ($e^x - 1$) before reporting.

### Target Engineering
The target variable captures 90-day recharge values discounted by individual churn probabilities:
$$\text{clv\_90d} = \text{total\_recharge\_90d} \times (1 - \text{churn\_label} \times 0.8)$$
This adjusts historical value to account for the risk of attrition.

### Financial Budget Allocation Tiers
We sort the predicted 90-day CLV values into percentiles to define campaign budgets:
- **Platinum** (Top 10%): High-value. Allocated premium retention budget (up to ₦5,000 per user).
- **Gold** (10% - 25%): Medium-high value. Allocated standard budget (₦2,500).
- **Silver** (25% - 50%): Medium value. Allocated basic budget (₦1,000).
- **Bronze** (Bottom 50%): Low value. Minimal budget (allocated low-cost SMS-only retention offers).

---

## 7. PySpark Implementation

For processing large-scale telecom datasets (millions of daily subscriber records), the platform implements an ETL pipeline in PySpark.

### Scale Rationale & Lazy Evaluation
Pandas is memory-bound and executes commands eagerly, which fails at scale. PySpark uses **lazy evaluation**, building a directed acyclic graph (DAG) of transformations. Computation only triggers when an action (e.g., `.write()` or `.collect()`) is called. This allows the Catalyst Optimizer to optimize physical execution plans, prune columns, and manage partition shuffles efficiently.

### Window Functions for Recharge Trends
Instead of loading entire datasets to run linear regressions (like `np.polyfit`), we compute the slope of recharge amounts over time using **Spark Window Functions**:
$$\text{Slope} = \frac{N \sum (X \cdot Y) - (\sum X)(\sum Y)}{N \sum X^2 - (\sum X)^2}$$
We partition the logs by `customer_id` and order by `event_date` to calculate row indices ($X$) and recharge amounts ($Y$), computing the slope natively inside Spark SQL.

---

## 8. Model Monitoring

To ensure predictive reliability over time, we monitor data and prediction drift.

### Population Stability Index (PSI)
We implement PSI to compare the distribution of features or predictions at training time (expected) against current data (actual) across 10 equal-width bins:
$$\text{PSI} = \sum_{i=1}^{10} (P_i - Q_i) \times \ln\left(\frac{P_i}{Q_i}\right)$$
Where $P_i$ is the actual proportion in bin $i$, and $Q_i$ is the expected proportion.
- **PSI < 0.1**: Stable. No drift.
- **0.1 <= PSI < 0.2**: Moderate drift. Monitor features closely.
- **PSI >= 0.2**: Severe drift. Retraining required.

### Feature Drift vs. Prediction Drift
- **Prediction Drift**: Monitored on the final model output scores (`churn_risk_score`). It acts as a fast indicator of performance decay.
- **Feature Drift**: Monitored across independent features (like `arpu` or `data_usage`). This helps isolate the root cause of drift (e.g., changes in user behavior or data ingestion errors).

---

## 9. Limitations & Production Upgrade Path

While functionally complete for development, a production deployment would include several upgrades:
1. **Experiment Tracking**: Integrating **MLflow** or weights & biases to track hyperparameters, model runs, and version metrics.
2. **Caching Layer**: Integrating **Redis** to cache scored customer profiles, enabling fast sub-millisecond API response times.
3. **Pipeline Orchestration**: Implementing **Apache Airflow** DAGs to schedule daily Spark ETL runs, trigger model monitoring, and run retraining pipelines.
4. **Campaign A/B Testing**: Implementing a control-group framework to measure lift (campaign response vs. control) and optimize propensity thresholds.
