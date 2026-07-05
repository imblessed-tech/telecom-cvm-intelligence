import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

class CVMPreprocessor:
    def __init__(self):
        self.is_fitted = False
        self.feature_stats = {} #stores mean/std for normalisation reference

    def fit_transform(self, ibm_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
        """Fit preprocessor and transform data"""
        if self.is_fitted:
            raise RuntimeError("Preprocessor already fitted")
        
        logger.info("Starting preprocessing pipeline")
        
        ibm_cleaned = self._clean_ibm(ibm_df)
        events_agg = self._aggregate_events(events_df)
        joined_df = self._join_data(ibm_cleaned, events_agg)
        encoded_df = self._encode_categoricals(joined_df)
        cleaned_df = self._handle_missing(encoded_df)

        self.is_fitted = True
        logger.info("Preprocessing complete")
        
        return cleaned_df

    def _clean_ibm(self, df: pd.DataFrame) -> pd.DataFrame: 
        """Clean IBM churn dataset"""
        df = df.copy()
        binary_columns = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
        yesno_columns = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
        for col in binary_columns:
            df[col] = df[col].map({"Yes": 1, "No": 0})
        for col in yesno_columns:
            df[col] = df[col].map({"Yes": 1, "No": 0, "No internet service": 0})
        df = df.rename(columns={
            "customerID": "customer_id",
            "tenure": "tenure_months",
            "MonthlyCharges": "monthly_charges",
            "TotalCharges": "total_charges",
            "Churn_binary": "churn_label"
        })
        df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
        df["monthly_charges"] = pd.to_numeric(df["monthly_charges"], errors="coerce")
        df = df.drop(columns=["CustomerID"], errors="ignore").reset_index(drop=True)
        logger.info("Cleaned IBM churn data")
        return df

    def _aggregate_events(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate events data"""
        df = events_df.copy()
        
        if 'customer_id' not in df.columns:
            raise ValueError("customer_id column not found in events_df")
            
        # Ensure numeric columns are properly typed to prevent categorical encoding issues
        df['recharge_amount'] = pd.to_numeric(df['recharge_amount'], errors='coerce')
        df['data_mb_used'] = pd.to_numeric(df['data_mb_used'], errors='coerce')
        df['call_duration_minutes'] = pd.to_numeric(df['call_duration_minutes'], errors='coerce')
        
        # Ensure event_date is datetime
        df['event_date'] = pd.to_datetime(df['event_date'])
        
        # End and start dates of the dataset
        max_date = df['event_date'].max()
        min_date = df['event_date'].min()
        
        # Cutoff for last 30 days
        cutoff_30d = max_date - pd.Timedelta(days=30)
        
        # 1. Overall Activity & Channel & Offer features (aggregated for all events)
        df['has_offer'] = df['offer_presented'].notna() & (df['offer_presented'] != '') & (df['offer_presented'] != 'None')
        df['is_accepted'] = (df['offer_accepted'] == True) | (df['offer_accepted'] == 1) | (df['offer_accepted'] == 'True')
        
        overall_stats = df.groupby('customer_id').agg(
            active_days_90d=('event_date', lambda s: s.dt.normalize().nunique()),
            last_activity_date=('event_date', 'max'),
            total_events_count=('customer_id', 'count'),
            offers_presented=('has_offer', 'sum'),
            offers_accepted=('is_accepted', 'sum')
        ).reset_index()
        
        overall_stats['days_since_last_activity'] = (max_date - overall_stats['last_activity_date']).dt.days
        overall_stats['offer_acceptance_rate'] = (overall_stats['offers_accepted'] / overall_stats['offers_presented']).fillna(0.0)
        
        # App usage rate: count of APP channel / total events
        app_events = df[df['channel'] == 'APP'].groupby('customer_id').size().rename('app_events')
        overall_stats = overall_stats.merge(app_events, on='customer_id', how='left').fillna({'app_events': 0})
        overall_stats['app_usage_rate'] = overall_stats['app_events'] / overall_stats['total_events_count']
        overall_stats['app_usage_rate'] = overall_stats['app_usage_rate'].fillna(0.0)
        
        # Preferred channel: mode of channel column
        channel_counts = df.groupby(['customer_id', 'channel']).size().reset_index(name='count')
        preferred_channel = channel_counts.sort_values(['customer_id', 'count'], ascending=[True, False])\
                                          .groupby('customer_id').first().reset_index()
        overall_stats = overall_stats.merge(preferred_channel[['customer_id', 'channel']], on='customer_id', how='left')
        overall_stats = overall_stats.rename(columns={'channel': 'preferred_channel'})
        
        # 2. Recharge aggregations (filter event_type == "RECHARGE")
        recharges = df[df['event_type'] == 'RECHARGE'].copy()
        recharge_features = pd.DataFrame({'customer_id': overall_stats['customer_id'].unique()})
        
        if len(recharges) > 0:
            recharges_30d = recharges[recharges['event_date'] >= cutoff_30d]
            
            recharge_stats_90d = recharges.groupby('customer_id').agg(
                recharge_count_90d=('recharge_amount', 'count'),
                total_recharge_90d=('recharge_amount', 'sum'),
                avg_recharge_amount=('recharge_amount', 'mean'),
                last_recharge_date=('event_date', 'max')
            ).reset_index()
            
            recharge_stats_30d = recharges_30d.groupby('customer_id').agg(
                recharge_count_30d=('recharge_amount', 'count'),
                total_recharge_30d=('recharge_amount', 'sum')
            ).reset_index()
            
            def compute_recharge_trend(group):
                if len(group) < 2:
                    return 0.0
                x = (group['event_date'] - min_date).dt.days.values
                y = group['recharge_amount'].values
                try:
                    slope, _ = np.polyfit(x, y, 1)
                    return float(slope)
                except:
                    return 0.0
            
            recharge_trends = recharges.groupby('customer_id').apply(
                lambda g: compute_recharge_trend(g)
            ).rename('recharge_trend').reset_index()
            
            recharge_stats_90d['days_since_last_recharge'] = (max_date - recharge_stats_90d['last_recharge_date']).dt.days
            
            recharge_features = recharge_features.merge(recharge_stats_90d, on='customer_id', how='left')
            recharge_features = recharge_features.merge(recharge_stats_30d, on='customer_id', how='left')
            recharge_features = recharge_features.merge(recharge_trends, on='customer_id', how='left')
            
        # 3. Data usage aggregations (filter event_type == "DATA_SESSION")
        data_sessions = df[df['event_type'] == 'DATA_SESSION'].copy()
        data_features = pd.DataFrame({'customer_id': overall_stats['customer_id'].unique()})
        
        if len(data_sessions) > 0:
            data_sessions_30d = data_sessions[data_sessions['event_date'] >= cutoff_30d]
            
            data_stats_90d = data_sessions.groupby('customer_id').agg(
                data_usage_mb_90d=('data_mb_used', 'sum'),
                avg_session_mb=('data_mb_used', 'mean')
            ).reset_index()
            
            data_stats_30d = data_sessions_30d.groupby('customer_id').agg(
                data_usage_mb_30d=('data_mb_used', 'sum')
            ).reset_index()
            
            def compute_data_usage_trend(group):
                if len(group) == 0:
                    return 0.0
                daily = group.groupby(group['event_date'].dt.normalize())['data_mb_used'].sum().reset_index()
                if len(daily) < 2:
                    return 0.0
                x = (daily['event_date'] - min_date).dt.days.values
                y = daily['data_mb_used'].values
                try:
                    slope, _ = np.polyfit(x, y, 1)
                    return float(slope)
                except:
                    return 0.0
            
            data_trends = data_sessions.groupby('customer_id').apply(
                lambda g: compute_data_usage_trend(g)
            ).rename('data_usage_trend').reset_index()
            
            data_features = data_features.merge(data_stats_90d, on='customer_id', how='left')
            data_features = data_features.merge(data_stats_30d, on='customer_id', how='left')
            data_features = data_features.merge(data_trends, on='customer_id', how='left')
            
        # 4. Merge all components
        final_df = overall_stats.merge(recharge_features, on='customer_id', how='left')
        final_df = final_df.merge(data_features, on='customer_id', how='left')
        
        fill_values = {
            'recharge_count_30d': 0,
            'recharge_count_90d': 0,
            'total_recharge_30d': 0.0,
            'total_recharge_90d': 0.0,
            'avg_recharge_amount': 0.0,
            'days_since_last_recharge': 90,
            'recharge_trend': 0.0,
            'data_usage_mb_30d': 0.0,
            'data_usage_mb_90d': 0.0,
            'data_usage_trend': 0.0,
            'avg_session_mb': 0.0,
            'offers_presented': 0,
            'offers_accepted': 0,
            'offer_acceptance_rate': 0.0,
            'app_usage_rate': 0.0,
            'preferred_channel': 'USSD'
        }
        final_df = final_df.fillna(value=fill_values)
        
        int_cols = [
            'recharge_count_30d', 'recharge_count_90d', 'days_since_recency',
            'offers_presented', 'offers_accepted', 'active_days_90d', 'days_since_last_activity',
            'days_since_last_recharge'
        ]
        for col in int_cols:
            if col in final_df.columns:
                final_df[col] = final_df[col].astype(int)
        
        cols_to_keep = [
            'customer_id',
            'recharge_count_30d',
            'recharge_count_90d',
            'total_recharge_30d',
            'total_recharge_90d',
            'avg_recharge_amount',
            'days_since_last_recharge',
            'recharge_trend',
            'data_usage_mb_30d',
            'data_usage_mb_90d',
            'data_usage_trend',
            'avg_session_mb',
            'preferred_channel',
            'app_usage_rate',
            'offers_presented',
            'offers_accepted',
            'offer_acceptance_rate',
            'active_days_90d',
            'days_since_last_activity'
        ]
        
        return final_df[cols_to_keep]

    def _join_data(self, ibm_df: pd.DataFrame, events_agg: pd.DataFrame) -> pd.DataFrame:
        """Join IBM churn data and aggregated events data on customer_id"""
        joined_df = ibm_df.merge(events_agg, on="customer_id", how="left")
        no_events = joined_df[joined_df['recharge_count_90d'].isna()]
        logger.info(f"Customers with no event records: {len(no_events)}")
        return joined_df
    
    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features with custom prefixes and column mappings"""
        df = df.copy()
        
        # 1. gender -> gender_male (binary, drop one)
        if 'gender' in df.columns:
            gender_lower = df['gender'].str.lower()
            gender_dummies = pd.get_dummies(gender_lower, prefix='gender', dtype=int)
            if 'gender_male' in gender_dummies.columns:
                df['gender_male'] = gender_dummies['gender_male']
            else:
                df['gender_male'] = 0
            df = df.drop(columns=['gender'])
            
        # 2. InternetService -> internet_dsl, internet_fiber, internet_none
        if 'InternetService' in df.columns:
            internet_mapped = df['InternetService'].str.lower().replace({
                'fiber optic': 'fiber',
                'no': 'none'
            })
            internet_dummies = pd.get_dummies(internet_mapped, prefix='internet', dtype=int)
            for col in ['internet_dsl', 'internet_fiber', 'internet_none']:
                if col in internet_dummies.columns:
                    df[col] = internet_dummies[col]
                else:
                    df[col] = 0
            df = df.drop(columns=['InternetService'])
            
        # 3. Contract -> contract_monthly, contract_1yr, contract_2yr
        if 'Contract' in df.columns:
            contract_mapped = df['Contract'].str.lower().replace({
                'month-to-month': 'monthly',
                'one year': '1yr',
                'two year': '2yr'
            })
            contract_dummies = pd.get_dummies(contract_mapped, prefix='contract', dtype=int)
            for col in ['contract_monthly', 'contract_1yr', 'contract_2yr']:
                if col in contract_dummies.columns:
                    df[col] = contract_dummies[col]
                else:
                    df[col] = 0
            df = df.drop(columns=['Contract'])
            
        # 4. PaymentMethod -> four binary columns
        if 'PaymentMethod' in df.columns:
            payment_mapped = df['PaymentMethod'].str.lower().replace({
                'electronic check': 'electronic_check',
                'mailed check': 'mailed_check',
                'bank transfer (automatic)': 'bank_transfer',
                'credit card (automatic)': 'credit_card'
            })
            payment_dummies = pd.get_dummies(payment_mapped, prefix='payment_method', dtype=int)
            expected_payments = [
                'payment_method_electronic_check', 'payment_method_mailed_check',
                'payment_method_bank_transfer', 'payment_method_credit_card'
            ]
            for col in expected_payments:
                if col in payment_dummies.columns:
                    df[col] = payment_dummies[col]
                else:
                    df[col] = 0
            df = df.drop(columns=['PaymentMethod'])
            
        # 5. preferred_channel -> preferred_channel_ussd, preferred_channel_app, preferred_channel_agent, preferred_channel_web
        if 'preferred_channel' in df.columns:
            channel_mapped = df['preferred_channel'].str.lower()
            channel_dummies = pd.get_dummies(channel_mapped, prefix='preferred_channel', dtype=int)
            expected_channels = [
                'preferred_channel_ussd', 'preferred_channel_app',
                'preferred_channel_agent', 'preferred_channel_web'
            ]
            for col in expected_channels:
                if col in channel_dummies.columns:
                    df[col] = channel_dummies[col]
                else:
                    df[col] = 0
            df = df.drop(columns=['preferred_channel'])
            
        # 6. MultipleLines -> standard drop_first dummy columns
        if 'MultipleLines' in df.columns:
            multiple_dummies = pd.get_dummies(df['MultipleLines'], prefix='MultipleLines', drop_first=True, dtype=int)
            df = pd.concat([df, multiple_dummies], axis=1)
            df = df.drop(columns=['MultipleLines'])
            
        # 7. Drop raw Churn column to avoid duplicate label features (churn_label is used)
        if 'Churn' in df.columns:
            df = df.drop(columns=['Churn'])
            
        return df

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        df[non_numeric_cols] = df[non_numeric_cols].fillna('Unknown')
        #Log how many nulls were filled per column
        for col in numeric_cols:
            logger.info(f"Nulls filled in {col}: {df[col].isna().sum()}")
        for col in non_numeric_cols:
            logger.info(f"Nulls filled in {col}: {df[col].isna().sum()}")
        return df

    def save(self, path: Path) -> None: 
        """Save the preprocessor to a file"""
        joblib.dump(self, path)
        logger.info(f"Preprocessor saved to {path}")

    @classmethod
    def load(cls, path: Path) -> 'CVMPreprocessor':
        """Load a preprocessor from a file"""
        logger.info(f"Loading preprocessor from {path}")
        return joblib.load(path)
