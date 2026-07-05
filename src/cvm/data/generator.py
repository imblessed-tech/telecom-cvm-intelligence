import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path
from src.cvm.config import settings

logger = logging.getLogger(__name__)

# Constants
RECHARGE_AMOUNTS = [100, 200, 500, 1000, 2000, 5000]  # Naira
CHANNELS = ["USSD", "APP", "AGENT", "WEB"]
CHANNEL_WEIGHTS = [0.5, 0.3, 0.15, 0.05]  # USSD dominates in Nigeria
EVENT_TYPES = ["RECHARGE", "DATA_SESSION", "CALL", "SMS"]
EVENT_WEIGHTS = [0.25, 0.35, 0.30, 0.10]
OFFERS = ["DATA_BUNDLE_1GB", "DATA_BUNDLE_5GB", "VOICE_TOPUP_100",
          "LOYALTY_BONUS", "WEEKEND_DATA", None, None, None] # None appears 3x so most sessions show no offer


class CustomerEventGenerator:
    def __init__(self, n_day: int = 90, start_date: str = "2024-01-01"):
        self.n_day = n_day
        self.start = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = self.start + timedelta(days=n_day)
        self.all_dates = pd.date_range(start=start_date, end=self.end_date, freq='D')
        
    
    def generate(self, customer_ids: list[str], churn_label: dict[str, int]) -> pd.DataFrame:
        """Generate synthetic event logs for a list of customers over a fixed time period."""
        all_events = []
        for customer_id in customer_ids:
            is_churner = churn_label.get(customer_id, 0) == 1
            customer_events = self._generate_customer_events(customer_id, is_churner)
            all_events.append(customer_events)
        return pd.concat(all_events, ignore_index=True).sort_values(["customer_id", "event_date"])
        
    def _generate_customer_events(self, customer_id: str, is_churner: bool) -> pd.DataFrame:
        """Generates synthetic event logs for a single customer."""
        
        seed = hash(customer_id) % (2**31)
        rng = np.random.default_rng(seed)
        
        #1. Get customer preferred/favourite recharge amount in Naira
        recharge_preference_idx = rng.integers(0, len(RECHARGE_AMOUNTS))
        preferred_amount = RECHARGE_AMOUNTS[recharge_preference_idx]

        #2. Get base customer-specific data usage in MB per session
        base_data_usage = rng.uniform(50, 1024)

        #3. Determine customer recharge frequency (days between recharges)
        if is_churner:
            base_frequency = rng.integers(7, 21) #assume churners recharge every 7 - 20 days initially
        else:
            base_frequency = rng.integers(3, 14) #assume loyal customers recharge 3 -13 days
        

        
        if is_churner:
            normal_end_day = settings.NORMAL_PHASE_DAYS   # Days 1-60: normal behaviour
            decline_end_day =settings.DECLINE_PHASE_DAYS    # Days 61-75: declining behaviour  
                                                              # Days 76-90: complete silence (churn window)
            decline_period = decline_end_day - normal_end_day
                    
        events = []
        for day_idx, current_date in enumerate(self.all_dates):
            #4. Decide if the customer is active today
            if is_churner:
                if day_idx < normal_end_day:
                    daily_active_prob = 1 / base_frequency
                elif day_idx < decline_end_day:
                    # Gradually reduce activity in decline phase
                    decline_factor = 1 - ((day_idx - normal_end_day) / decline_period) * 0.75
                    daily_active_prob = (1 / base_frequency) * decline_factor
                else:
                    daily_active_prob = 0.01 #very low chance of activity
            else:
                daily_active_prob = 1 / base_frequency
            
            if rng.random() >= daily_active_prob:
                continue # No activity today, move to next day

            #5. Determine the type of event for the current date
            event_type = rng.choice(EVENT_TYPES, p=EVENT_WEIGHTS) #event types with higher weights are more likely to occur
            
            # Initialize metrics
            recharge_amount = None
            data_mb_used = None
            call_duration_minutes = None
            
            #6.Generate a transaction amount
            if event_type == "RECHARGE":
                #Assume customer use thier preferred recharge amount 75% of the time
                if rng.random() < 0.75:
                    amount = preferred_amount
                else:
                    amount = rng.choice([a for a in RECHARGE_AMOUNTS if a != preferred_amount])
                
                # In declining phase, churners recharge less
                if is_churner and day_idx >= normal_end_day:
                    decline_scale = 1 - ((day_idx - normal_end_day) / decline_period) * 0.5
                    amount = amount * decline_scale
                    amount = round(amount / 100) * 100 #Round to nearest 100 (realistic for Naira)
                    amount = max(100, amount) #Ensure minimum is 100 Naira
                    
                recharge_amount = float(amount)
                data_mb_used = None
                call_duration_minutes = None

            elif event_type == "DATA_SESSION":
                #apply normal distribution to simulate natural variation in user's daily data usage
                daily_variation = rng.normal(loc=1.0, scale=0.3) 
                daily_variation = max(0.1, daily_variation)   # never negative      

                data_mb = base_data_usage * daily_variation
                
                # Heavy users have occasional big sessions
                if base_data_usage > 800 and rng.random() < 0.15:
                    data_mb = data_mb * rng.uniform(2.0, 5.0)   # big streaming session
                
                # In declining phase, churners use less data
                if is_churner and day_idx >= normal_end_day:
                    decline_weeks = (day_idx - normal_end_day) // 7
                    data_mb *= (0.9 ** decline_weeks)
                
                recharge_amount = None
                data_mb_used = round(float(data_mb), 2)
                call_duration_minutes = None
                
            elif event_type == "CALL":
                # Most calls are short, few are very long
                duration = rng.exponential(scale=8.0)   # mean of 8 minutes
                duration = max(0.5, min(duration, 120))  # clip to 30 seconds - 2 hours
                
                recharge_amount = None
                data_mb_used = None
                call_duration_minutes = round(float(duration), 1)

            #6. Generate channerl and offer
            # Channel: weighted random choice
            channel = rng.choice(
                CHANNELS,
                p=CHANNEL_WEIGHTS  
            ) 

            # Offer: shown occasionally during recharges
            offer_presented = None
            offer_accepted = False

            # Special offers for data sessions
            if event_type == "RECHARGE":
                if rng.random() < 0.30:   # 30% of recharges show an offer
                    offer_presented = rng.choice(OFFERS)
                    
                    if offer_presented is not None:
                        # Acceptance probability based on offer relevance
                        base_acceptance = 0.20
                        
                        # Higher data users more likely to accept data offers
                        if "DATA" in str(offer_presented) and base_data_usage > 300:
                            acceptance_prob = base_acceptance * 1.5
                        else:
                            acceptance_prob = base_acceptance
                            
                        offer_accepted = bool(rng.random() < acceptance_prob)
            # Append this event to the list
            events.append({
                "customer_id": customer_id,
                "event_date": current_date.date(),
                "event_type": event_type,
                "recharge_amount": recharge_amount,       # None if not a recharge
                "data_mb_used": data_mb_used,             # None if not a data session
                "call_duration_minutes": call_duration_minutes,  # None if not a call
                "channel": channel,
                "offer_presented": offer_presented,
                "offer_accepted": offer_accepted,
            })

        if not events:
            return pd.DataFrame(columns=[
                "customer_id", "event_date", "event_type", "recharge_amount",
                "data_mb_used", "call_duration_minutes", "channel",
                "offer_presented", "offer_accepted"
            ])

        return pd.DataFrame(events)

    def save(self, df: pd.DataFrame, path: Path) -> None:
        """Save the generated events to a CSV file."""
        df.to_csv(path, index=False)
        logger.info(f"Saved {len(df)} events to {path}")


def generate_events(customer_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Generate events for all customers."""
    customer_df = customer_df.copy()
    if 'Churn_binary' not in customer_df.columns and 'Churn' in customer_df.columns:
        customer_df['Churn_binary'] = customer_df['Churn'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
        
    customer_ids = customer_df['customerID'].tolist()
    churn_label = dict(customer_df[['customerID', 'Churn_binary']].set_index('customerID')['Churn_binary'])
    event_generator = CustomerEventGenerator()
    generated_df = event_generator.generate(customer_ids, churn_label)
    event_generator.save(generated_df, Path(output_path))
    return generated_df

    
    

        
