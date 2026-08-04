import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# np.random included, not from random library

WEEKS = 4
INTERVAL_MIN = 30
DAILY_MARKS = 48 # 24 hours * 2 marks per hour

# Traffic ranges !mb
QUIET_RANGE = (10, 200)    # 12am-6am
NORMAL_RANGE = (200, 2000) # Evening
PEAK_RANGE = (2000, 10000) # Business hours !9am-5pm

def get_traffic_value(dt):
    hour = dt.hour + dt.minute / 60
    is_weekend = dt.weekday() >= 5 # 5 is Saturday, 6 is Sunday
    
    # Original logic for time of day pattern
    if hour < 6:
        val = np.random.uniform(*QUIET_RANGE)
    elif hour < 9:
        # Ramping up to peak
        t = (hour - 6) / 3
        val = QUIET_RANGE[1] + t * (PEAK_RANGE[0] - QUIET_RANGE[1])
    elif hour < 17:
        # Business hour peaks
        val = np.random.uniform(*PEAK_RANGE)
    elif hour < 19:
        # Ramping down
        t = (hour - 17) / 2
        val = PEAK_RANGE[0] + (1 - t) * (NORMAL_RANGE[1] - PEAK_RANGE[0])
    else:
        val = np.random.uniform(NORMAL_RANGE[0], NORMAL_RANGE[1])
    
    # Apply weekend reduction
    if is_weekend:
        val = val * 0.15 
        
    return round(val, 2)

def generate_network_data(output_file="network_traffic_data.csv"):
    start_date = datetime(2025, 1, 6) # This is monday
    timestamps = [start_date + timedelta(minutes=i * INTERVAL_MIN) 
                  for i in range(WEEKS * 7 * DAILY_MARKS)]
    # https://docs.python.org/3/library/datetime.html#timedelta-objects

    traffic_values = [get_traffic_value(ts) for ts in timestamps]

    df = pd.DataFrame({
        "timestamp": timestamps,
        "traffic_mb": traffic_values
    })

    df.to_csv(output_file, index=False)
    print(f"Data successfully generated and saved to {output_file}")
    return df

if __name__ == "__main__":
    generated_df = generate_network_data()
    print(generated_df.head(10))