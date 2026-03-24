# -----------------------------------------------------------------------
# This module extracts and cleans Carlos Sainz's lap and stint data
# from the OpenF1 API for the 2024 Australian Grand Prix.
#
# The output is a simplified JSON log containing:
#   - lap number
#   - tire compound
#   - tire age
#   - cumulative total time
#
# Carlos Sainz won the 2024 Australian Grand Prix, so his real-world
# strategy serves as a reference for this project.
# -----------------------------------------------------------------------

import pandas as pd
import json

with open('./data/raw/laps.json') as f:
    df_laps = pd.DataFrame(json.load(f))
with open('./data/raw/stints.json') as f:
    df_stints = pd.DataFrame(json.load(f))

driver_number = 55  # Driver number for Carlos Sainz

driver_laps = df_laps[df_laps['driver_number'] == driver_number].sort_values('lap_number')
driver_stints = df_stints[df_stints['driver_number'] == driver_number]

states = []
total_time = 0.0

for _, lap_row in driver_laps.iterrows():
    lap = lap_row['lap_number']
    
    if pd.isna(lap_row['lap_duration']):
        continue
        
    total_time += lap_row['lap_duration']
    
    stint = driver_stints[(driver_stints['lap_start'] <= lap) & (driver_stints['lap_end'] >= lap)]
    
    if not stint.empty:
        stint = stint.iloc[0]
        compound = stint['compound']
        tire_age = stint['tyre_age_at_start'] + (lap - stint['lap_start'])
    else:
        compound = "UNKNOWN"
        tire_age = 0
        
    states.append({
        'lap': int(lap),
        'compound': compound,
        'tire_age': int(tire_age),
        'total_time': round(total_time, 3)
    })

with open('./data/cleaned/states.json', 'w') as output:
    json.dump(states, output, indent=4)