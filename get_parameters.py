'''
This module extracts the key parameters from raw race data:
    - Median pit lane time loss (pit stop cost)
    - Maximum observed stint length per tire compound
    - Tire degradation model by compound and tire age
'''

import pandas as pd

df_pits = pd.read_json('data/raw/pit-stops.json')
df_laps = pd.read_json('data/raw/laps.json')
df_stints = pd.read_json('data/raw/stints.json')


def get_median_pit_loss():
    '''
    Gets the median pit lane time loss.
    '''
    median_pit_loss = df_pits['lane_duration'].median()
    return median_pit_loss


def get_max_stint_length():
    '''
    Determines the maximum observed stint length for each tire compound.
    '''
    df_stints['stint_length'] = df_stints['lap_end'] - df_stints['lap_start'] + 1
    max_stints = df_stints.groupby('compound')['stint_length'].max()
    max_stints.reset_index().to_csv('data/parameter/max_stint_lengths.csv', index=False)
    return max_stints.to_dict()


def smooth_and_normalize(degradation):
    '''
    Applies smoothing and normalization to degradation data.
    Lap times are smoothed using rolling average.
    '''
    degradation = degradation.sort_values(['compound', 'tire_age']).copy()
    # Applies rolling average
    degradation['smoothed_time'] = (
        degradation
        .groupby('compound')['expected_lap_time']
        .transform(lambda x: x.rolling(window=3, min_periods=1, center=True).mean())
    )
    # Normalization
    degradation['base_time'] = (
        degradation
        .groupby('compound')['smoothed_time']
        .transform('min')
    )
    degradation['relative_deg'] = (
        degradation['smoothed_time'] - degradation['base_time']
    )
    return degradation


def get_degradation_model():
    '''
    Generates tire degradation model from lap and stint data.
    '''
    # Filters out pit-out laps and Safety Car / VSC laps
    valid_laps = df_laps.dropna(subset=['lap_duration'])
    valid_laps = valid_laps[valid_laps['is_pit_out_lap'] == False]
    upper_bound = valid_laps['lap_duration'].quantile(0.95)
    valid_laps = valid_laps[valid_laps['lap_duration'] < upper_bound]

    merged_data = []
    for _, lap in valid_laps.iterrows():
        driver = lap['driver_number']
        lap_num = lap['lap_number']
        stint = df_stints[
            (df_stints['driver_number'] == driver) &
            (df_stints['lap_start'] <= lap_num) &
            (df_stints['lap_end'] >= lap_num)
        ]
        if stint.empty:
            continue
        stint = stint.iloc[0]
        tire_age = stint['tyre_age_at_start'] + (lap_num - stint['lap_start'])
        if tire_age <= 0:
            continue
        merged_data.append({
            'compound': stint['compound'],
            'tire_age': int(tire_age),
            'lap_duration': lap['lap_duration']
        })
    df_deg = pd.DataFrame(merged_data)
    degradation = (
        df_deg
        .groupby(['compound', 'tire_age'], as_index=False)
        .agg(
            expected_lap_time=('lap_duration', 'median'),
            sample_size=('lap_duration', 'count')
        )
    )

    # Removes data with sample size < 5 unless it's softs
    degradation = degradation[(degradation['sample_size'] >= 5) | (degradation['tire_age'] <= 5)]

    # Applies smoothing and normalization
    degradation = smooth_and_normalize(degradation)
    
    # Interpolation
    degradation = degradation.set_index(['compound', 'tire_age'])
    degradation = degradation.groupby(level=0).apply(
        lambda x: x.droplevel(0).reindex(
            range(1, int(x.index.get_level_values(1).max()) + 1)
        )
    )
    degradation = degradation.interpolate().reset_index()

    # Recompute normalization
    degradation['base_time'] = (
        degradation.groupby('compound')['smoothed_time']
        .transform('min')
    )
    degradation['relative_deg'] = (
        degradation['smoothed_time'] - degradation['base_time']
    )

    tire_model = {}
    for _, row in degradation.iterrows():
        comp = row['compound']
        age = int(row['tire_age'])
        time = float(row['smoothed_time'])

        if comp not in tire_model:
            tire_model[comp] = {}

        tire_model[comp][age] = time

    degradation.to_csv('data/parameter/tire_degradation_model.csv', index=False)
    return tire_model