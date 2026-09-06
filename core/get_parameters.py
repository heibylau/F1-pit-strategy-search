'''
Derives the key parameters used by the search algorithm from pre-fetched
race data stored in data/raw/.
'''

import pandas as pd
import numpy as np


# ── Dirty air penalty constants ─────────────────────────────────────────────

DIRTY_AIR_THRESHOLD = 1.0   # seconds — aerodynamic influence zone boundary
DIRTY_AIR_COEFF     = 0.47  # seconds lost per lap while inside the threshold
DIRTY_AIR_HORIZON   = 15    # max laps to look forward per pit stop


# ── Raw data loaders ──────────────────────────────────────────────────────────

def _load_raw():
    df_pits   = pd.read_json('data/raw/pit-stops.json')
    df_laps   = pd.read_json('data/raw/laps.json')
    df_stints = pd.read_json('data/raw/stints.json')
    return df_pits, df_laps, df_stints


# ── Public API ────────────────────────────────────────────────────────────────

def get_median_pit_loss():
    '''
    Returns the median pit lane time loss (seconds) across the race.
    '''
    df_pits, _, _ = _load_raw()
    return float(df_pits['lane_duration'].median())


def get_max_stint_length():
    '''
    Returns the maximum observed stint length (laps) for each tire compound
    as a plain dict: {compound: max_laps}.
    '''
    _, _, df_stints = _load_raw()
    df_stints = df_stints.copy()
    df_stints['stint_length'] = df_stints['lap_end'] - df_stints['lap_start'] + 1
    return df_stints.groupby('compound')['stint_length'].max().to_dict()


def get_per_lap_temperatures():
    '''
    Returns {lap: (air_temperature, track_temperature)} from the notebook-derived CSV.
    '''
    df = pd.read_csv("data/parameter/per_lap_temperatures.csv")
    return {
        int(row["lap"]): (float(row["air_temperature"]), float(row["track_temperature"]))
        for _, row in df.iterrows()
    }


def get_traffic_penalties():
    '''
    Returns {lap: traffic_penalty} from the notebook-derived CSV. Missing laps default to 0.0 at call sites.
    '''
    df = pd.read_csv("data/parameter/traffic_penalties.csv")
    return {int(row["lap"]): float(row["traffic_penalty"]) for _, row in df.iterrows()}


def get_degradation_model():
    '''
    Builds and returns the tire degradation model as a nested dict:
        tire_model[compound][tire_age] = expected_lap_time (seconds)

    Processing steps
    ----------------
    1. Filter laps — remove nulls, pit-out laps, and top-5% slow laps
       (Safety Car / VSC affected).
    2. Join with stints to compute tire_age per lap.
    3. Median lap time per (compound, tire_age); drop groups with < 5 samples
       unless tire_age ≤ 5.
    4. 3-lap centered rolling average per compound (noise smoothing).
    5. Interpolate missing ages within each compound's observed range.
    6. Recompute normalization: relative_deg = smoothed_time − min per compound.
    '''
    _, df_laps, df_stints = _load_raw()

    # Step 1 — Filter
    valid = df_laps.dropna(subset=['lap_duration']).copy()
    valid = valid[valid['is_pit_out_lap'] == False]
    upper = valid['lap_duration'].quantile(0.95)
    valid = valid[valid['lap_duration'] < upper]

    # Step 2 — Join with stints for tire_age
    rows = []
    for _, lap in valid.iterrows():
        driver, lap_num = lap['driver_number'], lap['lap_number']
        stint = df_stints[
            (df_stints['driver_number'] == driver) &
            (df_stints['lap_start']     <= lap_num) &
            (df_stints['lap_end']       >= lap_num)
        ]
        if stint.empty:
            continue
        stint = stint.iloc[0]
        tire_age = stint['tyre_age_at_start'] + (lap_num - stint['lap_start'])
        if tire_age <= 0:
            continue
        rows.append({
            'compound': stint['compound'],
            'tire_age': int(tire_age),
            'lap_duration': lap['lap_duration'],
        })

    df_deg = pd.DataFrame(rows)

    # Step 3 — Aggregate
    degradation = (
        df_deg
        .groupby(['compound', 'tire_age'], as_index=False)
        .agg(
            expected_lap_time=('lap_duration', 'median'),
            sample_size=('lap_duration', 'count'),
        )
    )
    degradation = degradation[
        (degradation['sample_size'] >= 5) | (degradation['tire_age'] <= 5)
    ]

    # Step 4 — Smooth
    degradation = degradation.sort_values(['compound', 'tire_age']).copy()
    degradation['smoothed_time'] = (
        degradation
        .groupby('compound')['expected_lap_time']
        .transform(lambda x: x.rolling(window=3, min_periods=1, center=True).mean())
    )

    # Step 5 — Interpolate missing ages
    degradation = degradation.set_index(['compound', 'tire_age'])
    degradation = degradation.groupby(level=0).apply(
        lambda g: g.droplevel(0).reindex(
            range(1, int(g.index.get_level_values(1).max()) + 1)
        )
    ).interpolate().reset_index()

    # Step 6 — Normalize
    degradation['base_time'] = (
        degradation.groupby('compound')['smoothed_time'].transform('min')
    )
    degradation['relative_deg'] = (
        degradation['smoothed_time'] - degradation['base_time']
    )

    # Build nested dict
    tire_model = {}
    for _, row in degradation.iterrows():
        comp = row['compound']
        age  = int(row['tire_age'])
        time = float(row['smoothed_time'])
        tire_model.setdefault(comp, {})[age] = time

    return tire_model
