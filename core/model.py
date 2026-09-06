'''
Fits the two logistic regression policy models used by Levin Tree Search.

Both models are trained on data derived from pre-fetched stint data in
data/raw/stints.json and the tire degradation model in
data/parameter/tire_degradation_model.csv.
'''

import warnings
import os

import pandas as pd
from sklearn.linear_model import LogisticRegression


def _load_data():
    df_stints          = pd.read_json('./data/raw/stints.json')
    df_tire_degradation = pd.read_csv('./data/parameter/tire_degradation_model.csv')
    return df_stints, df_tire_degradation


def _build_pit_dataset(df_stints, df_tire_degradation, total_laps=58):
    '''
    One row per lap-in-stint. Binary target `pit`:
        1 — driver pitted at the end of this lap (last lap of a non-final stint)
        0 — driver continued
    '''
    tire_lookup = (
        df_tire_degradation
        .set_index(['compound', 'tire_age'])['expected_lap_time']
        .to_dict()
    )

    temp_lookup    = {}
    traffic_lookup = {}

    if os.path.exists("./data/parameter/per_lap_temperatures.csv"):
        df_temp = pd.read_csv("./data/parameter/per_lap_temperatures.csv")
        temp_lookup = df_temp.set_index("lap")[["air_temperature", "track_temperature"]].to_dict("index")
    else:
        warnings.warn("per_lap_temperatures.csv not found — temperature features omitted.")

    if os.path.exists("./data/parameter/traffic_penalties.csv"):
        df_traf = pd.read_csv("./data/parameter/traffic_penalties.csv")
        traffic_lookup = df_traf.set_index("lap")["traffic_penalty"].to_dict()
    else:
        warnings.warn("traffic_penalties.csv not found — traffic_penalty feature omitted.")

    rows = []
    df = df_stints.sort_values(
        ['meeting_key', 'session_key', 'driver_number', 'stint_number']
    )
    for _, stint in df.iterrows():
        for lap in range(stint.lap_start, stint.lap_end + 1):
            tire_age = stint.tyre_age_at_start + (lap - stint.lap_start)
            action   = 1 if (lap == stint.lap_end and lap < total_laps) else 0

            elt = tire_lookup.get(
                (stint.compound, tire_age),
                df_tire_degradation[
                    df_tire_degradation['compound'] == stint.compound
                ]['expected_lap_time'].mean(),
            )
            rows.append({
                'lap':               lap,
                'laps_remaining':    total_laps - lap,
                'tire_age':          tire_age,
                'compound':          stint.compound,
                'is_soft':           1 if stint.compound == 'SOFT'   else 0,
                'is_medium':         1 if stint.compound == 'MEDIUM' else 0,
                'is_hard':           1 if stint.compound == 'HARD'   else 0,
                'expected_lap_time': elt,
                'air_temperature':  temp_lookup.get(lap, {}).get("air_temperature",  None),
                'track_temperature': temp_lookup.get(lap, {}).get("track_temperature", None),
                'traffic_penalty':  traffic_lookup.get(lap, 0.0),
                'pit':               action,
            })
    return pd.DataFrame(rows)


def _build_compound_dataset(df_stints, total_laps=58):
    '''
    One row per inter-stint transition. Multiclass target `next_compound`:
    the compound the driver switched to at the pit stop.
    '''
    temp_lookup = {}
    if os.path.exists("./data/parameter/per_lap_temperatures.csv"):
        df_temp = pd.read_csv("./data/parameter/per_lap_temperatures.csv")
        temp_lookup = df_temp.set_index("lap")[["air_temperature", "track_temperature"]].to_dict("index")
    else:
        warnings.warn("per_lap_temperatures.csv not found — temperature features omitted.")

    rows = []
    df = df_stints.sort_values(
        ['meeting_key', 'session_key', 'driver_number', 'stint_number']
    )
    for _, group in df.groupby(['meeting_key', 'session_key', 'driver_number']):
        group = group.sort_values('stint_number').reset_index(drop=True)
        for i in range(len(group) - 1):
            cur = group.iloc[i]
            nxt = group.iloc[i + 1]
            lap = cur.lap_end
            if lap >= total_laps:
                continue
            tire_age = cur.tyre_age_at_start + (lap - cur.lap_start)
            rows.append({
                'lap':               lap,
                'laps_remaining':    total_laps - lap,
                'tire_age':          tire_age,
                'compound_before':   cur.compound,
                'is_soft_before':    1 if cur.compound == 'SOFT'   else 0,
                'is_medium_before':  1 if cur.compound == 'MEDIUM' else 0,
                'is_hard_before':    1 if cur.compound == 'HARD'   else 0,
                'air_temperature':  temp_lookup.get(lap, {}).get("air_temperature",  None),
                'track_temperature': temp_lookup.get(lap, {}).get("track_temperature", None),
                'next_compound':     nxt.compound,
            })
    return pd.DataFrame(rows)


def create_regression_models():
    '''
    Fits and returns both policy models.

    Returns
    -------
    (model_pit, model_comp) : tuple of fitted logistic regression models
    '''
    df_stints, df_tire_degradation = _load_data()

    df_pit      = _build_pit_dataset(df_stints, df_tire_degradation)
    df_compound = _build_compound_dataset(df_stints)

    # Pit decision model (binary)
    X_pit = df_pit.drop(columns=['compound', 'pit', 'is_soft'])
    y_pit = df_pit['pit']
    # Fill any missing temperature/traffic values with column medians so no rows are dropped
    X_pit = X_pit.fillna(X_pit.median(numeric_only=True))
    model_pit = LogisticRegression(solver='newton-cholesky', max_iter=1000)
    model_pit.fit(X_pit, y_pit)

    # Compound choice model (multiclass)
    X_comp = df_compound.drop(columns=['compound_before', 'next_compound', 'is_soft_before'])
    y_comp = df_compound['next_compound']
    X_comp = X_comp.fillna(X_comp.median(numeric_only=True))
    model_comp = LogisticRegression(solver='newton-cholesky', max_iter=1000)
    model_comp.fit(X_comp, y_comp)

    return model_pit, model_comp
