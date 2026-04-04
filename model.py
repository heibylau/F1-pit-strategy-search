'''
This module creates two modified datasets, which are used to fit two different
logistic regression models that will be used to calculating the probability of the action 
taken to reach a node in Levin Tree Search.

The purpose of building these models is to provide a more realistic guiding policy 
that better reflects real-world behaviour during a race. For example, a driver is less 
likely to pit at the very beginning or end of a race, and more likely to pit as their 
stint becomes longer.
'''

import pandas as pd
from sklearn.linear_model import LogisticRegression

df_stints = pd.read_json('./data/raw/stints.json')
df_tire_degradation = pd.read_csv('./data/parameter/tire_degradation_model.csv')


def _build_pit_dataset(total_laps=58):
    '''
    Creates a dataset that is used to train a logistic regression model to see if 
    a driver should continue or pit. 
    '''
    rows = []
    df = df_stints.sort_values(
        ["meeting_key", "session_key", "driver_number", "stint_number"]
    )

    tire_lookup = df_tire_degradation.set_index(["compound", "tire_age"])["expected_lap_time"].to_dict()

    for _, stint in df.iterrows():
        for lap in range(stint.lap_start, stint.lap_end + 1):
            tire_age = stint.tyre_age_at_start + (lap - stint.lap_start)
            action = 0  # continue

            if lap == stint.lap_end and lap < total_laps:
                action = 1  # pit

            is_soft = 1 if stint.compound == "SOFT" else 0
            is_medium = 1 if stint.compound == "MEDIUM" else 0
            is_hard = 1 if stint.compound == "HARD" else 0

            # Lookup expected lap time from tire degradation model
            expected_lap_time = tire_lookup.get((stint.compound, tire_age), None)
            if expected_lap_time is None:
                expected_lap_time = df_tire_degradation[df_tire_degradation["compound"] == stint.compound]["expected_lap_time"].mean()

            rows.append({
                "lap": lap,
                "laps_remaining": total_laps - lap,
                "tire_age": tire_age,
                "compound": stint.compound,
                "is_soft": is_soft,
                "is_medium": is_medium,
                "is_hard": is_hard,
                "expected_lap_time": expected_lap_time,
                "pit": action
            })
    df = pd.DataFrame(rows)
    df.to_csv('./data/model/pit_dataset.csv', index=False)
    return df


def _build_compound_dataset(total_laps=58):
    '''
    Creates a dataset that is used to train a multiclass logistic regression model to see what 
    tire compound a driver should pit far if they decide to pit. 
    '''
    rows = []
    df = df_stints.sort_values(
        ["meeting_key", "session_key", "driver_number", "stint_number"]
    )
    grouped = df.groupby(["meeting_key", "session_key", "driver_number"])
    for _, group in grouped:
        group = group.sort_values("stint_number").reset_index(drop=True)
        for i in range(len(group) - 1):
            current = group.iloc[i]
            next_stint = group.iloc[i + 1]
            lap = current.lap_end
            tire_age = current.tyre_age_at_start + (lap - current.lap_start)

            # skip final lap
            if lap >= total_laps:
                continue

            is_soft_before = 1 if current.compound == "SOFT" else 0
            is_medium_before = 1 if current.compound == "MEDIUM" else 0
            is_hard_before = 1 if current.compound == "HARD" else 0

            rows.append({
                "lap": lap,
                "laps_remaining": total_laps - lap,
                "tire_age": tire_age,
                "compound_before": current.compound,
                "is_soft_before": is_soft_before,
                "is_medium_before": is_medium_before,
                "is_hard_before": is_hard_before,
                "next_compound": next_stint.compound,
            })
    df = pd.DataFrame(rows)
    df.to_csv('./data/model/pit_compound_dataset.csv', index=False)
    return df


def create_regression_models():
    '''
    Fits the datasets into two logistic regression models: one for predicting 
    pit decisions and another for predicting the next tire compound choice.

    The "newton-cholesky" solver is chosen because it is efficient when the 
    number of samples is much larger than the number of features.
    '''
    df_pit = _build_pit_dataset()
    df_compound = _build_compound_dataset()

    X_pit = df_pit.drop(columns=["compound", "pit", "is_soft"])
    y_pit = df_pit["pit"]

    model_pit = LogisticRegression(solver="newton-cholesky", max_iter=1000)
    model_pit.fit(X_pit, y_pit)

    X_comp = df_compound.drop(columns=["compound_before", "next_compound", "is_soft_before"])
    y_comp = df_compound["next_compound"]

    model_comp = LogisticRegression(solver="newton-cholesky", max_iter=1000)
    model_comp.fit(X_comp, y_comp)
    return model_pit, model_comp