'''
Runs the Levin Tree Search strategy computation, independent of plotting or
file output. Shared by main.py (CLI) and app.py (Streamlit dashboard).

Prerequisites
-------------
Run data_pipeline.ipynb first to fetch raw race data from OpenF1 and to
generate the parameter and model artefacts in data/.
'''

from core.get_parameters import (
    get_median_pit_loss, get_degradation_model,
    get_per_lap_temperatures, get_traffic_penalties,
)
from core.search import LevinTreeSearch
from core.race_log import extract_sainz_race_log, generate_race_log
from core.F1State import F1State

TOTAL_LAPS = 58


def run_search() -> dict:
    '''
    Runs the Levin Tree Search (with a pruning-threshold sweep) and extracts
    Sainz's real-world race log as a baseline for comparison.

    Returns a dict with:
        path_levin, path_sainz : list[dict]  — per-lap race logs
        cost, expanded         : float, int  — unpruned search result
        strategy               : list        — action sequence
        sweep                  : list[dict]  — {threshold, race_time_s, nodes_expanded}
        traffic_penalties      : dict[int, float]
        temperatures           : dict[int, tuple[float, float]]
    '''
    pit_loss          = get_median_pit_loss()
    temperatures      = get_per_lap_temperatures()
    traffic_penalties = get_traffic_penalties()
    tire_model        = get_degradation_model()

    initial_state = F1State(0, 'MEDIUM', 0)

    levin = LevinTreeSearch(
        total_laps=TOTAL_LAPS,
        pit_loss=pit_loss,
        tire_model=tire_model,
        traffic_penalties=traffic_penalties,
        temperatures=temperatures,
    )

    path, path_states, cost, expanded = levin.levin_tree_search(initial_state)
    generate_race_log(path_states, './data/paths/path_levin.json')

    sweep = []
    prob = 1e-4
    while prob < 0.5:
        levin.set_pruning_threshold(prob)
        _, _, sweep_cost, sweep_expanded = levin.levin_tree_search(initial_state)
        sweep.append({
            "threshold": prob,
            "race_time_s": sweep_cost,
            "nodes_expanded": sweep_expanded,
        })
        prob *= 2
    levin.set_pruning_threshold(0)

    extract_sainz_race_log()

    import json
    with open('./data/paths/path_levin.json') as f:
        path_levin = json.load(f)
    with open('./data/paths/path_sainz.json') as f:
        path_sainz = json.load(f)

    return {
        "path_levin": path_levin,
        "path_sainz": path_sainz,
        "cost": cost,
        "expanded": expanded,
        "strategy": path,
        "sweep": sweep,
        "traffic_penalties": traffic_penalties,
        "temperatures": temperatures,
    }
