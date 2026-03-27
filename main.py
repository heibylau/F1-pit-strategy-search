from get_parameters import get_median_pit_loss, get_max_stint_length, get_degradation_model
from search import AStar, LevinTreeSearch
from generate_race_log import generate_race_log
from F1State import F1State

initial_state = F1State(0, "MEDIUM", 0)

astar = AStar(
    total_laps=58,
    pit_loss=get_median_pit_loss(),
    max_stints=get_max_stint_length(),
    tire_model=get_degradation_model()
)


path, path_states, cost, expanded = astar.a_star(initial_state)
generate_race_log(path_states, "./data/paths/path_a_star.json")
print(f"Found solution with cost {cost} and expanded {expanded} nodes.")
print(path)


levin = LevinTreeSearch(
    total_laps=58, 
    pit_loss=get_median_pit_loss(), 
    tire_model=get_degradation_model()
)

path, path_states, cost, expanded = levin.levin_tree_search(initial_state)
generate_race_log(path_states, "./data/paths/path_levin.json")
print(f"Found solution with cost {cost} and expanded {expanded} nodes.")
print(path)