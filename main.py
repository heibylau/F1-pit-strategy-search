from get_parameters import get_median_pit_loss, get_max_stint_length, get_degradation_model
from search import AStar
from F1State import F1State


astar = AStar(
    total_laps=58,
    pit_loss=get_median_pit_loss(),
    max_stints=get_max_stint_length(),
    tire_model=get_degradation_model()
)

initial_state = F1State(0, "MEDIUM", 0)
path, cost, expanded = astar.a_star(initial_state)
print(path)
print(cost)