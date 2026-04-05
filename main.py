'''
This module runs the A* and Levin Tree Search algorithms.

Recovered paths are saved as JSON files in data/paths/ directory.

Visualizations are saved in images/ directory.
'''

import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from get_parameters import get_median_pit_loss, get_max_stint_length, get_degradation_model
from search import AStar, LevinTreeSearch
from race_log import extract_sainz_race_log, generate_race_log
from F1State import F1State


initial_state = F1State(0, "MEDIUM", 0)

astar = AStar(
    total_laps=58,
    pit_loss=get_median_pit_loss(),
    max_stints=get_max_stint_length(),
    tire_model=get_degradation_model()
)

print("Running A*:")
path, path_states, cost, expanded = astar.a_star(initial_state)
generate_race_log(path_states, "./data/paths/path_a_star.json")
print(f"Found solution with total race time of {cost:.3f}s and expanded {expanded} nodes.")
print(f"{path}\n")


levin = LevinTreeSearch(
    total_laps=58, 
    pit_loss=get_median_pit_loss(), 
    tire_model=get_degradation_model()
)

print("Running Levin Tree Search:")
path, path_states, cost, expanded = levin.levin_tree_search(initial_state)
generate_race_log(path_states, "./data/paths/path_levin.json")
print(f"Found solution with total race time of {cost:.3f}s and expanded {expanded} nodes.")
print(path)


#------------------
# Visualization
#------------------
extract_sainz_race_log()  # for baseline reference
with open("./data/paths/path_a_star.json", "r") as f:
    path_a_star = json.load(f)
with open("./data/paths/path_levin.json", "r") as f:
    path_levin = json.load(f)
with open("./data/paths/path_sainz.json", "r") as f:
    path_sainz = json.load(f)

laps = [lap["lap"] for lap in path_a_star]
def lap_times(path):
    return [path[i]["total_time"] - path[i-1]["total_time"] if i > 0 else path[i]["total_time"]
            for i in range(len(path))]

lap_times_a_star = lap_times(path_a_star)
lap_times_levin = lap_times(path_levin)
lap_times_sainz = lap_times(path_sainz)

gaps_a_star = [a["total_time"] - s["total_time"] for a, s in zip(path_a_star, path_sainz)]
gaps_levin = [l["total_time"] - s["total_time"] for l, s in zip(path_levin, path_sainz)]

plt.figure(figsize=(10,6))
plt.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Sainz')  # Horizontal baseline 
plt.plot(laps, gaps_levin, color='green', linewidth=2, label='Levin')
plt.plot(laps, gaps_a_star, color='blue', linewidth=2, label='A*')
plt.xlabel("Lap")
plt.ylabel("Time Difference(s)")
plt.title("Lap-by-Lap Time Gap Relative to Carlos Sainz")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("./images/gapper_plot.png", dpi=300)


a_star_stints = [(1, 30, 'HARD'), (31, 58, 'HARD')]
levin_stints = [(1, 12, 'MEDIUM'), (13, 39, 'HARD'), (40, 58, 'HARD')]
sainz_stints = [(1, 16, 'MEDIUM'), (17, 41, 'HARD'), (42, 58, 'HARD')]

compound_colors = {'MEDIUM': 'yellow', 'HARD': 'lightgray'}
pit_colors = {'A*': 'blue', 'Levin': 'green', 'Sainz': 'red'}

fig, ax = plt.subplots(figsize=(10,4))
for i, (label, stints) in enumerate([('A*', a_star_stints), ('Levin', levin_stints), ('Sainz', sainz_stints)]):
    for start, end, comp in stints:
        ax.broken_barh([(start, end - start + 1)], (i*10, 8), facecolors=compound_colors[comp])
    pit_laps = [start for start, _, _ in stints]
    if label == 'A*':
        pit_laps_to_plot = pit_laps 
    else:
        pit_laps_to_plot = pit_laps[1:]
    for pit in pit_laps_to_plot:
        ax.axvline(x=pit, color=pit_colors[label], linestyle='--', alpha=0.7, linewidth=2)

ax.set_yticks([5, 15, 25])
ax.set_yticklabels(['A*', 'Levin', 'Sainz'])
ax.set_xlabel("Lap")
ax.set_title("Pit Stop Strategy")
legend_lines = [Line2D([0], [0], color=color, lw=2, linestyle='--') for color in ['blue', 'green', 'red']]
ax.legend(legend_lines, ['A* pit', 'Levin pit', 'Sainz pit'], loc='upper right')
plt.tight_layout()
plt.savefig("./images/pit_strategy.png", dpi=300)