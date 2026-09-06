'''
Entry point for the F1 pit stop strategy search.

Prerequisites
-------------
Run notebooks/data_pipeline.ipynb first to fetch the raw race data from
OpenF1 and to generate the parameter and model artefacts in data/.

Usage
-----
    python main.py

Outputs
-------
data/paths/path_levin.json   — Levin Tree Search strategy
'''

from pipeline import run_search
from dashboard.plots import build_gapper_plot, build_stint_chart

result = run_search()

print('Running Levin Tree Search (no pruning):')
print(f'  Total race time : {result["cost"]:.3f}s')
print(f'  Nodes expanded  : {result["expanded"]}')
print(f'  Strategy        : {result["strategy"]}\n')

print('Pruning threshold sweep:')
print(f'  {"Threshold":>12}  {"Race time (s)":>14}  {"Nodes expanded":>15}')
print('  ' + '-' * 45)
for row in result['sweep']:
    print(f'  {row["threshold"]:>12.4f}  {row["race_time_s"]:>14.3f}  {row["nodes_expanded"]:>15}')

gapper_fig = build_gapper_plot(
    result['path_levin'], result['path_sainz'], result['traffic_penalties']
)

stint_fig = build_stint_chart(result['path_levin'], result['path_sainz'])
