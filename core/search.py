'''
Implements Levin Tree Search for F1 pit stop strategy optimization.

LevinTreeSearch uses two logistic regression policy models (fitted in
notebooks/data_pipeline.ipynb) to guide the search toward realistic
pit strategies while minimizing total race time.
'''

import heapq
import copy
import numpy as np
import pandas as pd
from core.node import LevinNode
from core.F1State import F1State
from core.model import create_regression_models


class LevinTreeSearch:
    '''
    Levin Tree Search for F1 pit stop strategy.

    Parameters
    ----------
    total_laps : int
        Number of laps in the race.
    pit_loss : float
        Median pit lane time loss in seconds (from get_parameters).
    tire_model : dict
        Nested dict tire_model[compound][tire_age] → expected lap time (s).
    pruning_threshold : float
        Actions with probability <= this value are pruned. Default 0 (no pruning).
    traffic_penalties : dict | None
        Optional {lap: penalty} dict of traffic (dirty air) time penalties.
    temperatures : dict | None
        Optional {lap: (air_temp, track_temp)} dict, forwarded to the policy
        models when they were fit with temperature features.
    '''

    def __init__(self, total_laps, pit_loss, tire_model,
                 pruning_threshold=0, traffic_penalties=None, temperatures=None):
        self.model_pit, self.model_comp = create_regression_models()
        self.total_laps = total_laps
        self.pit_loss = pit_loss
        self.tire_model = tire_model
        self.pruning_threshold = pruning_threshold
        self.traffic_penalties = traffic_penalties or {}
        self.temperatures = temperatures or {}

    def set_pruning_threshold(self, t):
        '''
        Sets the probability threshold below which actions are pruned.
        '''
        self.pruning_threshold = t

    def get_levin_cost(self, node):
        '''
        Gets the levin cost of a node.
        '''
        return np.log(node.get_depth()) - node.get_p()

    def get_expected_lap_time(self, compound, tire_age):
        '''
        Returns the expected lap time for a given compound and tire age.

        tire_model is a nested dict: tire_model[compound][tire_age] → time.
        Falls back to the compound mean if the exact age is not present.
        '''
        compound_model = self.tire_model.get(compound.upper(), {})
        lap_time = compound_model.get(tire_age, None)
        if lap_time is None:
            if compound_model:
                lap_time = np.mean(list(compound_model.values()))
            else:
                lap_time = 90.0  # last-resort fallback
        return lap_time

    def levin_tree_search(self, initial_state: F1State, budget=5000):
        '''
        Runs Levin Tree Search from `initial_state`.

        Parameters
        ----------
        initial_state : F1State
            Starting state (lap=0, compound=starting_compound, tire_age=0).
        budget : int
            Maximum number of node expansions. 0 means unlimited.

        Returns
        -------
        tuple : (path, path_states, total_cost, nodes_expanded)
            path         : list of (lap, action) tuples
            path_states  : list of (lap, compound, tire_age, cumulative_time)
            total_cost   : total race time in seconds (-1 if no solution found)
            nodes_expanded : number of nodes expanded
        '''
        open_list = []
        closed = {}
        nodes_expanded = 0

        root = LevinNode(state=initial_state, prob=1.0, depth=1)
        heapq.heappush(open_list, root)

        while open_list:
            if budget > 0 and nodes_expanded > budget:
                break

            parent = heapq.heappop(open_list)
            state = parent.get_state()
            compound = state.get_compound()
            tire_age = state.get_tire_age()

            action_probs = parent.get_action_probs(
                self.model_pit, self.model_comp, self.tire_model, self.total_laps,
                temperatures=self.temperatures, traffic_penalties=self.traffic_penalties,
            )

            for action, prob in action_probs.items():
                if prob <= self.pruning_threshold:
                    continue

                child_state = copy.deepcopy(state)
                child_state.apply_action(action)

                if action == 'continue':
                    lap_time = self.get_expected_lap_time(compound, tire_age + 1)
                elif action.startswith('pit_'):
                    new_comp = action.split('_')[1]
                    traffic_cost = self.traffic_penalties.get(state.get_lap(), 0.0)
                    lap_time = self.pit_loss + self.get_expected_lap_time(new_comp, 1) + traffic_cost
                else:
                    continue  # unknown action — skip

                child_node = LevinNode(
                    state=child_state,
                    parent=parent,
                    action=action,
                    g=parent.get_g() + lap_time,
                    prob=parent.get_p() * prob,
                    depth=parent.get_depth() + 1,
                )
                child_node.set_levin_cost(self.get_levin_cost(child_node))

                if child_node.is_goal(self.total_laps):
                    return (
                        child_node.get_path(),
                        child_node.get_path_states(),
                        child_node.get_g(),
                        nodes_expanded,
                    )

                h = hash(child_state)
                if h not in closed or child_node.levin_cost < closed[h]:
                    closed[h] = child_node.levin_cost
                    heapq.heappush(open_list, child_node)

            nodes_expanded += 1

        return (None, None, -1, nodes_expanded)
