import numpy as np
import pandas as pd


class Node:
    '''
    Base search node.

    Tracks cost-so-far (g), the action that produced this node, a pointer to
    the parent, and the set of tire compounds used along the path so far.
    The two-compound rule is enforced in `is_goal`.
    '''

    def __init__(self, state, parent=None, action=None, g=0.0, h=0.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.g = g
        self.h = h
        self.f = g + h

        if parent is None:
            self.depth = 0
            self.compounds_used = {state.compound}
        else:
            self.depth = parent.depth + 1
            self.compounds_used = parent.compounds_used | {state.compound}

    def __lt__(self, other):
        return (self.f, self.g) < (other.f, other.g)

    # ── Getters ────────────────────────────────────────────────────────────────

    def get_state(self):
        return self.state

    def get_parent(self):
        return self.parent

    def get_action(self):
        return self.action

    def get_g(self):
        return self.g

    def get_f(self):
        return self.f

    # ── Goal test ──────────────────────────────────────────────────────────────

    def is_goal(self, total_laps):
        '''
        Returns True when the node has completed the final lap using at least
        two distinct dry compounds (F1 mandatory two-compound rule).
        '''
        return (
            self.state.get_lap() == total_laps
            and len(self.compounds_used) >= 2
        )

    # ── Path reconstruction ────────────────────────────────────────────────────

    def get_path(self):
        '''Returns the action sequence as a list of (lap, action) tuples.'''
        actions = []
        node = self
        while node.parent is not None:
            actions.append((node.state.lap, node.action))
            node = node.parent
        return list(reversed(actions))

    def get_path_states(self):
        '''
        Returns the full path as a list of
        (lap, compound, tire_age, cumulative_time, action) tuples.
        '''
        states = []
        node = self
        while node.parent is not None:
            states.append((node.state.lap, node.state.compound,
                           node.state.tire_age, node.g, node.action))
            node = node.parent
        return list(reversed(states))


class LevinNode(Node):
    '''
    Search node for Levin Tree Search.

    Extends Node with:
    - `prob`  : cumulative product of action probabilities along the path
    - `depth` : number of actions taken from the root
    - `levin_cost` : log(depth) − log(prob)

    The heap is ordered by levin_cost (lowest first), so the search expands
    nodes that are both shallow and reached via high-probability actions.
    '''

    def __init__(self, state, parent=None, action=None,
                 g=0.0, h=0.0, prob=1.0, depth=1):
        super().__init__(state, parent, action, g, h)
        self.prob = prob
        self.depth = depth
        self.levin_cost = 0.0  # set via set_levin_cost() after construction

    def __lt__(self, other):
        return self.levin_cost < other.levin_cost

    def get_depth(self):
        return self.depth

    def get_p(self):
        '''
        Returns the cumulative action probability.
        '''
        return self.prob

    def set_levin_cost(self, cost):
        self.levin_cost = cost

    def get_action_probs(self, model_pit, model_comp, tire_model, total_laps=58,
                          temperatures=None, traffic_penalties=None):
        '''
        Queries both policy models to produce a probability distribution over
        the four available actions.

        `temperatures` ({lap: (air_temp, track_temp)}) and `traffic_penalties`
        ({lap: penalty}) are optional; when the policy models were fit with
        those feature columns, the corresponding value for the current lap
        (falling back to the nearest known lap, or 0.0) is supplied.

        Returns
        -------
        dict : {action_str → probability}
            Keys: "continue", "pit_SOFT", "pit_MEDIUM", "pit_HARD"
        '''
        state = self.get_state()
        lap = state.get_lap()
        tire_age = state.get_tire_age()
        compound = state.get_compound()
        laps_remaining = total_laps - lap

        is_medium = 1 if compound == 'MEDIUM' else 0
        is_hard   = 1 if compound == 'HARD'   else 0

        expected_lap_time = tire_model.get(compound, {}).get(tire_age, None)
        if expected_lap_time is None:
            compound_vals = tire_model.get(compound.upper(), tire_model.get(compound, {}))
            expected_lap_time = np.mean(list(compound_vals.values())) if compound_vals else 90.0

        temperatures = temperatures or {}
        if lap in temperatures:
            air_temp, track_temp = temperatures[lap]
        elif temperatures:
            nearest_lap = min(temperatures.keys(), key=lambda l: abs(l - lap))
            air_temp, track_temp = temperatures[nearest_lap]
        else:
            air_temp, track_temp = 0.0, 0.0
        traffic_penalty = (traffic_penalties or {}).get(lap, 0.0)

        x_pit = pd.DataFrame([{
            'lap': lap,
            'laps_remaining': laps_remaining,
            'tire_age': tire_age,
            'is_medium': is_medium,
            'is_hard': is_hard,
            'expected_lap_time': expected_lap_time,
            'air_temperature': air_temp,
            'track_temperature': track_temp,
            'traffic_penalty': traffic_penalty,
        }])
        x_pit = x_pit.reindex(columns=model_pit.feature_names_in_, fill_value=0.0)

        pit_probs = model_pit.predict_proba(x_pit)[0]
        p_continue = pit_probs[0]
        p_pit      = pit_probs[1]

        x_comp = pd.DataFrame([{
            'lap': lap,
            'laps_remaining': laps_remaining,
            'tire_age': tire_age,
            'is_medium_before': is_medium,
            'is_hard_before':   is_hard,
            'air_temperature': air_temp,
            'track_temperature': track_temp,
        }])
        x_comp = x_comp.reindex(columns=model_comp.feature_names_in_, fill_value=0.0)

        comp_probs = model_comp.predict_proba(x_comp)[0]
        prob_map   = dict(zip(model_comp.classes_, comp_probs))

        return {
            'continue':   p_continue,
            'pit_SOFT':   p_pit * prob_map.get('SOFT',   0.0),
            'pit_MEDIUM': p_pit * prob_map.get('MEDIUM', 0.0),
            'pit_HARD':   p_pit * prob_map.get('HARD',   0.0),
        }
