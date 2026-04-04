import numpy as np
import pandas as pd 

class Node:
    '''
    This class represents a node used for search algorithms.
    A node as the following attributes:
        - state (F1State): the current state
        - parent (Node): the parent node
        - action (str): the action taken to reach this node
        - g (float): the g-value
        - h (float): the h-value
        - f (float): the f-value
        - compounds_used: set of tire compounds that has been used so far. This is to make 
                        sure that the recovered path will use at least two different dry compounds

    The actions available are "continue", "pit_soft", "pit_medium", or "pit_hard"
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
    
    def __lt__(self, node):
        '''
        The less than operator compares the f-values between two nodes.
        '''
        return (self.f, self.g) < (node.f, node.g)
    
    def get_state(self):
        '''
        Getter for state
        '''
        return self.state
    
    def get_parent(self):
        '''
        Getter for parent
        '''
        return self.parent
    
    def get_action(self):
        '''
        Getter for action
        '''
        return self.action
    
    def get_g(self):
        '''
        Getter for g-value
        '''
        return self.g
    
    def get_f(self):
        '''
        Getter for f-value
        '''
        return self.f

    def is_goal(self, total_laps):
        '''
        Checks if the node has reached the goal(ie. finishes the last lap)
        '''
        return (
            self.state.get_lap() == total_laps and
            len(self.compounds_used) >= 2
        )
    
    def get_path(self):
        '''
        Gets the recovered path
        '''
        actions = []
        node = self
        while node.parent is not None:
            actions.append((node.state.lap, node.action))
            node = node.parent
        return list(reversed(actions))
    
    def get_path_states(self):
        '''
        Gets the recovered path that also includes the tire compound, tire age, 
        and cumulative race time.
        '''
        states = []
        node = self
        while node.parent is not None:
            states.append((node.state.lap, node.state.compound, node.state.tire_age, node.g))
            node = node.parent
        return list(reversed(states))


class LevinNode(Node):
    '''
    This subclass reprents a node used for Levin Tree Search. It extends Node.
    '''
    def __init__(self, state, parent=None, action=None, g=0.0, h=0.0, prob=1.0, depth=1):
        super().__init__(state, parent, action, g, h)
        self.prob = prob
        self.depth = depth

    def __lt__(self, node):
        '''
        The less than operator compares the levin cost between two nodes.
        '''
        return self.levin_cost < node.levin_cost
    
    def get_depth(self):
        '''
        Gets the depth of the node.
        '''
        return self.depth
    
    def get_p(self):
        '''
        Gets the probability to reach the node.
        '''
        return self.prob

    def get_action_probs(self, model_pit, model_comp, tire_model, total_laps=58):
        '''
        Computes the probabilities of possible actions for the current state.
        '''
        state = self.get_state()
        lap = state.get_lap()
        tire_age = state.get_tire_age()
        compound = state.get_compound()
        laps_remaining = total_laps - lap

        is_medium = 1 if compound == "MEDIUM" else 0
        is_hard = 1 if compound == "HARD" else 0

        expected_lap_time = tire_model.get((compound, tire_age), None)
        if expected_lap_time is None:
            # use mean lap time for the compound
            expected_lap_time = np.mean(list(tire_model[compound.upper()].values()))

        x_pit = pd.DataFrame([{
            "lap": lap,
            "laps_remaining": laps_remaining,
            "tire_age": tire_age,
            "is_medium": is_medium,
            "is_hard": is_hard,
            "expected_lap_time": expected_lap_time
        }])

        pit_probs = model_pit.predict_proba(x_pit)[0]
        P_continue = pit_probs[0]
        P_pit = pit_probs[1]

        x_comp = pd.DataFrame([{
            "lap": lap,
            "laps_remaining": laps_remaining,
            "tire_age": tire_age,
            "is_medium_before": is_medium,
            "is_hard_before": is_hard
        }])

        comp_probs = model_comp.predict_proba(x_comp)[0]
        classes = model_comp.classes_
        prob_map = dict(zip(classes, comp_probs))

        action_probs = {
            "continue": P_continue,
            "pit_SOFT": P_pit * prob_map.get("SOFT", 0.0),
            "pit_MEDIUM": P_pit * prob_map.get("MEDIUM", 0.0),
            "pit_HARD": P_pit * prob_map.get("HARD", 0.0),
        }
        return action_probs
    
    def set_levin_cost(self, c):
        '''
        Sets the levin cost.
        '''
        self.levin_cost = c