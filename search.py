import heapq
import copy
import numpy as np
import pandas as pd
from node import Node, LevinNode
from F1State import F1State
from model import create_regression_models

class AStar:
    """
    This class implements the A* algorithm.
    """
    def __init__(self, total_laps, pit_loss, max_stints, tire_model, compounds=None):
        self.total_laps = total_laps
        self.pit_loss = pit_loss
        self.max_stints = max_stints
        self.tire_model = tire_model
        self.compounds = compounds if compounds else list(tire_model.keys())

    def heuristic(self, state):
        """
        Calculates heuristic for A*: minimum possible lap times for remaining laps.
        """
        remaining_laps = max(0, self.total_laps - state.get_lap())
        # Use fastest compound's base lap time
        fastest_times = [min(self.tire_model[comp].values()) for comp in self.compounds]
        return remaining_laps * min(fastest_times)

    def get_successors(self, node):
        """
        Returns a list of successor nodes from the current node.
        """
        successors = []
        state = node.get_state()
        lap = state.get_lap()
        compound = state.get_compound()
        tire_age = state.get_tire_age()

        # Continue with current tire
        if tire_age + 1 <= self.max_stints[compound] and (tire_age + 1 in self.tire_model[compound]):
            next_lap = lap + 1
            next_tire_age = tire_age + 1 if next_lap > 1 else tire_age
            next_lap_time = self.tire_model[compound][tire_age + 1]
            next_state = F1State(next_lap, compound, next_tire_age)
            successors.append(Node(next_state, node, action="continue",
                                   g=node.g + next_lap_time,
                                   h=self.heuristic(next_state)))

        # Pit to a new tire
        for new_comp in self.compounds:
            # Only pit if there are valid tire times at age 1
            if 1 in self.tire_model[new_comp]:
                next_lap_time = self.pit_loss + self.tire_model[new_comp][1]
                next_state = F1State(lap + 1, new_comp, 1)
                successors.append(Node(next_state, node, action=f"pit_{new_comp}",
                                       g=node.g + next_lap_time,
                                       h=self.heuristic(next_state)))
        return successors

    def a_star(self, initial_state: F1State):
        open_list = []
        closed = {} 
        nodes_expanded = 0

        start_node = Node(
            initial_state,
            parent=None,
            action=None,
            g=0.0,
            h=self.heuristic(initial_state)
        )

        heapq.heappush(open_list, start_node)
        closed[hash(initial_state)] = 0.0

        while open_list:
            current = heapq.heappop(open_list)
            if current.is_goal(self.total_laps):
                return (current.get_path(), current.get_path_states(), current.get_g(), nodes_expanded)
            for neighbour in self.get_successors(current):
                state_hash = hash(neighbour.state)
                g_cost = neighbour.get_g()
                if state_hash not in closed or g_cost < closed[state_hash]:
                    closed[state_hash] = g_cost
                    heapq.heappush(open_list, neighbour)
            nodes_expanded += 1

        return (None, None, -1, nodes_expanded)
    

class LevinTreeSearch:
    """
    This class implements the Levin Tree Search algorithm.
    """
    def __init__(self, total_laps, pit_loss, tire_model, pruning_threshold=0):
        '''
        When creating an instance of this class, two regression models will also be created:
            - model_pit: logistic regression model for deciding if a driver should pit or continue
            - model_comp: logistic regression model for deciding which compound to pit for given the 
                          driver is pitting
        '''
        self.model_pit, self.model_comp = create_regression_models()
        self.total_laps = total_laps
        self.pit_loss = pit_loss
        self.tire_model = tire_model
        self.pruning_threshold = pruning_threshold

    def set_pruning_threshold(self, t):
        '''
        Sets the threshold for pruning.
        '''
        self.pruning_threshold = t

    def get_levin_cost(self, node):
        '''
        Gets the levin cost of a node.
        '''
        return np.log(node.get_depth()) - node.get_p()
    
    def get_expected_lap_time(self, compound, tire_age):
        '''
        Gets the expected lap time from the tire degradation model. 
        
        If a lap time for the given compound and tire age is not available, 
        the function falls back to using the average lap time for that compound.
        '''
        expected_lap_time = self.tire_model.get((compound.upper(), tire_age), None)
        if expected_lap_time is None:
            expected_lap_time = np.mean(list(self.tire_model[compound.upper()].values()))
        return expected_lap_time
    
    def levin_tree_search(self, initial_state: F1State, budget=5000):
        open = []
        closed = {}
        nodes_expanded = 0

        root = LevinNode(state=initial_state, prob=1.0, depth=1)

        heapq.heappush(open, root)

        while open:
            if budget > 0 and nodes_expanded > budget:
                break

            parent = heapq.heappop(open)
            state = parent.get_state()
            compound = state.get_compound()
            tire_age = state.get_tire_age()

            action_probs = parent.get_action_probs(self.model_pit, self.model_comp, self.tire_model, self.total_laps)

            for action, prob in action_probs.items():
                if prob <= 0 or prob < self.pruning_threshold:  # prunes based on the threshold for probability
                    continue 

                child_state = copy.deepcopy(state)
                child_state.apply_action(action) 

                if action == "continue":
                    lap_time = self.get_expected_lap_time(compound, tire_age + 1)
                elif action.startswith("pit_"):
                    new_comp = action.split("_")[1]
                    lap_time = self.pit_loss + self.get_expected_lap_time(new_comp, 1)

                child_node = LevinNode(
                    state=child_state,
                    parent=parent,
                    action=action,
                    g=parent.get_g() + lap_time,
                    prob=parent.get_p() * prob,
                    depth=parent.get_depth() + 1
                )
                child_node.set_levin_cost(self.get_levin_cost(child_node))

                if child_node.is_goal(self.total_laps):
                    return (child_node.get_path(), child_node.get_path_states(), child_node.get_g(), nodes_expanded)
                
                h = hash(child_state) 

                if h not in closed or child_node.levin_cost < closed[h]:
                    closed[h] = child_node.levin_cost
                    heapq.heappush(open, child_node)
            nodes_expanded += 1

        return (None, None, -1, nodes_expanded)