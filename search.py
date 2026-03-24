import heapq
from node import Node
from F1State import F1State

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
            next_lap_time = self.tire_model[compound][tire_age + 1]
            next_state = F1State(lap + 1, compound, tire_age + 1)
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
                return (current.get_path(), current.get_g(), nodes_expanded)
            for neighbour in self.get_successors(current):
                state_hash = hash(neighbour.state)
                g_cost = neighbour.get_g()
                if state_hash not in closed or g_cost < closed[state_hash]:
                    closed[state_hash] = g_cost
                    heapq.heappush(open_list, neighbour)
            nodes_expanded += 1

        return (None, -1, nodes_expanded)
    

class LevinTreeSearch:
    pass