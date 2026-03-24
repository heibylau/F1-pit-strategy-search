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


class LevinNode(Node):
    '''
    This subclass reprents a node used for Levin Tree Search. It extends Node.
    '''
    def __init__(self, state, parent=None, action=None, g=0.0, h=0.0, prob=1.0):
        super().__init__(state, parent, action, g, h)
        self.prob = prob