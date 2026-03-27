class F1State:
    '''
    This class represents a state in the pit stop strategy search problem.
    The state is represented by the following attributes:
        - lap (int): current lap number
        - compound (str): current tire compound (hard, medium, or soft)
        - tire_age (int): number of laps completed with the current tire
    '''

    def __init__(self, lap, compound, tire_age):
        self.lap = lap
        self.compound = compound
        self.tire_age = tire_age

    def __repr__(self):
        state_str = ""
        state_str += f"Current lap: {self.lap}\n"
        state_str += f"Current tire compound: {self.compound}\n"
        state_str += f"Current tire age: {self.tire_age:.2f}\n"
        return state_str

    def __hash__(self):
        return hash((self.lap, self.compound, self.tire_age))

    def __eq__(self, state):
        return (isinstance(state, F1State) and 
                self.lap == state.lap and
                self.compound == state.compound and
                self.tire_age == state.tire_age)
    
    def get_lap(self):
        '''
        Gets the current lap number
        '''
        return self.lap
    
    def get_compound(self):
        '''
        Gets the current tire compound
        '''
        return self.compound
    
    def get_tire_age(self):
        '''
        Gets the current tire age
        '''
        return self.tire_age
    
    def apply_action(self, action):
        """
        Updates the state based on the action taken.
        Actions can be:
            - "continue": keep current tire
            - "pit_soft", "pit_medium", "pit_hard": change to that compound
        """
        if action == "continue":
            self.lap += 1
            self.tire_age += 1
        elif action.startswith("pit_"):
            new_compound = action.split("_")[1].upper() 
            self.compound = new_compound
            self.tire_age = 0
            self.lap += 1