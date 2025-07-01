MAX_FACE = 6 # Maximum face value of a die (1-6)
MAX_DICE = 5 # To determine the number of dice each player starts with
MAX_PLAYERS = 6 # This is to determine the action space size
MAX_BID_ACTIONS = MAX_PLAYERS * MAX_DICE * MAX_FACE  # Total actions in the game
MAX_ACTIONS = MAX_BID_ACTIONS + 1  # Including call bluff (action 0)