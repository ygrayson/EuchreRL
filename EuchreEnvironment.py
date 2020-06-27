"""
Euchre Environment for RL Agent.

Card representation convention (total 24 cards) - 
0-5: Clubs 9-A
6-11: Spades 9-A
12-17: Heart 9-A
18-23: Diamond 9-A
E.g. 0 represents Clubs 9; 11 represents Spades A

Suits representation convention (total 4 suits) - 
0: Clubs
1: Spades
2: Heart
3: Diamond
"""

import numpy as np
from Environment import BaseEnvironment


class EuchreEnvironment(BaseEnvironment):
    """
    Environment for Euchre card game

    """
    def env_init(self, env_info={}):
        """Initialize the Euchre environment"""

        # reward, state, termination tuple
        reward = None
        state = None
        termination = None
        self.reward_state_termination = (reward, state, termination)

        # intermediate points
        self.own_points = 0
        self.oppoent_points = 0

        # TODO: DO I NEED THIS?
        # number of points to win this episode
        self.max_points = env_info.get("max_points", 10)

        # number of RL agents
        self.agent_num = env_info.get("agent_num", 1)

    def env_start(self):
        """Start environment by dealing the cards.

        Called when the episode starts, before calling agent_start
        RETURNS: initial state
        """
        # decide trump suit and deal cards
        trump_suit = np.random.randint(0, 4)
        all_cards = np.arange(24)
        np.random.shuffle(all_cards)
        main_agent_cards = np.sort(all_cards[0:5])
        agent_1_cards = np.sort(all_cards[5:10])
        agent_2_cards = np.sort(all_cards[10:15])
        agent_3_cards = np.sort(all_cards[15:20])
        
        # store the initial state
        # state representation - (trump_suit(0-3), cards_in_hand)
        init_state = np.concatenate([trump_suit, main_agent_cards])
        reward = 0
        self.reward_state_termination = (reward, init_state, False)
        return init_state


    def env_step(self, action):
        """A step taken by the environment.

        INPUT:
            action: The action taken by the agent

        RETURNS:
            (float, state, Boolean): a tuple of the reward, next state,
                and boolean indicating if it's terminal.
        """
        state = self.reward_state_termination[1]
        legal_actions = self.get_legal_action()

        # illegal action detected
        if action not in legal_actions:
            raise Exception(str(action) + "not in possible actions")

        # change to next state and return reward
        state = np.delete(state, action+1)

        # if not terminate, add points for this round of hand

        # if terminate, assign reward accordingly
        

    def env_end(self):
        pass

    def env_cleanup(self):
        self.own_points = 0
        self.oppoent_points = 0
        pass
    
    def get_legal_action(self):
        """
        consider the current state, get legal actions.
        
        RETURN: array of possible actions
        """
        state = self.reward_state_termination[1]
        return np.arange(len(state)-1)
    
    class RandomAgent():
        """Random Agent used to train RL agent, as partner or opponent"""
        def __init__(self):
            pass
    
    class DummyAgent():
        """Random Agent used to train RL agent, as partner or opponent"""
        def __init__(self):
            pass
    
    def state_to_hand(self, state):
        """translate from state to hand of cards"""
        pass

    def hand_to_state(self, cards):
        """
        translate from hand of cards to state

        INPUT: cards is 1-5th dimentional array with each number indicating a card
        RETURN: num_cards, combination
        """
        num_cards = len(cards)
        #TODO: given the list of cards, calculate the 1 number representation of state
        combination = 0
        return num_cards, combination

