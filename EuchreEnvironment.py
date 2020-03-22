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

        # number of points to win this episode
        self.max_points = env_info.get("max_points", 10)

        # number of RL agents
        self.agent_num = env_info.get("agent_num", 1)

    def env_start(self):
        """
        Start environment by dealing the cards

        Called when the episode starts, before calling agent_start
        """
        # decide trump suit and deal cards
        trump_suit = np.random.randint(0, 5)
        all_cards = np.arange(24)
        np.random.shuffle(all_cards)
        agent_cards = all_cards[0:5]
        
        # store the initial state
        num_cards, combination = self.hand_to_state(agent_cards)
        state = (trump_suit, num_cards, combination)
        reward = 0
        self.reward_state_termination = (reward, state, False)
        return state


    def env_step(self, action):
        """a step taken by the agent, environment respond with reward and next state"""
        pass

    def env_end(self):
        pass

    def env_cleanup(self):
        pass
    
    def state_to_hand(self, state):
        """translate from state to hand of cards"""
        pass

    def hand_to_state(self, cards):
        """
        translate from hand of cards to state

        INPUT: cards is nd array (1-5) each indicating a card
        RETURN: num_cards, combination
        """
        num_cards = len(cards)
        #TODO: given the list of cards, calculate the 1 number representation of state
        combination = 0
        return num_cards, combination

