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

        # game parameters for this episode
        self.trump_suit = None
        self.own_points = 0
        self.oppoent_points = 0

        # number of RL agents
        self.agent_num = env_info.get("agent_num", 1)

    def env_start(self):
        """Start environment by dealing the cards.

        Called when the episode starts, before calling agent_start
        RETURNS: initial state
        """
        # decide trump suit and deal cards
        self.trump_suit = np.random.randint(0, 4)
        all_cards = np.arange(24)
        np.random.shuffle(all_cards)
        main_agent_cards = np.sort(all_cards[0:5])

        # create 3 auxilary agents
        self.agent_1 = self.RandomAgent(np.sort(all_cards[5:10]))
        self.agent_2 = self.RandomAgent(np.sort(all_cards[10:15]))
        self.agent_3 = self.RandomAgent(np.sort(all_cards[15:20]))
        
        # store the initial state
        # state representation - (trump_suit(0-3), cards_in_hand)
        init_state = np.concatenate([self.trump_suit, main_agent_cards])
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

        # every agent plays a card
        main_agent_card = state[action+1]
        state = np.delete(state, action+1)
        card_1 = self.agent_1.play_card()
        card_2 = self.agent_2.play_card()
        card_3 = self.agent_3.play_card()

        # add a point according to biggest card
        biggest = self.compare_cards(main_agent_card, card_1, card_2, card_3)
        if biggest == 0 or biggest == 2:
            self.own_points += 1
        else:
            self.oppoent_points += 1

        # all cards played, the episode has terminated
        if len(state) == 1:
            # won the episode
            if self.own_points > self.oppoent_points:
                self.reward_state_termination = (1, state, True)
                return self.reward_state_termination
            # lost it
            else:
                self.reward_state_termination = (-1, state, True)
                return self.reward_state_termination
        else:
            # episode has not terminated
            self.reward_state_termination = (0, state, False)
            return self.reward_state_termination

    def env_end(self):
        pass

    def env_cleanup(self):
        self.trump_suit = None
        self.own_points = 0
        self.oppoent_points = 0
    
    def get_legal_action(self):
        """
        consider the current state, get legal actions.
        
        RETURN: array of possible actions
        """
        state = self.reward_state_termination[1]
        return np.arange(len(state)-1)
    
    def compare_cards(self, card_0, card_1, card_2, card_3):
        """RETURN: The biggest card 0/1/2/3, according to the trump suit"""
        # TODO
        return 0

    class RandomAgent():
        """Random Agent used to train RL agent, as partner or opponent"""
        def __init__(self, cards):
            """Initialize random agent with a number of cards."""
            self.cards = cards
        
        def play_card(self):
            """Play a random card at hand.
            
            RETURN: The card being played
            """
            card_idx = np.random.randint(0, len(self.cards))
            card = self.cards[card_idx]
            self.cards = np.delete(self.cards, card_idx)
            return card
    
    class DummyAgent():
        """Dummy Agent used to train RL agent, as partner or opponent"""
        def __init__(self):
            #TODO
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

