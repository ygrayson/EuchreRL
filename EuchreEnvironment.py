"""
Euchre Environment for RL Agent.

Card representation convention (total 24 cards) - 
0-5: Clubs 9-A
6-11: Spades 9-A
12-17: Heart 9-A
18-23: Diamond 9-A
E.g. 0 represents Clubs-9; 11 represents Spades-Ace

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

    implemented compatable with RLGlue
    """
    def env_init(self, env_info={}):
        """Initialize the Euchre environment"""

        # reward, state, termination tuple
        self.reward_state_termination = (None, None, None)

        # game parameters for this episode
        self.trump_suit = None
        self.own_points = 0
        self.oppoent_points = 0

        # number of RL agents, default to 1
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
        #TODO: SIMPLIFYING ASSUMPTION: RANDOMLY CHOOSE AN AGENT TO FIRST PLAY A CARD, 
        # THEN EVERYONE PLAY SIMOUTAOUSLY
        state = self.reward_state_termination[1]
        legal_actions = self.get_legal_action()

        # illegal action detected
        if action not in legal_actions:
            raise Exception(str(action) + "not in possible actions")

        # an agent plays the first card
        defining_card = np.random.randint(0, 4)

        # every agent plays a card
        card_0 = state[action+1]
        state = np.delete(state, action+1)
        card_1 = self.agent_1.play_card()
        card_2 = self.agent_2.play_card()
        card_3 = self.agent_3.play_card()

        # get the defining suit
        defining_suit = [card_0, card_1, card_2, card_3][defining_card] // 6

        # add a point according to biggest card
        biggest = self.compare_cards(card_0, card_1, card_2, card_3, defining_suit)
        if biggest == 0 or biggest == 2:
            self.own_points += 1
        else:
            self.oppoent_points += 1

        # all cards played, the episode has terminated
        if len(state) == 1:
            if self.own_points == 5:
                # won the episode with 5 points
                self.reward_state_termination = (2, state, True)
            elif self.own_points == 0:
                # lost the episode with 0 point
                self.reward_state_termination = (-2, state, True)
            elif self.own_points > self.oppoent_points:
                # normally won the episode
                self.reward_state_termination = (1, state, True)
            else:
                # normally lost the episode
                self.reward_state_termination = (-1, state, True)
        else:
            # episode has not terminated
            self.reward_state_termination = (0, state, False)
        
        return self.reward_state_termination

    def env_end(self):
        """Environment ends."""
        pass

    def env_cleanup(self):
        """Cleanup done after the environment ends."""
        self.trump_suit = None
        self.own_points = 0
        self.oppoent_points = 0
    
    def get_legal_action(self):
        #TODO: NEED TO CHECK SAME SUIT
        """
        consider the current state, get legal actions.
        
        RETURN: array of possible actions
        """
        state = self.reward_state_termination[1]
        return np.arange(len(state)-1)
    
    def compare_cards(self, card_0, card_1, card_2, card_3, defining_suit):
        """Compare and return which card is the biggest, given the current-round suit.

        RETURN: The biggest card 0/1/2/3, according to the trump suit and current-round suit"""
        card_0_val = self.card_value(card_0, defining_suit)
        card_1_val = self.card_value(card_1, defining_suit)
        card_2_val = self.card_value(card_2, defining_suit)
        card_3_val = self.card_value(card_3, defining_suit)
        return np.argmax(np.array([card_0_val, card_1_val, card_2_val, card_3_val]))

    #TODO: TO HAVE RL AGENT TO LEARN, WE MUST HAVE INTELLIGENT AGENT TO PLAY AGAINST WITH
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
    
    def card_value(self, card, the_suit):
        """Reevaluate the card value, based on trump suit and current-round suit.
        
        INPUT: card - the card to be reassign value
                  the_suit - the defining suit for this round
        RETURN: card_val - new card value from 0 to 23
        """
        if self.trump_suit == 0:
            if the_suit == 0:
                reassigned_value = [17, 18, 23, 19, 20, 21, 0, 0, 22, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0]
            elif the_suit == 1:
                reassigned_value = [17, 18, 23, 19, 20, 21, 12, 13, 22, 14, 15, 16, 0,0,0,0,0,0,0,0,0,0,0,0]
            elif the_suit == 2:
                reassigned_value = [17, 18, 23, 19, 20, 21, 0, 0, 22, 0, 0, 0, 11, 12, 13, 14, 15, 16, 0,0,0,0,0,0]
            else: #the_suit == 3
                reassigned_value = [17, 18, 23, 19, 20, 21, 0, 0, 22, 0, 0, 0, 0,0,0,0,0,0, 11, 12, 13, 14, 15, 16]
        elif self.trump_suit == 1:
            if the_suit == 0:
                reassigned_value = [12, 13, 22, 14, 15, 16, 17, 18, 23, 19, 20, 21, 0,0,0,0,0,0,0,0,0,0,0,0]
            elif the_suit == 1:
                reassigned_value = [0, 0, 22, 0, 0, 0, 17, 18, 23, 19, 20, 21, 0,0,0,0,0,0,0,0,0,0,0,0]
            elif the_suit == 2:
                reassigned_value = [0, 0, 22, 0, 0, 0, 17, 18, 23, 19, 20, 21, 11, 12, 13, 14, 15, 16, 0,0,0,0,0,0]
            else: #the_suit == 3
                reassigned_value = [0, 0, 22, 0, 0, 0, 17, 18, 23, 19, 20, 21, 0,0,0,0,0,0, 11, 12, 13, 14, 15, 16]
        elif self.trump_suit == 2:
            if the_suit == 0:
                reassigned_value = [11, 12, 13, 14, 15, 16, 0,0,0,0,0,0, 17, 18, 23, 19, 20, 21, 0, 0, 22, 0, 0, 0]
            elif the_suit == 1:
                reassigned_value = [0,0,0,0,0,0, 11, 12, 13, 14, 15, 16, 17, 18, 23, 19, 20, 21, 0, 0, 22, 0, 0, 0]
            elif the_suit == 2:
                reassigned_value = [0,0,0,0,0,0,0,0,0,0,0,0, 17, 18, 23, 19, 20, 21, 0, 0, 22, 0, 0, 0]
            else: #the_suit == 3
                reassigned_value = [0,0,0,0,0,0,0,0,0,0,0,0, 17, 18, 23, 19, 20, 21, 12, 13, 22, 14, 15, 16]
        else: #self.trump_suit == 3
            if the_suit == 0:
                reassigned_value = [11, 12, 13, 14, 15, 16, 0,0,0,0,0,0, 0, 0, 22, 0, 0, 0, 17, 18, 23, 19, 20, 21]
            elif the_suit == 1:
                reassigned_value = [0,0,0,0,0,0, 11, 12, 13, 14, 15, 16, 0, 0, 22, 0, 0, 0, 17, 18, 23, 19, 20, 21]
            elif the_suit == 2:
                reassigned_value = [0,0,0,0,0,0,0,0,0,0,0,0, 12, 13, 22, 14, 15, 16, 17, 18, 23, 19, 20, 21]
            else: #the_suit == 3
                reassigned_value = [0,0,0,0,0,0,0,0,0,0,0,0, 0, 0, 22, 0, 0, 0, 17, 18, 23, 19, 20, 21]

        card_val = reassigned_value[card]
        return card_val



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
