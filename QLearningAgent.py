"""Reinforcement Learning Q-Learning Agent."""

from numpy import np
from Agent import BaseAgent


class TDAgent(BaseAgent):
    """
    Q-Learning Agent in Euchre environment, using Q-table and TD Learning

    """
    def agent_init(self, agent_info={}):
        """Setup for the agent called when the experiment first starts."""

        # get agent parameters
        self.num_states = agent_info.get('num_states')
        self.num_actions = agent_info.get('num_actions')
        self.discount = agent_info.get('discount', 0.9)
        self.learning_rate = agent_info.get('learning_rate', 0.1)
        self.epsilon = agent_info.get('epsilon', 0.05)

        # initialize Q-values for all different state-action pair
        self.q_values = None

    def agent_start(self, state):
        pass

    def agent_step(self, reward, state):
        pass

    def agent_end(self, reward):
        pass

    def agent_cleanup(self):
        pass