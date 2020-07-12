"""Reinforcement Learning Q-Learning Agent."""

from numpy import np
from Agent import BaseAgent


class TDAgent(BaseAgent):
    """
    Q-Learning Agent in Euchre environment, using Q-table and TD Learning

    """
    def agent_init(self, agent_info={}):
        """Setup for the agent called when the experiment first starts."""
        # Discount factor (gamma) to use in the updates
        self.discount = agent_info.get('discount')
        # The learning rate or step size parameter (alpha) to use in updates
        self.learning_rate = agent_info.get('learning_rate')

        # initialize Q-values for all different state-action pair
        self.values = None

    def agent_start(self, state):
        pass

    def agent_step(self, reward, state):
        pass

    def agent_end(self, reward):
        pass

    def agent_cleanup(self):
        pass