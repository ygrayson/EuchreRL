# Euchre RL Agent Environment

class EuchreEnvironment():
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

    def env_start(self):
        """
        Start environment by dealing the cards

        Called when the episode starts, before calling agent_start
        """
        # TODO: deal the cards randomly, define the starting state for agent

    def env_step(self):
        pass


class QLearningAgent():
    """
    Q-Learning Agent in Euchre environment

    """
