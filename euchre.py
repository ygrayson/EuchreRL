# Euchre RL Agent Environment
# Author: Qianbo Yin


from rl_glue import RLGlue
from manager import Manager
from EuchreEnvironment import EuchreEnvironment
from QLearningAgent import TDAgent


def main():
    #TODO: specify policy somehow
    policy = None

    # define environment and agent
    env = EuchreEnvironment
    agent = TDAgent
    env_info = {'max_points': 10, 'agent_num': 1}
    agent_info = {'policy': policy}

    # use RLGlue to run experiment
    rl_glue = RLGlue(env, agent)
    rl_glue.rl_init(agent_info, env_info)



if __name__ == '__main__':
    main()
