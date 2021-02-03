import matplotlib
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy.special import softmax
from time import time
from qlearning import QLearningAgent
from sarsa import SARSAAgent
from dynaq import DYNAQAgent
from utils import *


if __name__ == '__main__':

    env = gym.make("gridworld-v0")
    plans_to_evaluate = [7]

    episode_count = 5000
    lr = 0.25
    lr_R_P = 0.25
    gamma = 1.0
    k = 5
    for plan in plans_to_evaluate:
        print(f'Evaluation Plan{plan}')
        env.setPlan(f"gridworldPlans/plan{plan}.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
        render = False
        env.seed(0)  # Initialise le seed du pseudo-random
        # env.render()  # permet de visualiser la grille du jeu
        # Execution avec plusieurs agents

        agents = [QLearningAgent(env), SARSAAgent(env), DYNAQAgent(env)]
        agents_names = ['Q-Learning Agent', 'SARSA Agent', 'DYNA-Q Agent']

        for agent, agent_name in zip(agents, agents_names):
            # train
            current_time = time()
            rewards_train = agent.train(_episode_count=episode_count, _lr=lr, _lr_R_P=lr_R_P, _gamma=gamma, _k=k,
                                        render=False, verbose=False)  # Define policy
            training_time = time() - current_time

            rewards_test = evaluate_agent(env, agent, _episode_count=200, render=False)
            print(f'\t{agent_name}:')
            print(f'\tTraining time:', training_time)
            print(f'\tmean_reward_train = {np.mean(rewards_train)}, std_reward_train = {np.std(rewards_train)}')
            print(f'\tmean_reward_test = {np.mean(rewards_test)}, std_reward_test = {np.std(rewards_test)}')

            # Plot learning lines
            plt.figure(0)
            to_plot_x = range(0, episode_count, 50)
            to_plot_y = [rewards_train[i] for i in to_plot_x]
            plt.plot(to_plot_x, to_plot_y, label=agent_name)

            plt.figure(1)
            plt.plot(np.cumsum(rewards_train) / np.arange(1, len(rewards_train) + 1), label=agent_name)

        plt.figure(0)
        plt.title(f'Learning curves for Plan{plan}')
        plt.xlabel('Epoch')
        plt.ylabel('Reward')
        plt.legend()
        # plt.show()
        plt.savefig(f'figures/plan{plan}_learning_curve')
        plt.clf()

        plt.figure(1)
        plt.title(f'Cumulative mean reward for Plan{plan}')
        plt.xlabel('Epoch')
        plt.ylabel('Cumulative Mean Reward')
        plt.legend()
        # plt.show()
        plt.savefig(f'figures/plan{plan}_cumrew')
        plt.clf()


    env.close()

