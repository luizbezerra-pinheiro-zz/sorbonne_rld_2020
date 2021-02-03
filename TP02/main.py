import matplotlib
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
import random
from time import time
from randomAgent import RandomAgent
from policeAgent import PoliceAgent
from valueAgent import ValueAgent
matplotlib.use("TkAgg")


def execution(_agent, episode_count=1000, render=False, verbose=False):
    reward = 0
    done = False
    FPS = 0.0001
    rewards_test = []
    n_actions_test = []
    for i in range(episode_count):
        obs = env.reset()
        env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        if render and env.verbose:
            env.render(FPS)
        j = 0
        rsum = 0
        while True:
            action = _agent.act(obs, reward, done)
            obs, reward, done, _ = env.step(action)
            rsum += reward
            j += 1
            if render and env.verbose:
                env.render(FPS)
            if done:
                rewards_test += [rsum]
                n_actions_test += [j]
                if i % 100 == 0 and i > 0 and verbose:
                    print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break
    return rewards_test, n_actions_test

if __name__ == '__main__':

    env = gym.make("gridworld-v0")
    plans_to_evaluate = [7]
    max_iter = 100000
    for plan in plans_to_evaluate:
        print(f'Evaluation Plan{plan}')
        env.setPlan(f"gridworldPlans/plan{plan}.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
        render = False
        env.seed(0)  # Initialise le seed du pseudo-random
        img = env.render(save_path=f'figures/plan{plan}.png')  # permet de visualiser la grille du jeu
        statedic, mdp = env.getMDP()  # recupere le mdp : statedic
        state, transitions = list(mdp.items())[0]
        print("\tNombre d'etats : ", len(statedic))  # nombre d'etats ,statedic : etat-> numero de l'etat
        # Execution avec plusieurs agents

        agents = [RandomAgent(env.action_space), ValueAgent(env.action_space), PoliceAgent(env.action_space)]
        agents_names = ['Random Agent', 'Value Agent', 'Police Agent']

        for agent, agent_name in zip(agents, agents_names):
            # train
            current_time = time()
            agent.train(statedic, mdp, max_iter=max_iter)  # Define policy
            training_time = time() - current_time

            rewards_test, n_actions_test = execution(agent)
            print(f'\t{agent_name}:')
            print(f'\tTraining time:', training_time)
            print(f'\tmean_reward_test = {np.mean(rewards_test)}, std_reward_test = {np.std(rewards_test)}')
            print(f'\tmean_actions_test = {np.mean(n_actions_test)}, std_actions_test = {np.std(n_actions_test)}')

        env.close()
