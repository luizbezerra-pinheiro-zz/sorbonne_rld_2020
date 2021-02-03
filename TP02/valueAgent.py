import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
import random


class ValueAgent(object):
    """Police iteration based agent"""

    def __init__(self, action_space):
        self.action_space = action_space
        self.policy = None

    def train(self, states, mdp, eps=0.001, gamma=0.99, max_iter=1000000):
        # statedic, mdp = env.getMDP()
        pi = {s: 0 for s in states.keys()}  # random action for each state

        V_i = {s: random.random() for s in states.keys()}  # random initiation of Value Function s |-> V(s)
        V_i_new = {s: random.random() for s in states.keys()}
        it = 0
        while np.linalg.norm([V_i[state] - V_i_new[state] for state in V_i.keys()]) >= eps and it < max_iter:
            V_i_old = copy.copy(V_i_new)
            for state, transitions in mdp.items():
                V_i_new[state] = np.max(
                [sum([el[0] * (el[2] + gamma * V_i_new[el[1]]) for el in mdp[state][action]]) for action in
                 range(self.action_space.n)])
            V_i = copy.copy(V_i_old)
            it += 1
        # Update policy
        for state in mdp.keys():
            pi[state] = np.argmax(
                [sum([el[0] * (el[2] + gamma * V_i_new[el[1]]) for el in mdp[state][action]]) for action in
                 range(self.action_space.n)])
        self.policy = copy.copy(pi)

    def act(self, observation, reward, done):
        return self.policy[gridworld.GridworldEnv.state2str(observation).replace('\n', '')]


if __name__ == '__main__':

    env = gym.make("gridworld-v0")
    env.setPlan("gridworldPlans/plan0.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})

    env.seed(0)  # Initialise le seed du pseudo-random
    print(env.action_space)  # Quelles sont les actions possibles
    print(env.step(1))  # faire action 1 et retourne l'observation, le reward, et un done un booleen (jeu fini ou pas)
    env.render()  # permet de visualiser la grille du jeu
    env.render(mode="human")  # visualisation sur la console
    statedic, mdp = env.getMDP()  # recupere le mdp : statedic
    print("Nombre d'etats : ", len(statedic))  # nombre d'etats ,statedic : etat-> numero de l'etat
    state, transitions = list(mdp.items())[0]
    print(state)  # un etat du mdp
    print(transitions)  # dictionnaire des transitions pour l'etat :  {action-> [proba,etat,reward,done]}

    # Execution avec un Agent
    agent = ValueAgent(env.action_space)
    agent.train(statedic, mdp)  # Define policy

    episode_count = 10000
    reward = 0
    done = False
    rsum = 0
    FPS = 0.0001
    for i in range(episode_count):
        obs = env.reset()
        env.verbose = False#(i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render(FPS)
        j = 0
        rsum = 0
        while True:
            action = agent.act(obs, reward, done)
            obs, reward, done, _ = env.step(action)
            rsum += reward
            j += 1
            if env.verbose:
                env.render(FPS)
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break

    print("done")
    env.close()
