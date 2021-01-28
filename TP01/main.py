""" Main imports """

import os
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

""" Basic functions """


def load_data(filename):
    """
    Read filename in the proper format:
        <numero de l'article>:<représentation de l'article en 5 dimensions séparées par des ";">:<taux de
clics sur les publicités de 10 annonceurs séparés par des ";">

    :return Return a pandas.DataFrame
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    page_indexes = []
    page_embeds = []
    page_rewards = []
    for line in lines:
        line_splited = line[:-1].split(':')
        page_indexes.append(line_splited[0])
        page_embeds.append(line_splited[1].split(';'))
        page_rewards.append(line_splited[2].split(';'))

    return {'states': np.array(page_embeds).astype(np.float), 'rewards': np.array(page_rewards).astype(np.float)}


""" Agents """


def random_baseline_agent(rewards):
    return [random.randint(0, 9) for _ in rewards]


def static_best_baseline_agent(rewards):
    mean_rewards = np.mean(rewards, axis=0)
    best_agent = np.argmax(mean_rewards)
    return [best_agent for _ in rewards]


def greedy_baseline_agent(rewards):
    best_agents = np.argmax(rewards, axis=1)
    return best_agents


def UCB_agent_step(rewards, last_actions=[], n_bras=10):
    #  Based on the last actions, we have access to their rewards (real case).
    #  So, we estimate the expected return of each arm
    random.seed(0)
    if not last_actions:
        return random.randint(0, 9)
    rewards_by_arm = [[] for i in range(n_bras)]
    [rewards_by_arm[action].append(rewards[i][action]) for i, action in enumerate(last_actions)]
    # With all the rewards computed, we may estimate the best action
    B = [((sum(rewards_i) + math.sqrt(2*math.log(len(last_actions))/len(rewards_i))) if rewards_i else 10000)
         for i, rewards_i in enumerate(rewards_by_arm)]
    return np.argmax(B)


def UCB_agent(rewards, n_bras=10):
    last_actions = []
    for i in range(len(rewards)):
        last_actions.append(UCB_agent_step(rewards, last_actions, n_bras=n_bras))
    return last_actions


def lin_UCB_agent(states, rewards, n_bras=10):
    N = len(rewards)
    d = np.shape(states)[1]
    alpha = 21

    # Initialize our matrices (regression matrices)
    best_actions = []
    A_matrices = [np.identity(d) for _ in range(n_bras)]
    b_matrices = [np.zeros((d, 1)) for _ in range(n_bras)]
    for t in range(N):
        x_t = states[t]
        p = [0 for _ in range(n_bras)]
        for a in range(n_bras):
            A_a_inverted = np.linalg.inv(A_matrices[a])
            teta_a = A_a_inverted @ b_matrices[a]
            p[a] = (teta_a.T @ x_t + alpha * math.sqrt(x_t.T @ A_a_inverted @ x_t))[0]

        action = np.argmax(p)
        best_actions.append(action)
        A_matrices[action] = A_matrices[action] + x_t @ x_t.T
        b_matrices[action] = b_matrices[action] + rewards[t][action] * x_t
    return best_actions

""" Plot Regret """


def regret_vs_t(states, rewards):
    random_agent_actions = random_baseline_agent(rewards)    # random_baseline
    best_agent_actions = static_best_baseline_agent(rewards)
    greedy_agent_actions = greedy_baseline_agent(rewards)
    ucb_actions = UCB_agent(rewards)
    lin_ucb_actions = lin_UCB_agent(states, rewards)

    # Rewards
    random_rewards = [rewards_t[action] for action, rewards_t in zip(random_agent_actions, rewards)]
    best_agent_rewards = [rewards_t[action] for action, rewards_t in zip(best_agent_actions, rewards)]
    greedy_agent_rewards = [rewards_t[action] for action, rewards_t in zip(greedy_agent_actions, rewards)]
    ucb_agent_rewards = [rewards_t[action] for action, rewards_t in zip(ucb_actions, rewards)]
    lin_ucb_agent_rewards = [rewards_t[action] for action, rewards_t in zip(lin_ucb_actions, rewards)]

    # Accumulated Reward
    random_rewards_cum = np.cumsum(random_rewards)
    best_agent_rewards_cum = np.cumsum(best_agent_rewards)
    greedy_agent_rewards_cum = np.cumsum(greedy_agent_rewards)
    ucb_agent_rewards_cum = np.cumsum(ucb_agent_rewards)
    lin_ucb_agent_rewards_cum = np.cumsum(lin_ucb_agent_rewards)

    # Regret
    random_agent_regrets = np.subtract(best_agent_rewards_cum, random_rewards_cum)
    best_agent_regrets = np.subtract(best_agent_rewards_cum, best_agent_rewards_cum)
    greedy_agent_regrets = np.subtract(best_agent_rewards_cum, greedy_agent_rewards_cum)
    ucb_agent_regrets = np.subtract(best_agent_rewards_cum, ucb_agent_rewards_cum)
    lin_ucb_agent_regrets = np.subtract(best_agent_rewards_cum, lin_ucb_agent_rewards_cum)

    plt.plot(random_agent_regrets, label='Random Agent')
    plt.plot(best_agent_regrets, label='Best Agent')
    plt.plot(greedy_agent_regrets, label='Optimal Agent')
    plt.plot(ucb_agent_regrets, label='UCB Agent')
    plt.plot(lin_ucb_agent_regrets, label='lin-UCB Agent')

    plt.legend()
    plt.xlabel('Time step')
    plt.ylabel('Regret')
    plt.show()

if __name__ == '__main__':
    FILENAME = 'CTR.txt'
    data = load_data(FILENAME)
    regret_vs_t(data['states'], data['rewards'])
