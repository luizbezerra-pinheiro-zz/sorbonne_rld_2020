import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
import matplotlib.pyplot as plt


def obs_to_state(env, obs):
    obs_i = env.state2str(obs)
    return env.states[obs_i]


if __name__ == '__main__':

    env = gym.make("gridworld-v0")
    env.setPlan("gridworldPlans/plan0.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
    env.seed(0)  # Initialise le seed du pseudo-random
    # env.render()  # permet de visualiser la grille du jeu
    env.getMDP() # only to get env.states
    # env.render(mode="human")  # visualisation sur la console

    episode_count = 1000
    reward = 0
    done = False
    rsum = 0
    FPS = 0.0001

    lr = 0.25

    gamma = 1.0

    n_actions = env.action_space.n
    n_states = len(env.states)

    q = np.zeros((n_states, n_actions))
    # training
    rewards_train = []
    for i in range(episode_count):
        obs = env.reset()
        env.verbose = False#(i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render(FPS)
        j = 0
        rsum = 0
        eps = 0.05 * ((episode_count - 1 - i) / (episode_count-1))
        while True:
            # Choose action
            state = obs_to_state(env, obs)

            if np.random.uniform(0, 1) < eps:  # Exploration
                action = np.random.choice(env.action_space.n)
            else:
                action = np.argmax(q[state])

            obs, reward, done, _ = env.step(action)
            rsum += reward

            # Update q table
            new_state = obs_to_state(env, obs)
            q[state][action] = q[state][action] + lr * (reward + gamma * np.max(q[new_state]) - q[state][action])

            j += 1
            if env.verbose:
                env.render(FPS)
            if done:
                rewards_train += [rsum]
                #print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break
    print(f'End training, mean = {np.mean(rewards_train)}, std = {np.std(rewards_train)}')
    to_plot_x = range(0, episode_count, 50)
    to_plot_y = [rewards_train[i] for i in to_plot_x]

    plt.plot(to_plot_x, to_plot_y)
    plt.title('Learning Line')
    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    plt.show()
    # testing
    episode_count = 200
    rewards_test = []
    for i in range(episode_count):
        obs = env.reset()
        rsum = 0
        env.verbose = False # (i % 10 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render(FPS)
        while True:
            # Choose action
            state = obs_to_state(env, obs)
            action = np.argmax(q[state])
            obs, reward, done, _ = env.step(action)
            rsum += reward

            if env.verbose:
                env.render(FPS)
            if done:
                rewards_test += [rsum]
                break
    print(f'End testing, mean = {np.mean(rewards_test)}, std = {np.std(rewards_test)}')


    print("done")
    env.close()
