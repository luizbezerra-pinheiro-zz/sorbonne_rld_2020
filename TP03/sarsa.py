import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
import matplotlib.pyplot as plt
from utils import FPS, obs_to_state, evaluate_agent


class SARSAAgent(object):
    def __init__(self, env):
        env.getMDP()  # only to get env.states
        n_actions = env.action_space.n
        n_states = len(env.states)
        self.env = env
        self.q = np.zeros((n_states, n_actions))

    def train(self, _episode_count=1000, _lr=0.25, _gamma=1.0, render=False, verbose=False, **kwargs):
        _rewards_train = []
        for i in range(_episode_count):
            obs = self.env.reset()
            self.env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
            if self.env.verbose and render:
                self.env.render(FPS)
            j = 0
            rsum = 0
            eps = 0.05 * ((_episode_count - 1 - i) / (_episode_count - 1))

            ## Choisir l'action à émetre a_j en fonction de Q
            state = obs_to_state(self.env, obs)
            if np.random.uniform(0, 1) < eps:  # Exploration
                action = np.random.choice(self.env.action_space.n)
            else:
                action = np.argmax(self.q[state])

            while True:
                ## émetre a_j
                last_action = action
                last_state = state
                obs, reward, done, _ = self.env.step(action)  ## New state, obs, reward
                rsum += reward

                state = obs_to_state(self.env, obs)

                # Choose action
                if np.random.uniform(0, 1) < eps:  # Exploration
                    action = np.random.choice(self.env.action_space.n)
                else:
                    action = np.argmax(self.q[state])

                # Update q table
                # new_state = obs_to_state(env, obs)
                self.q[last_state][last_action] = self.q[last_state][last_action] + _lr * (reward + _gamma * self.q[state][action] - self.q[last_state][last_action])

                j += 1
                if done:
                    _rewards_train += [rsum]
                    if self.env.verbose and verbose:
                        print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                    break
        return _rewards_train

    def act(self, state, *args):
        return np.argmax(self.q[state])


if __name__ == '__main__':
    env = gym.make("gridworld-v0")
    env.setPlan("gridworldPlans/plan0.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
    env.seed(0)  # Initialise le seed du pseudo-random
    env.verbose = False
    # env.render()  # permet de visualiser la grille du jeu

    episode_count = 1000
    lr = 0.25
    gamma = 1.0

    agent = SARSAAgent(env)

    # training
    rewards_train = agent.train(episode_count, lr, gamma, render=False, verbose=False)

    print(f'End training, mean = {np.mean(rewards_train)}, std = {np.std(rewards_train)}')

    to_plot_x = range(0, episode_count, 50)
    to_plot_y = [rewards_train[i] for i in to_plot_x]

    plt.plot(to_plot_x, to_plot_y)
    plt.title('Learning Line')
    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    plt.show()

    episode_count = 200
    rewards_test = evaluate_agent(env, agent, _episode_count=episode_count)
    print(f'End testing, mean = {np.mean(rewards_test)}, std = {np.std(rewards_test)}')

    print("done")
    env.close()
