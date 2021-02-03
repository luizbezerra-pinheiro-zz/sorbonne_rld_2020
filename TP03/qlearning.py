import matplotlib
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import matplotlib.pyplot as plt
from utils import FPS, obs_to_state, evaluate_agent
matplotlib.use("TkAgg")


class QLearningAgent(object):
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
            while True:
                # Choose action
                state = obs_to_state(self.env, obs)

                if np.random.uniform(0, 1) < eps:  # Exploration
                    action = np.random.choice(self.env.action_space.n)
                else:
                    action = np.argmax(self.q[state])  # Exploitation

                obs, reward, done, _ = self.env.step(action)
                rsum += reward

                # Update q table
                new_state = obs_to_state(self.env, obs)
                self.q[state][action] = self.q[state][action] + _lr * (reward + _gamma * np.max(self.q[new_state]) - self.q[state][action])

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
    # env.render()  # permet de visualiser la grille du jeu

    episode_count = 1000
    lr = 0.25
    gamma = 1.0
    save_path = None

    # training
    agent = QLearningAgent(env)
    rewards_train = agent.train(episode_count, lr, gamma, render=False, verbose=False)
    print(f'End training, mean = {np.mean(rewards_train)}, std = {np.std(rewards_train)}')
    to_plot_x = range(0, episode_count, 50)
    to_plot_y = [rewards_train[i] for i in to_plot_x]

    plt.plot(to_plot_x, to_plot_y)
    plt.title('Learning Line')
    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

    # testing
    episode_count = 200
    rewards_test = evaluate_agent(env, agent)
    print(f'End testing, mean = {np.mean(rewards_test)}, std = {np.std(rewards_test)}')

    print("done")
    env.close()
