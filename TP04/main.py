import os
import argparse
import sys
import matplotlib
import gym
import gridworld
import torch
from utils import *
from torch.utils.tensorboard import SummaryWriter
from torch.nn import SmoothL1Loss
import torch.optim as optim
from memory import Memory
from agents import DQNAgent
from copy import copy
from tqdm import tqdm

# matplotlib.use("Qt5agg")
matplotlib.use("TkAgg")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run(problem='cartpole', agent='DQN', verbose_global=False, log_tb=False,
        prior=False, mu=30, eps=0.10, update_R_steps=1, batch_size=300,
        lr=1e-3, gamma=0.99, num_hidden_layers=2, size_hidden_layers=30, episode_count=None, **kwargs):
    assert problem in ['cartpole', 'gridworld', 'lunar']

    config_path = './configs/config_random_' + problem + '.yaml'
    config = load_yaml(config_path)

    ## Set config
    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]

    env = gym.make(config["env"])
    if hasattr(env, 'setPlan'):
        env.setPlan(config["map"], config["rewards"])

    tstart = str(time.time())
    tstart = tstart.replace(".", "_")
    outdir = "./XP/" + config["env"] + "/random_" + "-" + tstart

    env.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    episode_count = config["nbEpisodes"] if not episode_count else episode_count
    if verbose_global:
        print("Saving in " + outdir)
    os.makedirs(outdir, exist_ok=True)
    # save_src(os.path.abspath(outdir))
    write_yaml(os.path.join(outdir, 'info.yaml'), config)
    if log_tb:
        logger = LogMe(SummaryWriter(outdir))
        loadTensorBoard(outdir)

    # Agent config
    if agent == 'DQN':
        layers = [size_hidden_layers] * num_hidden_layers
        agent = DQNAgent(env, config, prior=prior, mu=mu, eps=eps, layers=layers,
                         update_R_steps=update_R_steps, batch_size=batch_size, lr=lr, gamma=gamma)
    else:
        raise NotImplementedError

    # Experimentation setup
    rsum = 0
    mean = 0
    itest = 0
    reward = 0
    done = False
    agent.test = False
    test_rewards = []
    # Experiment (train/test)
    for i in tqdm(range(episode_count), position=0, leave=True):
        if i % int(config["freqVerbose"]) == 0 and i >= config["freqVerbose"]:
            verbose = True
        else:
            verbose = False

        if i % freqTest == 0 and i >= freqTest:  ##### Same as train for now
            if verbose_global:
                print("Test time! ")
            mean = 0
            agent.test = True

        if i % freqTest == nbTest and i > freqTest:
            if verbose_global:
                print(str(i) + " - End of test, mean reward=", mean / nbTest)
            itest += 1
            if log_tb:
                logger.direct_write("rewardTest", mean / nbTest, itest)
            agent.test = False
            test_rewards += [mean / nbTest]

        if i % freqSave == 0:
            agent.save(outdir + "/save_" + str(i))

        j = 0
        if verbose and verbose_global:
            env.render()
        ob = env.reset()
        while True:
            if verbose and verbose_global:
                env.render()
            with torch.no_grad():
                last_ob = torch.tensor(agent.featureExtractor.getFeatures(ob),
                                       device=device).clone().double()  # preprocess
                action = agent.act(last_ob, reward, done).item()
                ob, reward, done, info = env.step(action)
                ob = torch.tensor(agent.featureExtractor.getFeatures(ob), device=device).clone().double()  # preprocess
                j += 1
                rsum += reward
                success = info.get('TimeLimit.truncated', False)

            # train Loop
            if not agent.test:
                agent.train_loop(last_ob, action, reward, ob, done, success)
            # end train loop

            if done:
                if verbose and verbose_global:
                    print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                if log_tb:
                    logger.direct_write("reward", rsum, i)
                agent.nbEvents = 0
                mean += rsum
                rsum = 0
                if not agent.test:
                    agent.iteration += 1  # only increase iteration for training iterations
                break
    env.close()
    return np.mean(test_rewards).item()


def generate_grid(grid_dict: dict):
    c_grid_dict = copy(grid_dict)
    keys = list(c_grid_dict.keys())
    assert len(keys) >= 1
    if len(keys) == 1:
        return [{keys[0]: v} for v in c_grid_dict[keys[0]]]
    else:
        # get last key:
        last_key = keys[-1]
        last_values = c_grid_dict[last_key]

        # remove last key from dict
        del c_grid_dict[last_key]

        # get the list without that key

        list_grid = generate_grid(c_grid_dict)

        # append the values of last_key on the list_grid

        return sum([[dict(d, **{last_key: v}) for d in list_grid] for v in last_values], [])


if __name__ == '__main__':
    # problem = 'lunar'
    # problem = 'gridworld'

    GRID_SEARCH = False

    config_params = {
        "problem": 'cartpole',
        "agent": 'DQN',
        "episode_count": 2000,
        "log_tb": False,
        "verbose_global": False
    }

    if GRID_SEARCH:
        params_grid = {"prior": [True],
                       "mu": [30],
                       "eps": [0.10],
                       "lr": [1e-3],
                       "gamma": [0.99],
                       "update_R_steps": [1, 2],
                       "batch_size": [32, 256],
                       "num_hidden_layers": [1, 2],
                       "size_hidden_layers": [16, 32, 64]}
        list_params_grid = generate_grid(params_grid)

        print("Total of grids: ", len(list_params_grid))
        best_reward = 0
        best_grid = None
        best_grid_idx = 0
        for i, params in enumerate(list_params_grid):
            reward = run(**config_params, **params)
            if reward > best_reward:
                best_reward = reward
                best_grid = params
                best_grid_idx = i
            if i % 10:
                print(f'Best grid ATM: ', best_grid)
                print(f'Best reward ATM: ', best_reward)
        print('Best params grid: ', best_grid)
        print('Got:', best_reward)
    else:
        # params = {"prior": False,
        #           "mu": 30,
        #           "eps": 0.10,
        #           "update_R_steps": 1,
        #           "batch_size": 300,
        #           "lr": 1e-3,
        #           "gamma": 0.99,
        #           "num_hidden_layers": 2,
        #           "size_hidden_layers": 30}
        params = {'prior': True,
                  'mu': 30,
                  'eps': 0.1,
                  'update_R_steps': 1,
                  'batch_size': 256,
                  'lr': 0.001,
                  'gamma': 0.99,
                  'num_hidden_layers': 2,
                  'size_hidden_layers': 64}
        reward = run(**config_params, **params)
        print("Reward: ", reward)
