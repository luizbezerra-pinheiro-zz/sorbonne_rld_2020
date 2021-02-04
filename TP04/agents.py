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

# matplotlib.use("Qt5agg")
matplotlib.use("TkAgg")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DQNAgent(object):
    """DQN agent"""

    def __init__(self, env, opt, prior=False, mem_size=10000, mu=None, eps=0.1, layers=[],
                 update_R_steps=2, batch_size=32, lr=0.01, gamma=0.99):
        self.opt = opt
        self.env = env
        self.action_space = env.action_space
        self.featureExtractor = opt.featExtractor(env)

        # Initialize the networks
        self.input_size = self.featureExtractor.getFeatures(env.reset()).shape[0]
        self.output_size = self.action_space.n

        # We define the number of hidden layers according to the downscaling factor (k)
        self.memory = Memory(mem_size, prior=prior, p_upper=1., epsilon=.01, alpha=1, beta=1)
        self.Q = NN(self.input_size, self.output_size, layers=layers).double().to(device)
        self.R = NN(self.input_size, self.output_size, layers=layers).double().to(device)
        self.R.load_state_dict(self.Q.state_dict())
        self.R.requires_grad_(False)
        self.epsilon0 = 1
        self.mu = mu
        self.eps = eps
        self.test = False

        # training config
        self.iteration = 0
        self.r_steps = 0
        self.update_R_steps = update_R_steps
        self.gamma = gamma
        self.batch_size = batch_size
        self.criterion = SmoothL1Loss(reduction='sum')
        self.optimizer = optim.Adam(self.Q.parameters(), lr=lr)

        if opt.fromFile is not None:
            self.load(opt.fromFile)

    def act(self, observation, reward, done):
        # observation = observation.reshape(-1, self.input_size)  # batch format
        if self.mu:
            self.eps = self.epsilon0 / (1 + self.mu * self.iteration)
        if np.random.uniform(0, 1) > self.eps or self.test:
            _action = torch.argmax(self.Q.forward(observation), dim=-1)  # Exploitation
        else:
            _action = torch.tensor(self.action_space.sample())  # Exploration
        return _action

    def save(self, outputDir):
        torch.save(self.R.state_dict(), outputDir + '_R.pt')
        torch.save(self.Q.state_dict(), outputDir + '_Q.pt')

    def load(self, inputDir):
        self.R.load_state_dict(torch.load(inputDir + '_R.pt')).to(device)
        self.Q.load_state_dict(torch.load(inputDir + '_Q.pt')).to(device)

    def train_loop(self, last_ob, action, reward, ob, done, success):
        self.memory.store((last_ob, action, reward, ob, done, success))

        # sample random minibatch of transitions
        if self.memory.nentities > self.batch_size:
            out = self.memory.sample(self.batch_size)
        else:
            out = self.memory.sample(self.memory.nentities)
        if self.memory.prior:
            idx, w, batch = out
            max_w = max(w).item()
            w = torch.tensor(w / max_w)
            obs = torch.stack([el[0] for el in batch]).to(device)
            actions = torch.tensor([el[1] for el in batch], device=device)
            rewards = torch.tensor([el[2] for el in batch], device=device).double()
            new_obs = torch.stack([el[3] for el in batch]).to(device)
            dones = torch.tensor([el[4] for el in batch], device=device)
            successes = torch.tensor([el[5] for el in batch], device=device)
        else:
            batch = out
            idx = None
            w = torch.tensor([1] * len(batch))
            obs = torch.stack([el[0] for el in batch]).to(device)
            actions = torch.tensor([el[1] for el in batch], device=device)
            rewards = torch.tensor([el[2] for el in batch], device=device).double()
            new_obs = torch.stack([el[3] for el in batch]).to(device)
            dones = torch.tensor([el[4] for el in batch], device=device)
            successes = torch.tensor([el[5] for el in batch], device=device)

        # set y
        y_target = rewards + ((~dones).logical_or(successes)) * self.gamma * torch.max(self.R.forward(new_obs), -1)[0]
        # y_pred = self.Q.forward(obs)[actions]
        y_pred = torch.stack([el[ac] for el, ac in zip(self.Q.forward(obs), actions)])

        # compute loss + backward + optimize
        # loss = torch.tensor([criterion(y_i, y_t_i) for y_i, y_t_i in zip(y_pred, y_target)], requires_grad=True)
        # loss = torch.sum(w * loss)
        loss = self.criterion(y_pred, y_target)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.r_steps = (self.r_steps + 1) % (self.update_R_steps + 1)
        with torch.no_grad():
            self.memory.update(idx, torch.abs(y_target - y_pred).detach().cpu().numpy())
        if self.r_steps == self.update_R_steps:
            self.R.load_state_dict(self.Q.state_dict().copy())
            self.R.training = False
            self.r_steps = 0
