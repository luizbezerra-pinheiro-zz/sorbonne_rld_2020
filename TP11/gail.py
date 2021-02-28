import argparse
import os
import random
import sys
import time
import pickle
import gym
import matplotlib
import numpy as np
import torch
import torch.nn
import torch.optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from utils import *
from rollout import RolloutCollector, moving_average

# matplotlib.use("Qt5agg")
matplotlib.use("TkAgg")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WARM_UP = 1000


class BaseAgent(object):
    """Base agent for DQN."""

    def __init__(self, env, opt):
        self.opt = opt
        self.env = env
        if opt.fromFile is not None:
            self.load(opt.fromFile)
        self.action_space = env.action_space
        self.featureExtractor = opt.featExtractor(env)

    def act(self, observation):
        pass

    def learn(self, batch):
        pass

    def save(self, outputDir):
        pass

    def load(self, inputDir):
        pass

class Expert(nn.Module):
    def __init__(self, nbFeatures, nbActions, expert_file):
        super(Expert, self).__init__()
        self.nbFeatures = nbFeatures
        self.nbActions = nbActions
        self.floatTensor = torch.FloatTensor().to(device)
        self.longTensor = torch.LongTensor().to(device)
        with open(expert_file, 'rb') as handle:
            expert_data = pickle.load(handle).to(self.floatTensor)
            expert_states = expert_data[:, :self.nbFeatures]
            expert_actions = expert_data[:, self.nbFeatures:]
            self.expert_states = expert_states.contiguous()
            self.expert_actions = expert_actions.contiguous()

    def toIndexAction(self, oneHot):
        ac = self.longTensor.new(range(self.nbActions)).view(1, -1)
        ac = ac.expand(oneHot.size()[0], -1).contiguous().view(-1)
        actions = ac[oneHot.view(-1) > 0].view(-1)
        return actions

    def get_couples_expert(self, n_couples):
        # choose n_couples idx
        indices = np.random.choice(self.expert_states.size(0), n_couples)

        return self.expert_states[indices], self.toIndexAction(self.expert_actions[indices])

class Discriminator(nn.Module):
    def __init__(self, nbFeatures, nbActions):
        super(Discriminator, self).__init__()
        self.nbFeatures = nbFeatures
        self.nbActions = nbActions
        self.main = nn.Sequential(
            nn.Linear(self.nbFeatures + self.nbActions, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )
        self.floatTensor = torch.FloatTensor().to(device)
        self.longTensor = torch.LongTensor().to(device)

    def toOneHot(self, actions):
        if not torch.is_tensor(actions):
            actions = torch.tensor(actions)
        actions = actions.view(-1).to(self.longTensor)
        oneHot = torch.zeros(actions.size()[0], self.nbActions).to(self.floatTensor)
        oneHot[range(actions.size()[0]), actions] = 1
        return oneHot

    def forward(self, obs, actions, train=False):
        x = torch.cat([obs, self.toOneHot(actions)], -1)
        if train:
            x += torch.randn(x.shape) * 1E-4
        return self.main(x)

class GailAgent(BaseAgent):
    def __init__(self,
                 env, opt, p_layers=[64, 32], d_layers=[100, 100], v_layers=[100, 100], gamma=0.99, lr=3e-4,
                 grad_clip_val=1.0, batch_size=1000, clip_ratio=0.2, ent_coef=1e-3,
                 p_train_steps=10, d_train_steps=10, v_train_steps=1):

        super(GailAgent, self).__init__(env, opt)

        # training options
        self.gamma = opt.get('gamma', gamma)
        self.batch_size = opt.get('batch_size', batch_size)
        self.clip_val = opt.get('clipVal', grad_clip_val)
        self.ent_coef = opt.get('entropyCoef', ent_coef)
        self.p_train_steps = opt.get('policyTrainSteps', p_train_steps)
        self.d_train_steps = opt.get('discriminatorTrainSteps', d_train_steps)
        self.v_train_steps = opt.get('criticTrainSteps', v_train_steps)
        self.clip_ratio = opt.get('clipRatio', clip_ratio)

        # optimizer options
        self.lr = opt.get('learningRate', lr)

        self.test = False  # flag for testing mode

        # policy network
        obs_size, out_size = self.featureExtractor.outSize, env.action_space.n
        p_layers = opt.get('policyLayers', p_layers)
        self.P = NN(obs_size, out_size, layers=p_layers)

        # value network (critique)
        v_layers = opt.get('valueLayers', v_layers)
        self.V = NN(obs_size, 1, layers=v_layers)

        # discriminator network
        # d_layers = opt.get('discriminatorLayers', d_layers)
        self.D = Discriminator(obs_size, out_size)  #NN(obs_size+out_size, 1, layers=d_layers)

        # load expert
        self.expert = Expert(obs_size, out_size, 'expert.pkl')

        # optimizer and value loss
        self.v_loss_fn = torch.nn.SmoothL1Loss()
        self.p_optim = torch.optim.Adam(self.P.parameters(), self.lr)
        self.d_optim = torch.optim.Adam(self.D.parameters(), self.lr)
        self.v_optim = torch.optim.Adam(self.V.parameters(), self.lr)

    def learn(self, batch):
        obs, act, old_logp, tgt, path_start_indices, n_paths = batch
        batch_size = obs.shape[0]
        # take K training steps for discriminator network
        for _ in range(self.d_train_steps):
            # load expert trajectories of the same size
            couples_expert = self.expert.get_couples_expert(batch_size)
            # log expert
            fake_reward_expert = torch.sum(torch.log(self.D(*couples_expert, train=True)))
            fake_reward_pi = 0
            for i in range(n_paths):
                # compute the fake rewards
                if i < n_paths - 1:
                    path_slice = slice(path_start_indices[i], path_start_indices[i+1])
                else:
                    path_slice = slice(path_start_indices[i], batch_size)
                couple_pi = obs[path_slice], act[path_slice]
                # log police
                fake_reward_pi += torch.sum(torch.log(1-self.D(*couple_pi, train=True)))

            self.d_optim.zero_grad()
            loss_discriminator = -(fake_reward_expert + fake_reward_pi) / n_paths
            loss_discriminator.backward()
            torch.nn.utils.clip_grad_norm_(self.D.parameters(), self.clip_val)
            self.d_optim.step()

        # compute fake_rewards_togo
        fake_rewards_togo = torch.zeros((len(obs),))
        for i in range(n_paths):
            # compute the fake rewards
            if i < n_paths - 1:
                path_slice = slice(path_start_indices[i], path_start_indices[i + 1])
            else:
                path_slice = slice(path_start_indices[i], batch_size)

            couple_pi = obs[path_slice], act[path_slice]

            fake_rewards_togo[path_slice] = moving_average(torch.log(self.D(*couple_pi)))
        # mean_, std_ = torch.mean(fake_rewards_togo), torch.std(fake_rewards_togo)
        # fake_rewards_togo = ((fake_rewards_togo - mean_) / std_).detach()
        # fake_rewards_togo = fake_rewards_togo
        adv = (fake_rewards_togo - self.V(obs)).detach()
        # mean_, std_ = torch.mean(adv), torch.std(adv)
        # adv = ((adv - mean_) / std_)
        # take K training steps for the policy network
        for _ in range(self.p_train_steps):
            dist, logp = self._compute_policy_dist(obs, act)

            ratio = torch.exp(logp - old_logp)


            clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv

            p_loss = -(torch.min(ratio * adv, clip_adv)).mean()
            entropy = torch.mean(dist.entropy())

            self.p_optim.zero_grad()
            loss = p_loss + self.ent_coef * entropy
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.P.parameters(), self.clip_val)
            self.p_optim.step()

        # fit the value network
        for _ in range(self.v_train_steps):
            self.v_optim.zero_grad()
            v_loss = self.v_loss_fn(self.V(obs), tgt)
            v_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.V.parameters(), self.clip_val)
            self.v_optim.step()
        # compute scores
        with torch.no_grad():
            couples_expert = self.expert.get_couples_expert(batch_size)
            fake_reward_expert = self.D(*couples_expert).mean()
            fake_reward_pi = self.D(obs, act).mean()
        print('Losses: ', loss_discriminator.detach().item(),  loss.detach().item())
        return fake_reward_expert.item(), fake_reward_pi.item(), loss_discriminator.detach().item(),  loss.detach().item()

    def _compute_policy_dist(self, obs, act=None):
        logits = self.P(obs)
        dist = Categorical(logits=logits)
        if act is not None:
            logp = dist.log_prob(act)
            return dist, logp
        return dist, None

    def act(self, observation):
        with torch.no_grad():
            obs = torch.tensor(
                self.featureExtractor.getFeatures(observation),
                dtype=torch.float32)
            # value = self.V(obs)
            dist, _ = self._compute_policy_dist(obs)
            action = dist.sample()
            logp = dist.log_prob(action)
            # reward_d = agent.D(obs, action)[0]
        return action.item(), logp

    def save(self, save_dir):
        torch.save(self.P.state_dict(), save_dir + '_P.pt')
        torch.save(self.D.state_dict(), save_dir + '_D.pt')
        torch.save(self.V.state_dict(), save_dir + '_V.pt')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='lunar',
                        help="Runner environment, either 'gridworld', 'cartpole' or 'lunar'")
    parser.add_argument('--variant', type=str, default='clipped',
                        help="PPO variant to use, either 'adaptative' or 'clipped'.")
    parser.add_argument('--lam', type=float, default=0.97,
                        help='Lambda parameter for calculating targets and advantages.')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='If set, will render an episode occasionally.')
    args = parser.parse_args()

    # load configs
    config = load_yaml('./configs/config_random_{}.yaml'.format(args.env))
    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]

    # create environment
    env = gym.make(config["env"])
    if hasattr(env, 'setPlan'):
        env.setPlan(config["map"], config["rewards"])

    # set experiment directory
    tstart = str(time.time())
    tstart = tstart.replace(".", "_")
    name = '{}_ppo_lam{}'.format(args.variant, args.lam)
    outdir = "./XP/" + config["env"] + "/" + name + "_" + tstart

    # seed rngs
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    agent = GailAgent(env, config)
    obs_shape, act_shape = env.observation_space.shape, env.action_space.shape


    collector = RolloutCollector(obs_shape, act_shape, config['batchSize'])


    print("Saving in " + outdir)
    os.makedirs(outdir, exist_ok=True)
    save_src(os.path.abspath(outdir))
    write_yaml(os.path.join(outdir, 'info.yaml'), config)
    logger = LogMe(SummaryWriter(outdir))
    loadTensorBoard(outdir)

    # assert 1==0
    episode, timestep = 0, 0
    rsum, mean = 0, 0
    verbose = True
    itest, test_start = 0, 0
    obs, reward, done = env.reset(), 0, False
    while timestep < config['nbTimesteps']:
        if (args.verbose and
                episode % int(config["freqVerbose"]) == 0 and
                episode >= config["freqVerbose"]):
            verbose = True
        else:
            verbose = False

        if episode % freqTest == 0 and episode >= freqTest:  ##### Same as train for now
            print("Test time! ")
            mean = 0
            agent.test = True

        if episode % freqTest == nbTest and episode > freqTest:
            print("End of test, mean reward=", mean / nbTest)
            itest += 1
            logger.direct_write("rewardTest", mean / nbTest, itest)
            agent.test = False

        if episode % freqSave == 0:
            agent.save(outdir + "/save_" + str(episode))

        ep_steps = 0
        if verbose:
            env.render()

        while True:
            if verbose:
                env.render()

            action, logp = agent.act(obs)
            next_obs, reward, done, info = env.step(action)
            # assert 1==;
            # The reward is actually saved as D_w (s, a)

            ep_steps += 1

            if not agent.test:
                if collector.is_full():
                    collector.finish_path()
                    batch = collector.get()
                    score_expert, score_pi, _, _ = agent.learn(batch)
                    logger.direct_write("score_expert", score_expert, timestep)
                    logger.direct_write("score_pi", score_pi, timestep)
                truncated = info.get('TimeLimit.truncated', False)
                collector.store(obs, action, reward, done and not truncated, logp)
                timestep += 1

            obs = next_obs
            rsum += reward
            if done:
                print(str(episode) + " rsum=" + str(rsum) + ", " + str(ep_steps) + " actions ")
                logger.direct_write("reward", rsum, episode)
                agent.nbEvents = 0
                mean += rsum
                rsum = 0
                obs = env.reset()
                break

        episode += 1

    env.close()
