import itertools
from time import time
from collections import defaultdict

import gym
import numpy as np

from utils import check_discrete, tile_state_space, plot_rewards


class LinearPolicyGradLearner(object):
    """
    An linear policy gradient learner for tile-coded continuous observation
    spaces. Also handles continuous action spaces.
    """

    def __init__(self, env, **kwargs):
        self.__validate_env__(env)

        # a good heuristic for the learning rate is (1/10. * n_tiles)
        self.learning_rate = learning_rate
        self.episode_len = max_episode_len
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.n_tilings = n_tilings
        self.grid_dims = grid_dims

        # initialize policy weights
        self.n_tiles = np.prod(grid_dims) * n_tilings

        if self.discrete_actions:
            self.weights = np.zeros(self.n_tiles)
        else:
            # separate weights for mean + sd
            self.weights = np.zeros(self.n_tiles * 2)

    def __validate_env__(self, env):
        is_multi_obs, is_multi_act, is_disc_obs, is_disc_act = \
            check_discrete(env, 'Linear Semi-Gradient', obs=False, act=False)

        self.discrete_actions = is_disc_act
        self.discrete_obs = is_disc_obs

        sample = env.action_space.sample()
        self.n_action_dims = len(sample) if isinstance(sample, list) else 1.

        # action space is multidimensional
        if is_disc_act:
            if is_multi_act:
                n_actions = [space.n for space in env.action_space.spaces]
                one_dim_action = list(
                    itertools.product(*[range(i) for i in n_actions]))
            else:
                n_actions = env.action_space.n
                one_dim_action = range(n_actions)

            self.n_actions = np.prod(n_actions)

            # create action -> scalar dictionaries
            self.action2num = {act: i for i, act in enumerate(one_dim_action)}
            self.num2action = {i: act for act, i in self.action2num.items()}

        # initialize discretization function for continuous observation spaces
        self.discretize = lambda x: x
        if not is_disc_obs:
            self.discretize, _ = \
                tile_state_space(env, n_tilings, grid_dims)

        # initialize obs -> scalar dictionaries
        self.obs2num = {}
        self.num2obs = {}

    def approx_policy(self, s):
        if self.discrete_actions:
            # use a softmax policy for discrete action spaces
            prefs = np.array([self.preferences(s, aa) for aa in range(self.n_actions)])
            action_probs = np.exp(prefs) / np.sum(np.exp(prefs))
            action_num = np.random.multinomial(1, action_probs).argmax()
            action = self.num2action[action_num]
        else:
            # use a gaussian policy for continuous action spaces
            mu_weights = self.weights[:self.n_action_dims]
            sigma_weights = self.weights[self.n_action_dims:]
            mu = np.dot(mu_weights, s)
            sigma = np.exp(np.dot(sigma_weights, s))
            action = sample_gaussian(mu, sigma)
            # TODO: should check if the action is outside of a bounded
            # continuous action space here
        return action

    def __encode_obs__(self, obs):
        # incrementally build obs2num/num2obs with experience
        obs = self.discretize(obs)

        if any([type(i) == float for i in obs]):
            import ipdb
            ipdb.set_trace()

        if obs in self.obs2num.keys():
            obs_num = self.obs2num[obs]
        else:
            obs_num = 0
            if len(self.obs2num) > 0:
                obs_num = np.max(self.obs2num.values()) + 1
            self.obs2num[obs] = obs_num
            self.num2obs[obs_num] = obs
        return obs_num

    def preferences(self, s, a):
        # linear preferences with binary features
        obs = self.num2obs[s]
        q_val = np.sum([self.weights[i, a] for i in obs])
        return q_val

    def update_policy_weights(self, s, a, reward, s_, a_):
        Q_s_a_ = 0.0
        EQ_s_a_ = 0.0
        if s_ is not None and a_ is not None:
            Q_s_a_ = self.Q(s_, a_)
            EQ_s_a_ = np.sum(
                [self.epsilon_soft_policy(s_, aa) * self.Q(s_, aa) for aa in xrange(self.n_actions)]
            )

        # create binary state vector, which in the linear approximation case
        # corresponds to the gradient of Q(s, a) w.r.t. the weights
        state_vec = np.zeros(self.weights.shape[0])
        for i in self.num2obs[s]:
            state_vec[i] = 1.

        # semi-gradient SARSA update
        update = \
            self.learning_rate * \
            (reward + self.gamma * EQ_s_a_ - self.Q(s, a)) * \
            state_vec

        if any(np.isnan(update)):
            # this often happens if the learning rate is too high, causing the
            # weights to diverge to infinity
            import ipdb
            ipdb.set_trace()

        self.weights[:, a] += update

    def run_episode(self, env, render=False):
        obs = env.reset()

        # map observation to a unique scalar
        s = self.__encode_obs__(obs)

        # generate an action using an epsilon-soft policy
        action = self.epsilon_soft_policy(s)
        a = self.action2num[action]

        # run one episode of the RL problem
        reward_history = []
        for xx in xrange(self.episode_len):
            if render:
                env.render()

            # take action
            obs_, reward, done, info = env.step(action)
            s_ = self.__encode_obs__(obs_)

            # record rewards
            reward_history.append(reward)

            if done or xx == (self.episode_len - 1):
                # semi-gradient SARSA update
                self.update_policy_weights(s, a, reward, None, None)
                break

            # generate a new action using an epsilon-soft policy
            action_ = self.epsilon_soft_policy(s_)
            a_ = self.action2num[action_]

            # semi-gradient SARSA update
            self.update_policy_weights(s, a, reward, s_, a_)

            # update observations and actions
            obs, s, action, a = obs_, s_, action_, a_

        return np.sum(reward_history)

if __name__ == "__main__":
    # initialize RL environment
    env = gym.make('MountainCar-v0')

    # initialize run parameters
    n_epochs = 5000       # number of episodes to train the q network on
    epsilon = 0.10          # for epsilon-soft policy during training
    max_episode_len = 200   # max number of timesteps per episode/epoch
    discount_factor = 1.0   # temporal discount factor
    render = False          # render runs during training
    n_tilings = 8           # number of tilings to use if state space is continuous
    grid_dims = [8, 8]      # number of squares in the tiling grid if state
                            # space is continuous
    # good heuristic for the learning rate is one-tenth the reciporical of the
    # total number of tiles
    n_tiles = np.prod(grid_dims) * n_tilings
    learning_rate = 0.1 * (1. / n_tilings)

    pg_params = \
        {'epsilon': epsilon,
         'grid_dims': grid_dims,
         'n_tilings': n_tilings,
         'learning_rate': learning_rate,
         'max_episode_len': max_episode_len,
         'discount_factor': discount_factor}

    # initialize network and experience replay objects
    pg_learner = LinearPolicyGradLearner(env, **pg_params)

    # train monte carlo learner
    t0 = time()
    epoch_reward = []
    for idx in xrange(n_epochs):
        total_reward = pg_learner.run_episode(env)

        print('Total reward on epoch {}/{}:\t{}'
              .format(idx + 1, n_epochs, total_reward))

        epoch_reward.append(total_reward)

    print('\nTraining took {} mins'.format((time() - t0) / 60.))

    plot_rewards(epoch_reward, smooth=True)

    print('Executing greedy policy\n')

    pg_learner.epsilon = 0
    for idx in xrange(10):
        total_reward = pg_learner.run_episode(env, render=True)

        print('Total reward on greedy epoch {}/{}:\t{}\n'
              .format(idx + 1, 10, total_reward))