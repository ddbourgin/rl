import itertools
from time import time
from collections import defaultdict

import gym
import numpy as np

from utils import check_discrete, tile_state_space, plot_rewards, \
    sample_gaussian


class LinearPolicyGradLearner(object):
    """
    An linear policy gradient learner for tile-coded continuous observation
    spaces. Also handles continuous action spaces.
    """

    def __init__(self, env, **kwargs):
        self.__validate_env__(env)

        # a good heuristic for the learning rate is (1/10. * n_tiles)
        self.value_learning_rate = value_learning_rate
        self.policy_learning_rate = policy_learning_rate
        self.episode_len = max_episode_len
        self.gamma = discount_factor
        self.epsilon = epsilon

        # only use tiling parameters if the observation space is continuous
        if not self.discrete_obs:
            self.n_tilings = n_tilings
            self.grid_dims = grid_dims
            self.n_tiles = np.prod(grid_dims) * n_tilings
        else:
            self.n_tilings = None
            self.grid_dims = None
            self.n_tiles = self.n_obs

        # initialize policy weights
        if self.discrete_actions:
            self.policy_weights = np.zeros([self.n_tiles, self.n_actions])
        else:
            # separate weights for mean + sd
            self.policy_weights = np.zeros(
                [self.n_tiles * 2, self.n_action_dims])

        # initialize value weights
        self.value_weights = np.zeros(self.n_tiles)

    def __validate_env__(self, env):
        is_multi_obs, is_multi_act, is_disc_obs, is_disc_act = \
            check_discrete(env, 'Policy Gradient', obs=False, action=False)

        self.discrete_actions = is_disc_act
        self.discrete_obs = is_disc_obs

        sample = env.action_space.sample()
        self.n_action_dims = len(sample) if isinstance(sample, list) else 1

        # initialize a dictionary mapping actions to unique integers if the
        # action space is discrete
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

        # compute number of possible observations, if observation space is
        # discrete
        if is_disc_obs:
            if is_multi_obs:
                self.n_obs = \
                    np.prod([space.n for space in env.observation_space.spaces])
            else:
                self.n_obs = env.observation_space.n

        # initialize discretization function for continuous observation spaces
        self.discretize = lambda x: x
        if not is_disc_obs:
            self.discretize, _ = \
                tile_state_space(env, n_tilings, grid_dims)

        # initialize obs -> scalar dictionaries
        self.obs2num = {}
        self.num2obs = {}

    def approx_policy(self, s_vec, a=None):
        if self.discrete_actions:
            # use a softmax policy for discrete action spaces
            prefs = np.array([self.preferences(s_vec, aa)
                              for aa in range(self.n_actions)])
            action_probs = np.exp(prefs) / np.sum(np.exp(prefs))

            if a is not None:
                return action_probs[a]

            action_num = np.random.multinomial(1, action_probs).argmax()
            action = self.num2action[action_num]
        else:
            # use a gaussian policy for continuous action spaces
            mu_weights = self.policy_weights[:self.n_tiles, :].T
            sigma_weights = self.policy_weights[self.n_tiles:, :].T
            mu = np.dot(mu_weights, s_vec)
            sigma = np.exp(np.dot(sigma_weights, s_vec))

            if a is not None:
                return sample_gaussian(mu, sigma, a)

            action = sample_gaussian(mu, sigma)
            # TODO: should check if the action is outside of a bounded
            # continuous action space here

        if isinstance(action, tuple):
            if any([np.isnan(a) for a in action]):
                import ipdb; ipdb.set_trace()
        elif np.isnan(action):
            import ipdb; ipdb.set_trace()

        return action

    def preferences(self, s_vec, a):
        # linear preferences with binary features
        return np.dot(self.policy_weights[:, a], s_vec)

    def approx_value(self, s_vec):
        # assume a linear value approximation
        return np.dot(self.value_weights, s_vec)

    def __encode_obs__(self, obs):
        # incrementally build obs2num/num2obs with experience
        obs = self.discretize(obs)
        if obs in self.obs2num.keys():
            obs_num = self.obs2num[obs]
        else:
            obs_num = 0
            if len(self.obs2num) > 0:
                obs_num = np.max(self.obs2num.values()) + 1
            self.obs2num[obs] = obs_num
            self.num2obs[obs_num] = obs
        return obs_num

    def run_episode(self, env, render=False):
        obs = env.reset()

        I = 1.0

        # map observation to a unique scalar
        s = self.__encode_obs__(obs)

        # create binary state vector
        s_vec = np.zeros(self.n_tiles)
        obs = self.num2obs[s]
        idxs = list(obs) if isinstance(obs, tuple) else [obs]
        s_vec[idxs] = 1.

        # run one episode of the RL problem
        reward_history = []
        for _ in xrange(self.episode_len):
            if render:
                env.render()

            # generate an action using our parameterized policy
            action = self.approx_policy(s_vec)

            if self.discrete_actions:
                a = self.action2num[action]
            else:
                a = action

            # take action
            obs_, reward, done, _ = env.step(action)
            s_ = self.__encode_obs__(obs_)

            s_vec_ = np.zeros(self.n_tiles)
            d_obs_ = self.num2obs[s_]
            idxs_ = list(d_obs_) if isinstance(d_obs_, tuple) else [d_obs_]
            s_vec_[idxs_] = 1.

            # record rewards
            reward_history.append(reward)

            # compute delta
            v_s = self.approx_value(s_vec)
            v_s_ = self.approx_value(s_vec_) if not done else 0.
            delta = reward + (self.gamma * v_s_) - v_s

            # update the value function weights
            self.value_weights += self.value_learning_rate * delta * s_vec

            # compute eligibility vector
            if self.discrete_actions:
                # for linear function approximation with binary features and
                # softmax policy approximation
                vals = [self.approx_policy(s_vec, aa) * s_vec \
                            for aa in xrange(self.n_actions)]
                eligibility_vector = s_vec - np.sum(vals, axis=0)
            else:
                # this is incorrect - eligibility vector should be 2x
                # len(s_vec), since we have double weights for mean and sd....
                eligibility_vector = s_vec / self.approx_policy(s_vec, a)

            # update policy weights
            if self.discrete_actions:
                self.policy_weights[:, a] += \
                    self.policy_learning_rate * I * delta * eligibility_vector
            else:
                self.policy_weights += \
                    self.policy_learning_rate * I * delta * eligibility_vector

            if done:
                break

            I *= self.gamma

            # update observations
            obs, s, s_vec = obs_, s_, s_vec_

        return np.sum(reward_history)

if __name__ == "__main__":
    # initialize RL environment
    env = gym.make('Copy-v0')

    # initialize run parameters
    n_epochs = 5000       # number of episodes to train the q network on
    epsilon = 0.10          # for epsilon-greedy policy (discrete case)
    max_episode_len = 200   # max number of timesteps per episode/epoch
    discount_factor = 1.0   # temporal discount factor
    render = False          # render runs during training
    n_tilings = 8           # number of tilings to use if state space is continuous
    grid_dims = [8, 8]      # number of squares in the tiling grid if state
    # space is continuous

    # good heuristic for the learning rate is one-tenth the reciporical of the
    # total number of tiles
    n_tiles = np.prod(grid_dims) * n_tilings
    value_learning_rate = 0.001
    policy_learning_rate = 0.001

    pg_params = \
        {'epsilon': epsilon,
         'grid_dims': grid_dims,
         'n_tilings': n_tilings,
         'value_learning_rate': value_learning_rate,
         'policy_learning_rate': policy_learning_rate,
         'max_episode_len': max_episode_len,
         'discount_factor': discount_factor}

    # initialize network and experience replay objects
    pg_learner = LinearPolicyGradLearner(env, **pg_params)

    # train monte carlo learner
    t0 = time()
    episode_rewards = []
    for idx in xrange(n_epochs):
        total_reward = pg_learner.run_episode(env)

        print('Total reward on epoch {}/{}:\t{}'
              .format(idx + 1, n_epochs, total_reward))

        episode_rewards.append(total_reward)

    print('\nTraining took {} mins'.format((time() - t0) / 60.))

    env_name = env.spec.id
    param_str = 'policy_lr{:.3E}_value_lr{:.3E}_ntilings{}_griddim{}x{}.png'\
                    .format(policy_learning_rate, value_learning_rate,
                            n_tilings, *grid_dims)
    save_path = 'plots/policy_gradient_{}.png'.format(param_str)
    plot_rewards(episode_rewards, save_path, env_name)

    print('Executing greedy policy\n')

    pg_learner.epsilon = 0
    for idx in xrange(10):
        total_reward = pg_learner.run_episode(env, render=True)

        print('Total reward on greedy epoch {}/{}:\t{}\n'
              .format(idx + 1, 10, total_reward))
