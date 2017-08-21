import itertools
from time import time
import ipdb

import gym
import numpy as np

from utils import check_discrete, tile_state_space, plot_rewards


class LinearSemiGradLearner(object):
    """
    An linear semi-gradient expected SARSA-learner for tile-coded continuous
    observation spaces.

    NB. tends to be highly sensitive to the learning rate parameter.
    """

    def __init__(self, env, **kwargs):
        self.__validate_env__(env)

        self.learning_rate = kwargs['learning_rate']
        self.episode_len = kwargs['max_episode_len']
        self.gamma = kwargs['discount_factor']
        self.epsilon = kwargs['epsilon']
        self.n_tilings = kwargs['n_tilings']
        self.grid_dims = kwargs['grid_dims']

        self.n_tiles = np.prod(self.grid_dims) * self.n_tilings
        self.Q_weights = np.zeros((self.n_tiles, self.n_actions))

    def __validate_env__(self, env):
        is_multi_obs, is_multi_act, is_disc_obs, is_disc_act = \
            check_discrete(env, 'Linear Semi-Gradient', obs=False)

        # action space is multidimensional
        if is_multi_act:
            n_actions = [space.n for space in env.action_space.spaces]
            one_dim_action = list(
                itertools.product(*[range(i) for i in n_actions]))
        else:
            n_actions = env.action_space.n
            one_dim_action = range(n_actions)

        # record number of possible (discrete) actions
        self.n_actions = np.prod(n_actions)

        # create action -> scalar dictionaries
        self.action2num = {act: i for i, act in enumerate(one_dim_action)}
        self.num2action = {i: act for act, i in self.action2num.items()}

        # initialize obs -> scalar dictionaries
        self.obs2num = {}
        self.num2obs = {}

        # initialize discretization function for continuous observation spaces
        self.discretize = lambda x: x
        if not is_disc_obs:
            self.discretize, _ = \
                tile_state_space(env, n_tilings, grid_dims)

    def epsilon_soft_policy(self, s, action=None):
        """
        pi(a|s) = 1 - epsilon + (epsilon / |A(s)|) IFF a == a*
        pi(a|s) = epsilon / |A(s)|                 IFF a != a*
        """
        a_star = np.argmax([self.Q(s, aa) for aa in xrange(self.n_actions)])
        p_a_star = 1. - self.epsilon + (self.epsilon / self.n_actions)
        p_a = self.epsilon / self.n_actions

        action_probs = np.ones(self.n_actions) * p_a
        action_probs[a_star] = p_a_star

        if action is not None:
            return action_probs[action]

        # sample action
        action = np.random.multinomial(1, action_probs)
        return self.num2action[action.argmax()]

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

    def Q(self, s, a):
        # linear function approximation with binary features
        obs = self.num2obs[s]
        q_val = np.sum([self.Q_weights[i, a] for i in obs])
        return q_val

    def on_policy_update(self, s, a, reward, s_, a_):
        EQ_s_a_ = 0.0
        if s_ is not None and a_ is not None:
            EQ_s_a_ = np.sum(
                [self.epsilon_soft_policy(s_, aa) * self.Q(s_, aa)
                 for aa in xrange(self.n_actions)]
            )

        # create binary state vector, which in the linear approximation case
        # corresponds to the gradient of Q(s, a) w.r.t. the weights
        state_vec = np.zeros(self.Q_weights.shape[0])
        for i in self.num2obs[s]:
            state_vec[i] = 1.

        # semi-gradient expected SARSA update
        update = \
            self.learning_rate * \
            (reward + self.gamma * EQ_s_a_ - self.Q(s, a)) * \
            state_vec

        if any(np.isnan(update)):
            # this often happens if the learning rate is too high, causing the
            # weights to diverge to infinity
            ipdb.set_trace()

        self.Q_weights[:, a] += update

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
            obs_, reward, done, _ = env.step(action)
            s_ = self.__encode_obs__(obs_)

            # record rewards
            reward_history.append(reward)

            if done or xx == (self.episode_len - 1):
                self.on_policy_update(s, a, reward, None, None)
                break

            # generate a new action using an epsilon-soft policy
            action_ = self.epsilon_soft_policy(s_)
            a_ = self.action2num[action_]

            # semi-gradient SARSA update
            self.on_policy_update(s, a, reward, s_, a_)

            # update observations and actions
            obs, s, action, a = obs_, s_, action_, a_

        return np.sum(reward_history)

if __name__ == "__main__":
    # initialize RL environment
    env = gym.make('MountainCar-v0')

    # initialize run parameters
    n_epochs = 5000         # number of episodes to train the q network on
    epsilon = 0.10          # for epsilon-soft policy during training
    max_episode_len = 200   # max number of timesteps per episode/epoch
    discount_factor = 1.0   # temporal discount factor
    render = False          # render runs during training
    n_tilings = 8           # number of tilings to use if state space is continuous
    grid_dims = [8, 8]      # number of squares in the tiling grid if state
    # space is continuous

    # good rule of thumb for the learning rate is one-tenth the reciporical of
    # the total number of tiles (assuming continuous state space)
    n_tiles = np.prod(grid_dims) * n_tilings
    learning_rate = 0.1 * (1. / n_tilings)

    sg_params = \
        {'epsilon': epsilon,
         'grid_dims': grid_dims,
         'n_tilings': n_tilings,
         'learning_rate': learning_rate,
         'max_episode_len': max_episode_len,
         'discount_factor': discount_factor}

    # initialize LinearSemiGrad learner instance
    sg_learner = LinearSemiGradLearner(env, **sg_params)

    # train semi-gradient learner
    t0 = time()
    episode_rewards = []
    for idx in xrange(n_epochs):
        total_reward = sg_learner.run_episode(env)

        print('Total reward on epoch {}/{}:\t{}'
              .format(idx + 1, n_epochs, total_reward))

        episode_rewards.append(total_reward)

    print('\nTraining took {} mins'.format((time() - t0) / 60.))

    env_name = env.spec.id
    param_str = 'lr{:.3E}_ntilings{}_griddim{}x{}.png'\
        .format(learning_rate, n_tilings, *grid_dims)
    save_path = 'plots/linear_semi_gradient_{}.png'.format(param_str)
    plot_rewards(episode_rewards, save_path, env_name)

    print('Executing greedy policy\n')

    sg_learner.epsilon = 0
    for idx in xrange(10):
        total_reward = sg_learner.run_episode(env, render=True)

        print('Total reward on greedy epoch {}/{}:\t{}\n'
              .format(idx + 1, 10, total_reward))
