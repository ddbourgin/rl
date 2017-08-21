import itertools
from time import time
from collections import defaultdict

import gym
import numpy as np

from utils import check_discrete, tile_state_space, plot_rewards

class TDLearner(object):
    """
    Temporal difference learning agent with SARSA (on-policy) and TD(0)
    Q-learning (off-policy) updates
    """

    def __init__(self, env, **kwargs):
        self.__validate_env__(env)

        self.learning_rate = kwargs['learning_rate']
        self.episode_len = kwargs['max_episode_len']
        self.off_policy = kwargs['off_policy']
        self.gamma = kwargs['discount_factor']
        self.epsilon = kwargs['epsilon']
        self.n_tilings = kwargs['n_tilings']
        self.grid_dims = kwargs['grid_dims']

        # initialize Q function
        self.Q = defaultdict(np.random.rand)

    def __validate_env__(self, env):
        is_multi_obs, is_multi_act, is_disc_obs, is_disc_act = \
            check_discrete(env, 'Temporal Difference', obs=False)

        # action space is multidimensional
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

        # initialize obs -> scalar dictionaries
        self.obs2num = {}
        self.num2obs = {}

        self.obs_encoder = lambda x: x
        if not is_disc_obs:
            self.obs_encoder, _ = \
                tile_state_space(env, n_tilings, grid_size=grid_dims)


    def on_policy_update_Q(self, s, a, r, s_, a_):
        """
        Update Q function using the expected SARSA on-policy TD(0) update.
            Q(s, a) <- Q(s, a) + lr * [r + gamma * E[Q(s', a') | s'] - Q(s, a)]
        """
        # compute the expected value of Q(s', a') given that we are in state s'
        E_Q = np.sum(
            [self.epsilon_soft_policy(s_, aa) * self.Q[(s_, aa)]
             for aa in xrange(self.n_actions)]
        )

        self.Q[(s, a)] += self.learning_rate * \
            (r + self.gamma * E_Q - self.Q[(s, a)])

    def off_policy_update_Q(self, s, a, r, s_):
        """
        Update Q using the off-policy TD(0) Q-learning update
            Q(s, a) <- Q(s, a) + lr * [r + gamma * max_a { Q(s', a) } - Q(s, a)]
        """
        Qs_ = [self.Q[(s_, aa)] for aa in xrange(self.n_actions)]

        self.Q[(s, a)] += self.learning_rate * \
            (r + self.gamma * np.max(Qs_) - self.Q[(s, a)])

    def epsilon_soft_policy(self, s, action=None):
        """
        pi(a|s) = 1 - epsilon + (epsilon / |A(s)|) IFF a == a*
        pi(a|s) = epsilon / |A(s)|                 IFF a != a*
        """
        #  a_star = self.Q[s, :].argmax()
        a_star = np.argmax([self.Q[(s, aa)] for aa in xrange(self.n_actions)])
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
        obs = self.obs_encoder(obs)
        if obs in self.obs2num.keys():
            num = self.obs2num[obs]
        else:
            num = 0
            if len(self.obs2num) > 0:
                num = np.max(self.obs2num.values()) + 1
            self.obs2num[obs] = num
            self.num2obs[num] = obs
        return num

    def run_episode(self, env, render=False):
        obs = env.reset()
        s = self.__encode_obs__(obs)

        # generate an action using an epsilon-soft policy
        action = self.epsilon_soft_policy(s)
        a = self.action2num[action]

        # run one episode of the RL problem
        reward_history = []
        for _ in xrange(self.episode_len):
            if render:
                env.render()

            # take action
            obs_, reward, done, info = env.step(action)
            s_ = self.__encode_obs__(obs_)

            # record rewards
            reward_history.append(reward)

            if self.off_policy:
                # Q-learning off-policy update
                self.off_policy_update_Q(s, a, reward, s_)

                # update observations and actions
                obs, s = obs_, s_

                # generate an action using an epsilon-soft policy
                action = self.epsilon_soft_policy(s)
                a = self.action2num[action]

            else:
                # generate a new action using an epsilon-soft policy
                action_ = self.epsilon_soft_policy(s_)
                a_ = self.action2num[action_]

                # expected SARSA on-policy update
                self.on_policy_update_Q(s, a, reward, s_, a_)

                # update observations and actions
                obs, s, action, a = obs_, s_, action_, a_

            if done:
                break

        return np.sum(reward_history)

if __name__ == "__main__":
    # initialize RL environment
    env = gym.make('CartPole-v0')

    # initialize run parameters
    n_epochs = 50000        # number of episodes to train the q network on
    epsilon = 0.10          # for epsilon-soft policy during training
    max_episode_len = 200   # max number of timesteps per episode/epoch
    discount_factor = 0.95  # temporal discount factor
    learning_rate = 0.05    # learning rate parameter
    render = False          # render runs during training
    n_tilings = 8           # number of tilings to use if state space is continuous
    grid_dims = [8, 8]      # number of squares in the tiling grid if state
                            # space is continuous
    off_policy = False      # on_policy = expected SARSA update
                            # off_policy = Q-learning update

    mc_params = \
        {'epsilon': epsilon,
         'grid_dims': grid_dims,
         'n_tilings': n_tilings,
         'off_policy': off_policy,
         'learning_rate': learning_rate,
         'max_episode_len': max_episode_len,
         'discount_factor': discount_factor}

    # initialize network and experience replay objects
    td_learner = TDLearner(env, **mc_params)

    # train monte carlo learner
    t0 = time()
    episode_rewards = []
    for idx in xrange(n_epochs):
        total_reward = td_learner.run_episode(env)

        print('Total reward on epoch {}/{}:\t{}'
              .format(idx + 1, n_epochs, total_reward))

        episode_rewards.append(total_reward)
    print('\nTraining took {} mins'.format((time() - t0) / 60.))

    env_name = env.spec.id
    policy = 'off' if off_policy else 'on'
    param_str = '{}Policy_lr{:.3E}_ntilings{}_griddim{}x{}.png'\
                    .format(policy, learning_rate, n_tilings, *grid_dims)
    save_path = 'plots/td_learner_{}.png'.format(param_str)
    plot_rewards(episode_rewards, save_path, env_name)

    print('Executing greedy policy\n')
    td_learner.epsilon = 0
    for idx in xrange(10):
        total_reward = td_learner.run_episode(env, render=True)

        print('Total reward on greedy epoch {}/{}:\t{}\n'
              .format(idx + 1, 10, total_reward))
