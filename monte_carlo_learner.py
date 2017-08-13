from time import time
import itertools

import gym
import numpy as np

from utils import check_discrete


class MonteCarloLearner(object):
    """
    A tabular Monte Carlo learning agent.
    """

    def __init__(self, env, **kwargs):
        self.__validate_env__(env)

        self.episode_len = max_episode_len
        self.off_policy = off_policy
        self.gamma = discount_factor
        self.epsilon = epsilon

        # initialize Q function
        self.Q = np.random.rand(self.n_states, self.n_actions)

        # initialize returns object for each state-action pair
        self.returns = {
            (s, a): [] for s in range(self.n_states)
            for a in range(self.n_actions)
        }

        if self.off_policy:
            self.C = np.zeros((self.n_states, self.n_actions))

            # behavior policy is stochastic, epsilon-soft policy
            self.behavior_policy = self.epsilon_soft_policy

            # target policy is deterministic, greedy policy
            self.target_policy = self.greedy_policy

    def __validate_env__(self, env):
        is_multi_obs, is_multi_act = check_discrete(env, 'Monte Carlo')

        # action space is multidimensional
        if is_multi_act:
            n_actions = [space.n for space in env.action_space.spaces]
            one_dim_action = list(
                itertools.product(*[range(i) for i in n_actions]))
        else:
            n_actions = env.action_space.n
            one_dim_action = range(n_actions)

        # observation space is multidimensional
        if is_multi_obs:
            n_obs = [space.n for space in env.observation_space.spaces]
            one_dim_obs = list(
                itertools.product(*[range(i) for i in n_obs]))
        else:
            n_obs = env.observation_space.n
            one_dim_obs = range(n_obs)

        # create action -> scalar dictionaries
        self.action2num = {act: i for i, act in enumerate(one_dim_action)}
        self.num2action = {i: act for act, i in self.action2num.items()}

        # create obs -> scalar dictionaries
        self.obs2num = {act: i for i, act in enumerate(one_dim_obs)}
        self.num2obs = {i: act for act, i in self.obs2num.items()}

        self.n_actions = np.prod(n_actions)
        self.n_states = np.prod(n_obs)

    def on_policy_update_Q(self, episode_history, reward_history):
        """
        Update Q function using an on-policy first-visit Monte Carlo update.

            Q'(s, a) <- avg(return following first visit to (s, a) across all
                            episodes)
        """
        tuples = set(episode_history)
        locs = [episode_history.index(pair) for pair in tuples]
        cumulative_returns = [np.sum(reward_history[i:]) for i in locs]

        # update Q value with the average of the first-visit return across
        # episodes
        for pair, cr in zip(tuples, cumulative_returns):
            self.returns[pair].append(cr)
            self.Q[pair[0], pair[1]] = np.mean(self.returns[pair])

    def off_policy_update_Q(self, episode_history, reward_history):
        """
        Update Q using incremental weighted importance sampling

            G_t = total discounted return after time t until episode end
            W   = importance sampling weight
            C_n = sum of importance weights for the first n returns
        """
        G, W = 0.0, 1.0
        for t in reversed(range(len(reward_history) - 1)):
            sa_t = episode_history[t]
            G = self.gamma * G + reward_history[t + 1]
            self.C[sa_t[0], sa_t[1]] += W

            # update Q(s, a) using the incremental importance sampling update
            self.Q[sa_t[0], sa_t[1]] += \
                (W / self.C[sa_t[0], sa_t[1]]) * (G - self.Q[sa_t[0], sa_t[1]])

            # multiply the importance sampling ratio by the current weight
            W *= (self.target_policy(sa_t[0], sa_t[1]) /
                  self.behavior_policy(sa_t[0], sa_t[1]))

            if W == 0.:
                break

    def greedy_policy(self, obs, action=None):
        if action is None:
            return self.num2action[self.Q[obs, :].argmax()]
        else:
            return np.abs(self.Q[obs, action]) / np.sum(np.abs(self.Q[obs, :]))

    def epsilon_soft_policy(self, obs, action=None):
        """
        pi(a|s) = 1 - epsilon + (epsilon / |A(s)|) IFF a == a*
        pi(a|s) = epsilon / |A(s)|                 IFF a != a*
        """
        a_star = self.Q[obs, :].argmax()
        p_a_star = 1. - self.epsilon + (self.epsilon / self.n_actions)
        p_a = self.epsilon / self.n_actions

        action_probs = np.ones(self.n_actions) * p_a
        action_probs[a_star] = p_a_star

        if action is not None:
            return action_probs[action]

        # sample action
        action = np.random.multinomial(1, action_probs)
        return self.num2action[action.argmax()]

    def run_episode(self, env, render=False):
        obs = env.reset()
        s = self.obs2num[obs]

        # run one episode of the RL problem
        episode_history, reward_history = [], []
        for _ in xrange(self.episode_len):
            if render:
                env.render()

            # generate an action using epsilon-soft policy
            action = self.epsilon_soft_policy(s)
            a = self.action2num[action]

            # store (state, action) tuple
            episode_history.append((s, a))

            # take action
            obs_next, reward, done, info = env.step(action)
            s_ = self.obs2num[obs_next]

            # record rewards
            reward_history.append(reward)

            if done:
                break

            obs, s = obs_next, s_

        if self.off_policy:
            self.off_policy_update_Q(episode_history, reward_history)
        else:
            # update Q values at the end of the episode
            self.on_policy_update_Q(episode_history, reward_history)

        return np.sum(reward_history)


if __name__ == "__main__":
    # initialize RL environment
    env = gym.make('Copy-v0')

    # initialize run parameters
    n_epochs = 1000000      # number of episodes to train the q network on
    epsilon = 0.10          # for epsilon-soft policy during training
    max_episode_len = 200   # max number of timesteps per episode/epoch
    discount_factor = 0.9   # temporal discount factor if `off_policy` is true
    off_policy = False      # whether to use on or off-policy MC updates
    render = False          # render runs during training

    mc_params = \
        {'epsilon': epsilon,
         'off_policy': off_policy,
         'max_episode_len': max_episode_len,
         'discount_factor': discount_factor}

    # initialize network and experience replay objects
    mc_learner = MonteCarloLearner(env, **mc_params)

    # train monte carlo learner
    t0 = time()
    epoch_reward = []
    for idx in xrange(n_epochs):
        total_reward = mc_learner.run_episode(env)

        print('Total reward on epoch {}/{}:\t{}'
              .format(idx + 1, n_epochs, total_reward))

        epoch_reward.append(total_reward)

    print('\nTraining took {} mins'.format((time() - t0) / 60.))
    print('Executing greedy policy\n')

    mc_learner.epsilon = 0
    for idx in xrange(10):
        total_reward = mc_learner.run_episode(env, render=True)

        print('Total reward on greedy epoch {}/{}:\t{}\n'
              .format(idx + 1, 10, total_reward))
