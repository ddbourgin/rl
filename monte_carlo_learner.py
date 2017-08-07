from time import time
import itertools

import gym
import numpy as np

from utils import check_discrete


class MonteCarloLearner(object):
    """
    A tabular Monte Carlo learning agent.
    """

    def __init__(self, **kwargs):
        self.n_actions = np.prod(kwargs['n_actions'])
        self.episode_len = kwargs['max_episode_len']
        self.off_policy = kwargs['off_policy']
        self.gamma = kwargs['discount_factor']
        self.n_states = np.prod(kwargs['n_obs'])
        self.epsilon = kwargs['epsilon']

        # create action -> scalar dictionaries
        action_space = kwargs['n_actions']
        if isinstance(action_space, list):
            one_dim_action = list(
                itertools.product(*[range(i) for i in action_space]))
        else:
            one_dim_action = range(action_space)

        self.action2num = {act: i for i, act in enumerate(one_dim_action)}
        self.num2action = {i: act for act, i in self.action2num.items()}

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
            self.target_policy = \
                lambda obs, action=None: \
                self.num2action[self.Q[obs, :].argmax()] if action is None \
                else np.abs(self.Q[obs, action]) / np.sum(np.abs(self.Q[obs, :]))

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

    def epsilon_soft_policy(self, observation, action=None):
        """
        pi(a|s) = 1 - epsilon + (epsilon / |A(s)|) IFF a == a*
        pi(a|s) = epsilon / |A(s)|                 IFF a != a*
        """
        a_star = self.Q[observation, :].argmax()
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

        # run one episode of the RL problem
        episode_history, reward_history = [], []
        for _ in xrange(self.episode_len):
            if render:
                env.render()

            # generate an action using epsilon-soft policy
            action = self.epsilon_soft_policy(obs)
            action_number = self.action2num[action]

            # store (state, action) tuple
            episode_history.append((obs, action_number))

            # take action
            obs_next, reward, done, info = env.step(action)

            # record rewards
            reward_history.append(reward)

            if done:
                break
            obs = obs_next

        if self.off_policy:
            self.off_policy_update_Q(episode_history, reward_history)
        else:
            # update Q values at the end of the episode
            self.on_policy_update_Q(episode_history, reward_history)

        return np.sum(reward_history)


if __name__ == "__main__":
    # initialize RL environment
    env = gym.make('Copy-v0')
    is_multi_obs, is_multi_act = check_discrete(env, 'Monte Carlo')

    # action space is multidimensional
    if is_multi_act:
        n_actions = [space.n for space in env.action_space.spaces]
    else:
        n_actions = env.action_space.n

    # observation space is multidimensional
    if is_multi_obs:
        n_obs = [space.n for space in env.observation_space.spaces]
    else:
        n_obs = env.observation_space.n

    # initialize run parameters
    n_epochs = 1000000      # number of episodes to train the q network on
    epsilon = 0.10          # for epsilon-soft policy during training
    max_episode_len = 200   # max number of timesteps per episode/epoch
    discount_factor = 0.9   # temporal discount factor if `off_policy` is true
    off_policy = False      # whether to use on or off-policy MC updates
    render = False          # render runs during training

    mc_params = \
        {'epsilon': epsilon,
         'n_actions': n_actions,
         'n_obs': n_obs,
         'off_policy': off_policy,
         'max_episode_len': max_episode_len,
         'discount_factor': discount_factor}

    # initialize network and experience replay objects
    mc_learner = MonteCarloLearner(**mc_params)

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
