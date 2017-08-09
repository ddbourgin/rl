from time import time
import itertools

import gym
import numpy as np

from utils import check_discrete

class TDLearner(object):
    """
    A temporal difference learning agent
    """

    def __init__(self, env, **kwargs):
        self.__validate_env__(env)

        self.learning_rate = kwargs['learning_rate']
        self.episode_len = kwargs['max_episode_len']
        self.off_policy = kwargs['off_policy']
        self.gamma = kwargs['discount_factor']
        self.epsilon = kwargs['epsilon']

        # initialize Q function
        self.Q = np.random.rand(self.n_states, self.n_actions)

    def __validate_env__(self, env):
        is_multi_obs, is_multi_act = check_discrete(env, 'Temporal Difference')

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

    def on_policy_update_Q(self, s, a, r, s_, a_):
        """
        Update Q function using the expected SARSA on-policy TD(0) update.
            Q(s, a) <- Q(s, a) + lr * [r + gamma * E[Q(s', a') | s'] - Q(s, a)]
        """
        # compute the expected value of Q(s', a') given that we are in state s'
        E_Q = np.sum(
            [self.epsilon_soft_policy(s_, aa) * self.Q[s_, aa]
             for aa in xrange(self.n_actions)]
        )

        self.Q[s, a] += self.learning_rate * \
            (r + self.gamma * E_Q - self.Q[s, a])

    def off_policy_update_Q(self, s, a, r, s_):
        """
        Update Q using the off-policy TD(0) Q-learning update
            Q(s, a) <- Q(s, a) + lr * [r + gamma * max_a { Q(s', a) } - Q(s, a)]
        """
        self.Q[s, a] += self.learning_rate * \
            (r + self.gamma * np.max(self.Q[s_, :]) - self.Q[s, a])

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
        s = self.obs2num[obs]

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
            s_ = self.obs2num[obs_]

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
                action_ = self.epsilon_soft_policy(obs_)
                a_ = self.action2num[action]

                # expected SARSA on-policy update
                self.on_policy_update_Q(obs, a, reward, obs_, a_)

                # update observations and actions
                obs, s, action, a = obs_, s_, action_, a_

            if done:
                break

        return np.sum(reward_history)

if __name__ == "__main__":
    # initialize RL environment
    env = gym.make('Copy-v0')

    # initialize run parameters
    n_epochs = 100000       # number of episodes to train the q network on
    epsilon = 0.10          # for epsilon-soft policy during training
    max_episode_len = 200   # max number of timesteps per episode/epoch
    discount_factor = 0.95  # temporal discount factor
    learning_rate = 0.1     # learning rate parameter
    off_policy = True       # on_policy = expected SARSA update
                            # off_policy = Q-learning update
    render = False          # render runs during training

    mc_params = \
        {'epsilon': epsilon,
         'off_policy': off_policy,
         'learning_rate': learning_rate,
         'max_episode_len': max_episode_len,
         'discount_factor': discount_factor}

    # initialize network and experience replay objects
    td_learner = TDLearner(env, **mc_params)

    # train monte carlo learner
    t0 = time()
    epoch_reward = []
    for idx in xrange(n_epochs):
        total_reward = td_learner.run_episode(env)

        print('Total reward on epoch {}/{}:\t{}'
              .format(idx + 1, n_epochs, total_reward))

        epoch_reward.append(total_reward)

    print('\nTraining took {} mins'.format((time() - t0) / 60.))
    print('Executing greedy policy\n')

    td_learner.epsilon = 0
    for idx in xrange(10):
        total_reward = td_learner.run_episode(env, render=True)

        print('Total reward on greedy epoch {}/{}:\t{}\n'
              .format(idx + 1, 10, total_reward))
