from time import time
import itertools
import gym
import numpy as np


class MonteCarloLearner(object):
    """
    Implements on-policy, first visit Monte Carlo learning for an epsilon-soft
    policy.
    """

    def __init__(self, **kwargs):
        self.n_actions = np.prod(kwargs['n_actions'])
        self.n_states = kwargs['n_obs_dims']
        self.epsilon = kwargs['epsilon']

        # create action -> scalar dictionaries
        action_space = kwargs['n_actions']
        if isinstance(action_space, list):
            one_dim_action = list(
                itertools.product([range(i) for i in action_space]))
        else:
            one_dim_action = range(action_space)

        self.action2num = {act : i for i, act in enumerate(one_dim_action)}
        self.num2action = {i : act for act, i in self.action2num.items()}

        # initialize Q function
        self.Q = np.random.rand(self.n_states, self.n_actions)

        # initialize returns object for each state-action pair
        self.returns = {
            (s, a): [] for s in range(self.n_states)
            for a in range(self.n_actions)
        }

    def update_Q(self, episode_history, reward_history):
        """
        Update Q-table at the end of each episode.
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

    def epsilon_soft_policy(self, observation):
        """
        pi(a|s) = 1 - epsilon + (epsilon / |A(s)|) IFF a == a*
        pi(a|s) = epsilon / |A(s)|                 IFF a != a*
        """
        a_star = self.Q[observation, :].argmax()
        p_a_star = 1. - self.epsilon + (self.epsilon / self.n_actions)
        p_a = self.epsilon / self.n_actions

        action_probs = np.ones(self.n_actions) * p_a
        action_probs[a_star] = p_a_star

        # sample action
        action = np.random.multinomial(1, action_probs)
        return self.num2action[action.argmax()]


def run_episode(env, mc_learner, render=False, episode_len=200):
    obs = env.reset()

    # run one episode of the RL problem
    episode_history, reward_history = [], []
    for _ in xrange(episode_len):
        if render:
            env.render()

        # generate an action using epsilon-soft policy
        action = mc_learner.epsilon_soft_policy(obs)
        action_number = mc_learner.action2num[action]

        # store (state, action) tuple
        episode_history.append((obs, action_number))

        # take action
        obs_next, reward, done, info = env.step(action)

        # record rewards
        reward_history.append(reward)

        if done:
            break
        obs = obs_next

    # update Q values at the end of the episode
    mc_learner.update_Q(episode_history, reward_history)
    return mc_learner, np.sum(reward_history)

if __name__ == "__main__":
    # initialize RL environment
    env = gym.make('Copy-v0')

    # action space is multidimensional
    n_actions = [env.action_space.spaces[i].n for i in range(3)]

    n_obs_dims = env.observation_space.n

    # initialize run parameters
    n_epochs = 100000    # number of episodes to train the q network on
    epsilon = 0.10       # for epsilon-soft policy during training
    max_episode_len = 200      # max number of timesteps per episode/epoch
    render = False  # render runs during training

    mc_params = \
        {'epsilon': epsilon,
         'n_actions': n_actions,
         'n_obs_dims': n_obs_dims}

    # initialize network and experience replay objects
    mc_learner = MonteCarloLearner(**mc_params)

    # train monte carlo learner
    t0 = time()
    epoch_reward = []
    for idx in xrange(n_epochs):
        mc_learner, total_reward = \
            run_episode(env, mc_learner, episode_len=max_episode_len)

        print('Total reward on epoch {}/{}:\t{}'
              .format(idx + 1, n_epochs, total_reward))

        epoch_reward.append(total_reward)

    print('\nTraining took {} mins'.format((time() - t0) / 60.))
    print('Executing greedy policy\n')

    mc_learner.epsilon = 0
    for idx in xrange(10):
        mc_learner, total_reward = \
            run_episode(env, mc_learner, episode_len=max_episode_len, render=True)

        print('Total reward on greedy epoch {}/{}:\t{}\n'
              .format(idx + 1, 10, total_reward))
