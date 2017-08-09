from collections import defaultdict
from time import time
import itertools

import gym
import numpy as np

from utils import check_discrete


class DynaLearner(object):
    """
    A tabular Dyna-Q / Dyna-Q+ learning agent with full backups selected via
    prioritized sweeping
    """

    def __init__(self, env, **kwargs):
        self.__validate_env__(env)

        self.n_simulated_actions = kwargs['n_simulated_actions']
        self.episode_len = kwargs['max_episode_len']
        self.gamma = kwargs['discount_factor']
        self.epsilon = kwargs['epsilon']
        self.learning_rate = kwargs['learning_rate']
        self.q_plus = kwargs['dyna_q_plus']
        self.explore_weight = kwargs['explore_weight']

        # initialize Q function
        self.Q = np.random.rand(self.n_states, self.n_actions)

        # initialize model
        self.model = defaultdict(lambda: defaultdict(lambda: 0))

        # initialize list of visited states
        self.visited = set([])

        # intialize the prioritized sweeping queue
        self.sweep_queue = {}

        # initialize time since last visit table
        if self.q_plus:
            self.steps_since_last_visit = \
                np.zeros(self.n_states, self.n_actions)

    def __validate_env__(self, env):
        is_multi_obs, is_multi_act = check_discrete(env, 'Dyna')

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

    def epsilon_greedy_policy(self, obs):
        """
        Select a random action with probability epsilon, otherwise greedily
        select action based on current Q estimate
        """
        val = np.random.rand()
        if val <= epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action = self.Q[obs, :].argmax()
        return self.num2action[action]

    def full_backup_update(self, s, a):
        """
        Update Q using a full-backup version of the TD(0) Q-learning update
            Q(s, a) <- Q(s, a) + lr *
                sum_{r, s'} [
                    p(r, s' | s, a) * (r + gamma * max_a { Q(s', a) } - Q(s, a))
                ]
        """
        outcomes, outcome_probs = self.next_state_probs(s, a)

        update = 0.0
        for (r, s_), p_rs_ in zip(outcomes, outcome_probs):
            # Dyna-Q+ version to encourage visiting long-untried actions by
            # adding a "bonus" reward proportional to the square root of
            # the time since last visit
            if self.q_plus:
                r += self.explore_weight * np.sqrt(
                    self.steps_since_last_visit[s, a])

            update += \
                p_rs_ * (r + self.gamma * np.max(self.Q[s_, :]) - self.Q[s, a])

        # update Q value for (s, a) pair
        self.Q[s, a] += self.learning_rate * update

    def update_queue(self, s, a):
        outcomes, outcome_probs = self.next_state_probs(s, a)

        priority = 0.0
        for (r, s_), p_rs_ in zip(outcomes, outcome_probs):
            priority += \
                p_rs_ * (r + self.gamma * np.max(self.Q[s_, :]) - self.Q[s, a])
        priority = np.abs(priority)

        # TODO: what's a good threshold here?
        if priority >= 0.001:
            if (s, a) in self.sweep_queue and \
                    priority > self.sweep_queue[(s, a)]:
                self.sweep_queue[(s, a)] = priority
            else:
                self.sweep_queue[(s, a)] = priority

    def run_episode(self, env, render=False):
        obs = env.reset()
        s = self.obs2num[obs]

        # run one episode of the RL problem
        reward_history = []
        for _ in xrange(self.episode_len):
            if render:
                env.render()

            # generate an action using epsilon-greedy policy
            action = self.epsilon_greedy_policy(s)
            a = self.action2num[action]

            # take action
            obs_next, reward, done, info = env.step(action)
            s_ = self.obs2num[obs_next]

            # update model
            self.model[(s, a)][(reward, s_)] += 1

            # record rewards
            reward_history.append(reward)

            if self.q_plus:
                self.steps_since_last_visit += 1.
                self.steps_since_last_visit[s, a] = 0.

            # update Q function using full-backup version of TD(0) Q-learning
            # update
            self.update_queue(s, a)

            if not done:
                self.visited.add(s)

            # begin simulated steps with prioritized sweeping
            self.simulate_actions()

            if done:
                break

            # update to the real next state
            obs, s = obs_next, s_

        return np.sum(reward_history)

    def simulate_actions(self):
        for _ in xrange(self.n_simulated_actions):
            if len(self.sweep_queue) == 0:
                break

            # select (s, a) pair with the largest update (priority)
            (s_sim, a_sim), _ = sorted(self.sweep_queue.items(),
                                       key=lambda x: x[1],
                                       reverse=True)[0]

            # remove entry from queue
            del self.sweep_queue[(s_sim, a_sim)]

            # update Q function using a full backup verion of the TD(0)
            # Q-learning update using simulated data
            self.full_backup_update(s_sim, a_sim)

            # get all (_s, _a) pairs predicted to lead to s:
            pairs = [i for i in self.model.keys() if s in
                     [j for r, j in self.model[i].keys()]]

            for (_s, _a) in pairs:
                self.update_queue(_s, _a)

    def next_state_probs(self, s, a):
        items = self.model[(s, a)].items()
        total_count = np.sum([c for (_, c) in items])
        outcome_probs = [float(c) / total_count for (_, c) in items]
        outcomes = [p for (p, _) in items]
        return outcomes, outcome_probs


if __name__ == "__main__":
    # initialize RL environment
    env = gym.make('Copy-v0')

    # initialize run parameters
    render = False            # render runs during training
    epsilon = 0.10            # for epsilon-greedy policy during training
    n_epochs = 10000          # number of episodes to train the q network on
    learning_rate = 0.1       # learning rate parameter (alpha)
    max_episode_len = 200     # max number of timesteps per episode/epoch
    discount_factor = 0.9     # temporal discount factor (gamma)
    n_simulated_actions = 50  # number of simulated actions to perform for each
                              # "real" action
    dyna_q_plus = False       # whether to add incentives for visiting states
                              # which we haven't seen in a while
    explore_weight = 0.05     # amount to incentivize exploring long-untried
                              # states if `dyna_q_plus` is true (kappa)

    mc_params = \
        {'epsilon': epsilon,
         'n_simulated_actions': n_simulated_actions,
         'dyna_q_plus': dyna_q_plus,
         'explore_weight': explore_weight,
         'learning_rate': learning_rate,
         'max_episode_len': max_episode_len,
         'discount_factor': discount_factor}

    # initialize network and experience replay objects
    dyna_learner = DynaLearner(env, **mc_params)

    # train Dyna learner
    t0 = time()
    epoch_reward = []
    for idx in xrange(n_epochs):
        total_reward = dyna_learner.run_episode(env)

        print('Total reward on epoch {}/{}:\t{}'
              .format(idx + 1, n_epochs, total_reward))

        epoch_reward.append(total_reward)

    print('\nTraining took {} mins'.format((time() - t0) / 60.))
    print('Executing greedy policy\n')

    dyna_learner.epsilon = 0
    for idx in xrange(10):
        total_reward = dyna_learner.run_episode(env, render=True)

        print('Total reward on greedy epoch {}/{}:\t{}\n'
              .format(idx + 1, 10, total_reward))
