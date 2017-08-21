from time import time
import itertools

import gym
import numpy as np

from utils import softmax, check_discrete

class CrossEntropyLearner(object):
    """
    A cross-entropy method learning agent
    """

    def __init__(self, env, **kwargs):
        # initialize theta dimensions and environment properties
        self.__validate_env__(env)

        self.n_theta_samples = kwargs['n_theta_samples']
        self.episode_len = kwargs['max_episode_len']
        self.top_n = kwargs['top_n']

        # init mean and variance for mv gaussian with dimensions theta_dim
        self.theta_mean = np.random.rand(self.theta_dim)
        self.theta_var = np.ones(self.theta_dim)

    def __validate_env__(self, env):
        _, is_multi_act, _, _ = \
            check_discrete(env, 'Cross Entropy', action=True, obs=False)

        # action space is multidimensional
        if is_multi_act:
            n_actions = [space.n for space in env.action_space.spaces]
            one_dim_action = list(
                itertools.product(*[range(i) for i in n_actions]))
        else:
            n_actions = env.action_space.n
            one_dim_action = range(n_actions)

        try:
            n_obs_dim = env.observation_space.shape[0]
        except AttributeError:
            n_obs_dim = 1

        # create action -> scalar dictionaries
        self.action2num = {act: i for i, act in enumerate(one_dim_action)}
        self.num2action = {i: act for act, i in self.action2num.items()}

        self.n_actions = np.prod(n_actions)
        self.obs_dim = n_obs_dim

        self.bias_len = np.prod(n_actions)
        self.weights_len = np.prod(n_actions) * np.prod(n_obs_dim)
        self.theta_dim = self.weights_len + self.bias_len

    def softmax_policy(self, obs, W, b):
        """
        Assumes that the pmf over actions in state x_t is given by

            softmax( x_t * W + b )

        where W is a learned weight matrix, x_t is the observation at timestep
        t, and b is a learned bias vector.
        """
        if self.obs_dim == 1:
            obs = np.array([obs])

        probs = softmax(obs, W, b)
        action = np.random.multinomial(1, probs).argmax()
        return self.num2action[action]

    def evaluate_sample(self, env, W, b, render=False):
        obs = env.reset()

        total_reward = 0.0
        for _ in xrange(self.episode_len):
            if render:
                env.render()

            action = self.softmax_policy(obs, W, b)
            obs, reward, done, _ = env.step(action)
            total_reward += reward

            if done:
                break

        return total_reward

    def run_episode(self, env, render=False, freeze_theta=False):
        # sample n_theta_samples (x_vals) from a mv gaussian with mean
        # theta_mean and diagonal covariance
        theta_samples = np.random.multivariate_normal(
            self.theta_mean, np.diag(self.theta_var), self.n_theta_samples)

        # evaluate each of the theta_samples (x_vals) on an episode of the RL
        # problem
        y_vals = []
        for theta in theta_samples:
            W = theta[:self.weights_len].reshape(self.obs_dim, self.n_actions)
            b = theta[self.weights_len:]

            total_reward = self.evaluate_sample(env, W, b, render=render)
            y_vals.append(total_reward)

        # sort the y_vals (total_rewards) from greatest to least
        sorted_y_val_idxs = np.argsort(y_vals)[::-1]
        top_idxs = sorted_y_val_idxs[:top_n]

        # update theta_mean and theta_var with the best theta value
        if not freeze_theta:
            self.theta_mean = np.mean(theta_samples[top_idxs], axis=0)
            self.theta_var = np.var(theta_samples[top_idxs], axis=0)

        # get the average reward for the top performing theta samples
        avg_reward = np.mean(np.array(y_vals)[top_idxs])
        return avg_reward


if __name__ == "__main__":
    # initialize rl environment
    env = gym.make('LunarLander-v2')

    # initialize run parameters
    render = False          # render runs during training?
    n_episodes = 100        # number of episodes to train the XE learner on
    max_episode_len = 200   # max number of timesteps per episode
    n_theta_samples = 500   # number of samples to generate per run

    # average over the `top_n` best performing theta samples
    top_n = int(n_theta_samples * 0.2)

    xe_params = \
        {'max_episode_len':  max_episode_len,
         'n_theta_samples': n_theta_samples,
         'top_n': top_n}

    # initialize network and experience replay objects
    xe_learner = CrossEntropyLearner(env, **xe_params)

    # train cross entropy learner
    t0 = time()
    epoch_reward = []
    for idx in xrange(n_episodes):
        avg_reward = xe_learner.run_episode(env)

        print('Average reward on epoch {}/{}:\t{}'
              .format(idx + 1, n_episodes, avg_reward))

        epoch_reward.append(avg_reward)

    print('\nTraining took {} mins'.format((time() - t0) / 60.))
    print('Executing greedy policy\n')

    for idx in xrange(10):
        avg_reward = xe_learner.run_episode(env, render=True,
                                            freeze_theta=True)

        print('Average reward on greedy epoch {}/{}:\t{}\n'
              .format(idx + 1, 10, avg_reward))
