import gym
import numpy as np
import cPickle as pickle

def softmax(obs, W, b):
    out = obs.dot(W) + b
    probs = np.exp(out) / np.sum(np.exp(out))
    return probs

def gen_action(obs, W, b):
    """
    Assumes that the pmf over actions in state x_t is given by

        softmax( x_t * W + b )

    where W is a learned weight matrix, x_t is the observation at timestep t, and
    b is a learned bias vector.
    """
    probs = softmax(obs, W, b)
    action = np.random.multinomial(1, probs).argmax()
    return action

def run_episode(env, W, b, horizon=400, render=False):
    total_reward = 0.0
    obs = env.reset()

    for _ in xrange(horizon):
        if render:
            env.render()

        action = gen_action(obs, W, b)
        obs, reward, done, info = env.step(action)
        total_reward += reward

        if done:
           break
    return total_reward

def cross_entropy_solver(theta_mean, theta_var, **kwargs):
    """
    Implementation of cross-entropy method for black-box optimization of the
    total accumuluated reward during an episode
    """
    n_theta_samples = kwargs['n_theta_samples']
    weights_len = kwargs['weights_len']
    n_elites = kwargs['n_elites']
    bias_len = kwargs['bias_len']
    horizon = kwargs['horizon']
    render = kwargs['render']
    env = kwargs['env']

    n_actions = env.action_space.n
    n_obs_dims = env.observation_space.shape[0]

    # sample n_theta_samples (x_vals) from a mv gaussian with mean theta_mean and
    # diagonal covariance
    theta_samples = np.random.multivariate_normal(theta_mean,
            np.diag(theta_var), n_theta_samples)

    # evaluate each of the theta_samples (x_vals) on an episode of the RL problem
    y_vals = []
    for theta in theta_samples:
        W = theta[:weights_len].reshape(n_obs_dims, n_actions)
        b = theta[weights_len:]
        total_reward = run_episode(env, W, b, horizon=horizon, render=render)
        y_vals.append(total_reward)

    avg_reward = np.mean(y_vals)
    min_reward = np.min(y_vals)

    # sort the y_vals (total_rewards) from greatest to least
    sorted_y_val_idxs = np.argsort(y_vals)[::-1]
    elite_idxs = sorted_y_val_idxs[:n_elites]

    theta_mean = np.mean(theta_samples[elite_idxs], axis=0)
    theta_var = np.var(theta_samples[elite_idxs], axis=0)
    return theta_mean, theta_var, avg_reward, min_reward

if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    n_theta_samples = 500
    n_elites = int(n_theta_samples * 0.2)
    n_episodes = 50
    horizon = 200
    render = False

    num_actions = env.action_space.n
    num_obs_dims = env.observation_space.shape[0]

    bias_len = num_actions
    weights_len = num_actions * num_obs_dims
    theta_dim = weights_len + bias_len

    # init mean and variance for mv gaussian with dimensions theta_dim
    theta_mean = np.random.rand(theta_dim)
    theta_var = np.ones(theta_dim)

    ce_params = {'n_theta_samples': n_theta_samples,
                 'bias_len': bias_len,
                 'n_elites': n_elites,
                 'weights_len': weights_len,
                 'render': render,
                 'env': env,
                 'horizon': horizon}

    for ep_id in xrange(n_episodes):
        theta_mean, theta_var, avg_reward, min_reward = \
                cross_entropy_solver(theta_mean, theta_var, **ce_params)
        print('Episode {}... \tAvg reward = {}, Worst = {}'\
                .format(ep_id + 1, avg_reward, min_reward))

    ce_params['render'] = True
    for ep_id in xrange(20):
        print('Starting greedy session with frozen q function')
        _, _, avg_reward, min_reward = \
                cross_entropy_solver(theta_mean, theta_var, **ce_params)

        print('Episode {}... \tAvg reward = {}, Worst = {}'\
                .format(ep_id + 1, avg_reward, min_reward))
