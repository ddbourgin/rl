import gym
import numpy as np

def softmax(obs, W, b):
    out = obs.dot(W) + b
    probs = np.exp(out) / np.sum(np.exp(out))
    return probs


def check_discrete(env, name, action=True, obs=True):
    discrete = gym.spaces.discrete.Discrete
    tuple_space = gym.spaces.tuple_space.Tuple

    is_tuple_action = isinstance(env.action_space, tuple_space)
    is_tuple_obs = isinstance(env.observation_space, tuple_space)

    # action and observation spaces must by discrete for tabular learning
    if obs and is_tuple_obs:
        if not all([isinstance(i, discrete) for i in env.observation_space.spaces]):
            raise TypeError('{} Learner only works with discrete '
                            'observation spaces'.format(name))
    # action and observation spaces must by discrete for tabular learning
    elif obs and not isinstance(env.observation_space, discrete):
        raise TypeError('{} Learner only works with discrete '
                        'observation spaces'.format(name))

    if action and is_tuple_action:
        if not all([isinstance(i, discrete) for i in env.action_space.spaces]):
            raise TypeError('{} Learner only works with discrete '
                            'action spaces'.format(name))

    elif action and not isinstance(env.action_space, discrete):
        raise TypeError('{} Learner only works with discrete '
                        'action spaces'.format(name))

    return is_tuple_obs, is_tuple_action

