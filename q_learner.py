import os
import gym
import numpy as np
import cPickle as pickle


def epsilon_greedy_action(env, q_func, obs_hash, epsilon=0.5):
    rv = np.random.rand()
    state_actions = filter(lambda x: x[0] == obs_hash, q_func.items())

    if rv >= epsilon and len(state_actions) > 0:
        # greedily select best action
        qvals = [q_func[sa] for sa in state_actions]
        action = state_actions[np.argmax(qvals)][1]
    else:
        # uniform-random sample action
        action = np.random.randint(env.action_space.n)
    return action


def q_learner(env, max_time, alpha, gamma, q_func=None, epsilon=0.5, render=False):
    """
    Naive implementation of standard Q-learning using the update

        Q(s, a) <- Q(s, a) + alpha * (reward + gamma * max_a' Q(s', a') - Q(s, a))
    """
    obs = env.reset()
    obs_hash = hash(obs.tostring())
    total_reward = 0.

    if not q_func:
        q_func = {}

    # train on an episode
    for _ in range(max_time):
        if render:
            env.render()

        # generate action from epsilon-greedy behavior policy
        action = epsilon_greedy_action(env, q_func, obs_hash, epsilon=epsilon)
        state_action = (obs_hash, action)
        obs_new, reward, done, _ = env.step(action)
        total_reward += reward

        obs_new_hash = hash(obs_new.tostring())

        # select the optimal action according to the current Q estimate
        action_new = epsilon_greedy_action(env, q_func, obs_hash, epsilon=1.)
        state_action_new = (obs_new_hash, action_new)

        # if we haven't seen this state-action pair, initialize q(s,a) at 0
        if state_action not in q_func:
            q_func[state_action] = 0.

        if state_action_new not in q_func:
            q_func[state_action_new] = 0.

        # Off-policy update
        q_func[state_action] = q_func[state_action] + alpha * (
            reward + gamma * q_func[state_action_new] - q_func[state_action]
        )

        state_action = state_action_new

        if done:
            break

    print('Total reward: {}'.format(total_reward))
    return q_func


def greedy_policy(env, q_func, render=True):
    obs = env.reset()
    obs_hash = hash(obs.tostring())

    for _ in range(1000):
        if render:
            env.render()

        # execute greedy policy wrt the learned q function
        action = epsilon_greedy_action(env, q_func, obs_hash, epsilon=1.)
        state_action = (obs_hash, action)
        obs_new, reward, done, info = env.step(action)

        if done:
            break

        obs_hash = hash(obs_new.tostring())

if __name__ == "__main__":
    env = gym.make('Breakout-v0')

    max_time = 100   # episode duration
    alpha = 0.75     # learning rate
    gamma = 0.9      # discount rate
    epsilon = 0.1    # probability of not executing greedy policy
    n_episodes = 500
    render = False  # render progress during traning

    if os.path.exists('q_func.pkl'):
        with open('q_func.pkl') as handle:
            q_func = pickle.load(handle)
    else:
        q_func = None

    for idx in xrange(n_episodes):
        print('Running episode {}'.format(idx + 1))
        q_func = q_learner(env, max_time, alpha, gamma, q_func,
                           epsilon=epsilon, render=render)

    print('Done training!')

    for _ in xrange(20):
        print('Starting new session')
        greedy_policy(env, q_func, render=True)
