import gym
import numpy as np
from time import time

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd

def init_model(**kwargs):
    """
    Initialize a simple 2 hidden layer feedforward net for approximating the Q
    function.  Network takes as input an observation at time t, and returns the
    vector of Q values for each action available .
    """
    hidden_size = kwargs['hidden_dim']
    n_obs_dims = kwargs['n_obs_dims']
    n_actions = kwargs['n_actions']
    activation = kwargs['activation']

    model = Sequential()
    model.add(Dense(hidden_size, input_shape=(n_obs_dims,), activation=activation))
    model.add(Dense(hidden_size, activation=activation))
    model.add(Dense(n_actions, activation='linear'))
    model.compile(optimizer='rmsprop', loss='mse')
    return model

class ExperienceReplay(object):
    def __init__(self, n_actions, obs_dims, **kwargs):
        self.memory = []
        self.target_net = None
        self.n_actions = n_actions
        self.observation_dims = obs_dims

        self.max_memory = kwargs['mem_limit'] if 'mem_limit' in kwargs else 100
        self.discount = kwargs['gamma'] if 'gamma' in kwargs else 0.9

    def store(self, experience, done):
        # experience = [s, a, r, s']
        self.memory.append([experience, done])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def gen_batch(self, q_network, batch_size=10):
        len_memory = len(self.memory)
        mb_size = min(len_memory, batch_size)
        exp_idxs = np.random.randint(len_memory, size=mb_size)

        X, targets = [], []
        for idx, exp_idx in enumerate(exp_idxs):
            experience = self.memory[exp_idx]
            done = experience[-1]

            obs, act, reward, obs_next = experience[0]

            X.append(obs)

            # use current network to compute target Q values for all
            # non-executed actions
            q_vals_obs = q_network.predict(obs[None, :])[0]
            targets.append(q_vals_obs)

            # only update the target for the executed action
            if done:
                targets[-1][act] = reward
            else:
                # use cached network for computing the target Q value for
                # the executed action
                q_vals_next = self.target_net.predict(obs_next[None, :])[0]

                # reward_t + gamma * max_a' Q(s', a')
                targets[-1][act] = reward + self.discount * np.max(q_vals_next)

        return np.asarray(X), np.asarray(targets)

    def update_target_net(self, q_network):
        """
        Periodically store a snapshot of the q net to use when computing targets.
        This helps to smooth oscillations during learning
        """
        self.target_net = q_network


def epsilon_greedy_action(obs, q_network, epsilon):
    """
    select a random action with probability epsilon, otherwise greedily
    select action based on current q estimate
    """
    val = np.random.rand()
    if val <= epsilon:
        action = np.random.randint(n_actions)
    else:
        obs = obs.reshape(1, len(obs))
        q_vals = q_network.predict(obs)[0]
        action = np.argmax(q_vals)
    return action

def run_episode(env, q_network, exp_replay, epsilon, batch_size=10,
        freeze=False, render=False):
    total_reward = 0.0
    obs = env.reset()

    # run one episode of the RL problem
    for _ in xrange(4000):
        if render:
            env.render()

        # take a step
        action = epsilon_greedy_action(obs, q_network, epsilon)
        obs_next, reward, done, info = env.step(action)
        total_reward += reward

        if not freeze:
            # store experience tuple
            exp_replay.store([obs, action, reward, obs_next], done)

            # sample a random experience minibatch
            X, targets = exp_replay.gen_batch(q_network, batch_size=batch_size)

            # train model on experience minibatch
            q_network.train_on_batch(X, targets)

            # compute mse loss
            #loss = np.sum((targets - q_network.predict(X)) ** 2)
            #print('\tMB MSE: {}'.format(loss / targets.shape[0]))

        if done:
           break

        obs = obs_next
    return q_network, total_reward

if __name__ == "__main__":
    # initialize RL environment
    env = gym.make('CartPole-v0')
    n_actions = env.action_space.n
    n_obs_dims = env.observation_space.shape[0]

    # initialize run parameters
    gamma = 0.9        # temporal discount parameter
    n_epochs = 400     # number of episodes to train the q network on
    epsilon = 0.1      # for epsilon-greedy policy during training
    hidden_dim = 200   # size of the hidden layers in q network
    mem_limit = 100    # how many recent experiences to retain in memory
    batch_size = 100    # desired size of experience replay minibatches
    render = True     # render runs during training

    net_params = \
            {'activation': 'relu',
             'hidden_dim': hidden_dim,
             'n_actions': n_actions,
             'n_obs_dims': n_obs_dims}

    er_params = \
            {'gamma': gamma,
             'mem_limit': mem_limit}

    # initialize network and experience replay objects
    q_network = init_model(**net_params)
    exp_replay = ExperienceReplay(n_actions, n_obs_dims, **er_params)

    # initialize target network weights
    exp_replay.update_target_net(q_network)

    # train q network and accumulate experiences
    t0 = time()
    for idx in xrange(n_epochs):
        q_newtork, total_reward = run_episode(env, q_network, exp_replay,
                epsilon, batch_size=batch_size, render=render)

        # periodically store a "target network" to reduce oscillations during
        # training
        if idx % 25 == 0:
            print('Updating target network')
            exp_replay.update_target_net(q_network)

        print('Total reward on epoch {}/{}:\t{}'.format(idx+1, n_epochs, total_reward))
    print('Training took {} mins'.format((time() - t0) / 60.))

    print('Executing greedy policy using frozen Q-network')
    for idx in xrange(10):
        q_newtork, total_reward = run_episode(env, q_network, exp_replay,
                -1., batch_size=batch_size, freeze=True, render=True)
        print('Total reward on greedy epoch {}/{}:\t{}'.format(idx+1, 10, total_reward))

