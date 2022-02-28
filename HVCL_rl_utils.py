import random
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, InputLayer, BatchNormalization, Input, Concatenate, LayerNormalization
from tensorflow.keras.regularizers import l2
import tensorflow_probability as tfp
from collections import deque
from DenseMoe import *


class ReplayBuffer(object):
    def __init__(self, buffer_size: int) -> None:
        self.buffer_size: int = buffer_size
        self.num_experiences: int = 0
        self.buffer: deque = deque()

    def get_batch(self, num_samples: int) -> np.array:

        # Randomly sample batch_size examples
        experiences = random.sample(self.buffer, num_samples)
        return {
            "states0": np.asarray([exp[0] for exp in experiences], np.float32),
            "actions": np.asarray([exp[1] for exp in experiences], np.float32),
            "rewards": np.asarray([exp[2] for exp in experiences], np.float32),
            "states1": np.asarray([exp[3] for exp in experiences], np.float32),
            "terminals1": np.asarray([exp[4] for exp in experiences], np.float32)
        }

    def add(self, state: np.ndarray, action: np.ndarray, reward: float, new_state: np.ndarray, done: bool):
        experience = (state, action, reward, new_state, done)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)
        # print("Added Experience:", experience, "Count is now %d"%self.num_experiences)

    @property
    def size(self) -> int:
        return self.buffer_size

    @property
    def n_entries(self) -> int:
        # If buffer is full, return buffer size
        # Otherwise, return experience counter
        return self.num_experiences

    def get_last_n(self, n):
        if n > self.num_experiences:
            n = self.num_experiences
        data = []
        for _ in range(n):
            data.append(self.buffer.popleft())
        return data

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0

    def extend(self, data):
        self.num_experiences += len(data)
        self.buffer.extend(data)


class ActorNetwork(Model):
    def __init__(self, n_hidden_units, n_actions, logprob_epsilon, expert_beta, gating_beta, inputdim,
                 n_experts=1, vmoe=True, k=1):
        super(ActorNetwork, self).__init__()
        self.deep = True
        self.vmoe = vmoe
        self.n_actions = n_actions
        self.n_experts = n_experts
        self.expert_beta = expert_beta
        self.gating_beta = gating_beta
        self.diversity_bonus = 1e-1
        self.kl_divergence_function = (lambda q, p: ds.kl_divergence(q, p) / tf.cast(1.0, dtype=tf.float32))
        self.entropy_function = (lambda p: tf.maximum(p.entropy(), 0.0) / tf.cast(12560, dtype=tf.float32))
        self.logprob_epsilon = logprob_epsilon
        self.k = k
        input_layer = Input(shape=inputdim)
        if self.vmoe:
            if self.deep:
                x = DenseMoVE(units=n_hidden_units, expert_activation=tf.nn.leaky_relu,
                              gating_activation=tf.nn.softmax,
                              expert_beta=self.expert_beta, gating_beta=self.gating_beta,
                              n_experts=self.n_experts,
                              diversity_bonus=self.diversity_bonus,
                              kl_div_fun=self.kl_divergence_function,
                              entropy_fun=self.entropy_function,
                              k=self.k,
                              name="actor_dense_0")(input_layer)
                x = Concatenate(axis=-1)([x, input_layer])
                x = DenseMoVE(units=n_hidden_units, expert_activation=tf.nn.leaky_relu,
                              gating_activation=tf.nn.softmax,
                              expert_beta=self.expert_beta, gating_beta=self.gating_beta,
                              n_experts=self.n_experts,
                              diversity_bonus=self.diversity_bonus,
                              kl_div_fun=self.kl_divergence_function,
                              entropy_fun=self.entropy_function,
                              k=self.k,
                              name="actor_dense_1")(x)
                x = Concatenate(axis=-1)([x, input_layer])
                x = DenseMoVE(units=n_hidden_units, expert_activation=tf.nn.leaky_relu,
                              gating_activation=tf.nn.softmax,
                              expert_beta=self.expert_beta, gating_beta=self.gating_beta,
                              n_experts=self.n_experts,
                              diversity_bonus=self.diversity_bonus,
                              kl_div_fun=self.kl_divergence_function,
                              entropy_fun=self.entropy_function,
                              k=self.k,
                              name="actor_dense_2")(x)
                x = Concatenate(axis=-1)([x, input_layer])
                x = DenseMoVE(units=n_hidden_units, expert_activation=tf.nn.leaky_relu,
                              gating_activation=tf.nn.softmax,
                              expert_beta=self.expert_beta, gating_beta=self.gating_beta,
                              n_experts=self.n_experts,
                              diversity_bonus=self.diversity_bonus,
                              kl_div_fun=self.kl_divergence_function,
                              entropy_fun=self.entropy_function,
                              k=self.k,
                              name="actor_dense_3")(x)
                x = Concatenate(axis=-1)([x, input_layer])
            else:
                x = DenseMoVE(units=n_hidden_units, expert_activation=tf.nn.leaky_relu,
                              gating_activation=tf.nn.softmax,
                              expert_beta=self.expert_beta, gating_beta=self.gating_beta,
                              n_experts=self.n_experts,
                              diversity_bonus=self.diversity_bonus,
                              kl_div_fun=self.kl_divergence_function,
                              entropy_fun=self.entropy_function,
                              k=self.k,
                              name="actor_dense_0")(input_layer)
                x = Concatenate(axis=-1)([x, input_layer])
                x = DenseMoVE(units=n_hidden_units, expert_activation=tf.nn.leaky_relu,
                              gating_activation=tf.nn.softmax,
                              expert_beta=self.expert_beta, gating_beta=self.gating_beta,
                              n_experts=self.n_experts,
                              diversity_bonus=self.diversity_bonus,
                              kl_div_fun=self.kl_divergence_function,
                              entropy_fun=self.entropy_function,
                              k=self.k,
                              name="actor_dense_1")(x)
                x = Concatenate(axis=-1)([x, input_layer])
            actor_output = DenseMoVE(units=n_actions * 2, expert_activation=None,
                                     gating_activation=tf.nn.softmax,
                                     expert_beta=self.expert_beta, gating_beta=self.gating_beta,
                                     n_experts=self.n_experts, diversity_bonus=self.diversity_bonus,
                                     kl_div_fun=self.kl_divergence_function, entropy_fun=self.entropy_function,
                                     k=self.k,
                                     name="actor_output")(x)
        else:
            x = Dense(units=n_hidden_units, activation=tf.nn.relu)(input_layer)
            x = Dense(units=n_hidden_units, activation=tf.nn.relu)(x)
            actor_output = Dense(units=n_actions * 2, activation=None)(x)
        self.model_layers = Model(input_layer, actor_output)
        self.model_layers.compile()
        print(self.model_layers.summary())
        # self.log_std = Dense(units=n_actions, activation=None, kernel_initializer="glorot_uniform", name="actor_log_std")

    @tf.function(experimental_relax_shapes=True)
    def gaussian_likelihood(self, input, mu, log_std):
        """
        Helper to compute log likelihood of a gaussian.
        Here we assume this is a Diagonal Gaussian.
        :param input_: (tf.Tensor)
        :param mu_: (tf.Tensor)
        :param log_std: (tf.Tensor)
        :return: (tf.Tensor)
        """
        pre_sum = -0.5 * (
                ((input - mu) / (tf.exp(log_std) + self.logprob_epsilon)) ** 2 + 2 * log_std + np.log(2 * np.pi))
        return tf.reduce_sum(pre_sum, axis=1)

    @tf.function(experimental_relax_shapes=True)
    def call(self, inp):
        x = self.model_layers(inp)
        mean = x[:, :self.n_actions]
        log_std = x[:, self.n_actions:]
        log_std = tf.clip_by_value(t=log_std, clip_value_min=-20, clip_value_max=2)
        std = tf.exp(log_std)
        action = mean + tf.random.normal(tf.shape(mean)) * std
        squashed_actions = tf.tanh(action)
        # numerically unstable:
        # logprob = action_dist.log_prob(action) - tf.reduce_sum(
        #     tf.math.log((1.0 - tf.pow(squashed_actions, 2)) + self.logprob_epsilon), axis=-1)
        # ref: https://github.com/vitchyr/rlkit/blob/0073d73235d7b4265cd9abe1683b30786d863ffe/rlkit/torch/distributions.py#L358
        # ref: https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py#L73
        # numerically stable version:
        # logprob = self.gaussian_likelihood(action, mean, log_std) - tf.reduce_sum((2 * (np.log(2) - squashed_actions - tf.math.softplus(-2 * squashed_actions))), axis=-1)
        logprob = self.gaussian_likelihood(mu=mean, input=action, log_std=log_std) - tf.reduce_sum(
            tf.math.log((1.0 - tf.pow(squashed_actions, 2)) + self.logprob_epsilon), axis=-1)
        return squashed_actions, logprob, tf.nn.tanh(mean), std

    @tf.function(experimental_relax_shapes=True)
    def loss_call(self, inp):
        x = self.model_layers(inp)
        mean = x[:, :self.n_actions]
        log_std = x[:, self.n_actions:]
        log_std = tf.clip_by_value(t=log_std, clip_value_min=-20, clip_value_max=2)
        std = tf.exp(log_std)
        action = mean + tf.random.normal(tf.shape(mean)) * std
        squashed_actions = tf.tanh(action)
        logprob = self.gaussian_likelihood(mu=mean, input=action, log_std=log_std) - tf.reduce_sum(
            tf.math.log((1.0 - tf.pow(squashed_actions, 2)) + self.logprob_epsilon), axis=-1)
        return squashed_actions, logprob, tf.nn.tanh(mean), std, tf.reduce_sum(self.model_layers.losses)

    @tf.function
    def get_model_losses(self):
        return tf.reduce_sum(self.model_layers.losses)

    def _get_params(self):
        ''
        with self.graph.as_default():
            params = tf.trainable_variables()
        names = [p.name for p in params]
        values = self.sess.run(params)
        params = {k: v for k, v in zip(names, values)}
        return params

    def __getstate__(self):
        params = self._get_params()
        state = self.args_copy, params
        return state

    def __setstate__(self, state):
        args, params = state
        self.__init__(**args)
        self.restore_params(params)


class SoftQNetwork(Model):
    def __init__(self, n_hidden_units, expert_beta, gating_beta, inputdim, qi="0", n_experts=1,
                 vmoe=True, k=1):
        super(SoftQNetwork, self).__init__()
        self.deep = True
        self.vmoe = vmoe
        self.k = k
        self.n_experts = n_experts
        self.expert_beta = expert_beta
        self.gating_beta = gating_beta
        self.diversity_bonus = 1e-1
        self.kl_divergence_function = (lambda q, p: ds.kl_divergence(q, p) / tf.cast(1.0, dtype=tf.float32))
        self.entropy_function = (lambda p: tf.maximum(p.entropy(), 0.0) / tf.cast(10000.0, dtype=tf.float32))
        input_layer = Input(shape=inputdim)
        if self.vmoe:
            if self.deep:
                x = DenseMoVE(units=n_hidden_units, expert_activation=tf.nn.leaky_relu,
                              gating_activation=tf.nn.softmax,
                              name="softq_dense_in_%s" % (qi),
                              expert_beta=self.expert_beta, gating_beta=self.gating_beta,
                              n_experts=self.n_experts,
                              diversity_bonus=self.diversity_bonus,
                              kl_div_fun=self.kl_divergence_function,
                              k=self.k,
                              entropy_fun=self.entropy_function)(input_layer)
                x = Concatenate(axis=-1)([x, input_layer])
                x = DenseMoVE(units=n_hidden_units, expert_activation=tf.nn.leaky_relu,
                              gating_activation=tf.nn.softmax, name="softq_dense_0_%s" % (qi),
                              expert_beta=self.expert_beta, gating_beta=self.gating_beta,
                              n_experts=self.n_experts,
                              diversity_bonus=self.diversity_bonus,
                              kl_div_fun=self.kl_divergence_function,
                              k=self.k,
                              entropy_fun=self.entropy_function)(x)
                x = Concatenate(axis=-1)([x, input_layer])
                x = DenseMoVE(units=n_hidden_units, expert_activation=tf.nn.leaky_relu,
                              gating_activation=tf.nn.softmax, name="softq_dense_1_%s" % (qi),
                              expert_beta=self.expert_beta, gating_beta=self.gating_beta,
                              n_experts=self.n_experts,
                              diversity_bonus=self.diversity_bonus,
                              kl_div_fun=self.kl_divergence_function,
                              k=self.k,
                              entropy_fun=self.entropy_function)(x)
                x = Concatenate(axis=-1)([x, input_layer])
                x = DenseMoVE(units=n_hidden_units, expert_activation=tf.nn.leaky_relu,
                              gating_activation=tf.nn.softmax, name="softq_dense_3_%s" % (qi),
                              expert_beta=self.expert_beta, gating_beta=self.gating_beta,
                              n_experts=self.n_experts,
                              diversity_bonus=self.diversity_bonus,
                              kl_div_fun=self.kl_divergence_function,
                              k=self.k,
                              entropy_fun=self.entropy_function)(x)
                x = Concatenate(axis=-1)([x, input_layer])
            else:
                x = DenseMoVE(units=n_hidden_units, expert_activation=tf.nn.leaky_relu,
                              gating_activation=tf.nn.softmax,
                              name="softq_dense_in_%s" % (qi),
                              expert_beta=self.expert_beta, gating_beta=self.gating_beta,
                              n_experts=self.n_experts,
                              diversity_bonus=self.diversity_bonus,
                              kl_div_fun=self.kl_divergence_function,
                              k=self.k,
                              entropy_fun=self.entropy_function)(input_layer)
                x = Concatenate(axis=-1)([x, input_layer])
                x = DenseMoVE(units=n_hidden_units, expert_activation=tf.nn.leaky_relu,
                              gating_activation=tf.nn.softmax, name="softq_dense_0_%s" % (qi),
                              expert_beta=self.expert_beta, gating_beta=self.gating_beta,
                              n_experts=self.n_experts,
                              diversity_bonus=self.diversity_bonus,
                              kl_div_fun=self.kl_divergence_function,
                              k=self.k,
                              entropy_fun=self.entropy_function)(x)
                x = Concatenate(axis=-1)([x, input_layer])
            out = DenseMoVE(units=1, expert_activation=None,
                            gating_activation=tf.nn.softmax, name="softq_dense_out_%s" % (qi),
                            expert_beta=self.expert_beta, gating_beta=self.gating_beta,
                            n_experts=self.n_experts,
                            diversity_bonus=self.diversity_bonus,
                            kl_div_fun=self.kl_divergence_function,
                            k=self.k,
                            entropy_fun=self.entropy_function)(x)
        else:
            x = Dense(units=n_hidden_units, activation=tf.nn.relu)(input_layer)
            x = Dense(units=n_hidden_units, activation=tf.nn.relu)(x)
            out = Dense(units=1, activation=None)(x)
        self.model_layers = Model(input_layer, out)
        self.model_layers.compile()
        print(self.model_layers.summary())

    @tf.function
    def get_model_losses(self):
        return tf.reduce_sum(self.model_layers.losses)

    @tf.function(experimental_relax_shapes=True)
    def call(self, states, actions):
        x = tf.concat([states, actions], -1)
        return self.model_layers(x)

    @tf.function(experimental_relax_shapes=True)
    def loss_call(self, states, actions):
        x = tf.concat([states, actions], -1)
        return self.model_layers(x), tf.reduce_sum(self.model_layers.losses)


def plot_episode_stats(actor_losses, softq_losses, action_logprob_means, episode_rewards, smoothing_window=15):
    # Plot the episode length over time
    plt.figure(figsize=(10, 5))
    plt.plot(actor_losses, label="Actor")
    plt.xlabel("Episode")
    plt.ylabel("Actor Loss")
    plt.title("Actor Loss over Time")

    plt.figure(figsize=(10, 5))
    plt.plot(softq_losses, label="Soft-Q")
    plt.xlabel("Episode")
    plt.ylabel("Soft-Q Loss")
    plt.title("Soft-Q Loss over Time")

    plt.figure(figsize=(10, 5))
    plt.plot(action_logprob_means, label="log(p(a))")
    plt.xlabel("Episode")
    plt.ylabel("Log Prob")
    plt.title("Log Prob over Time")

    # Plot the episode reward over time
    plt.figure(figsize=(10, 5))
    rewards_smoothed = pd.Series(episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    plt.show()


def plot_reward(episode_rewards):
    plt.plot(range(len(episode_rewards)), episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("Episode Reward over Time")
    plt.show()


class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.2, min_sigma=-0.2, decay_period=100):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space
        self.low = -1.0
        self.high = 1.0
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action * 0.2 + ou_state, self.low, self.high)
