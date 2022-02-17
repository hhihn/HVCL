#! -*- coding:utf-8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import numpy as np
# import random
import sum_tree


class BetaSchedule(object):
    def __init__(self, conf=None):
        self.batch_size = int(conf['batch_size'] if 'batch_size' in conf else 32)

        self.beta_zero = conf['beta_zero'] if 'beta_zero' in conf else 0.4
        self.learn_start = int(conf['learn_start'] if 'learn_start' in conf else 2)
        # http://www.evernote.com/l/ACnDUVK3ShVEO7fDm38joUGNhDik3fFaB5o/
        self.total_steps = int(conf['total_steps'] if 'total_steps' in conf else 5)
        self.beta_grad = (1 - self.beta_zero) / (self.total_steps - self.learn_start)

    def get_beta(self, global_step):
        # beta, increase by global_step, max 1
        beta = min(self.beta_zero + (global_step - self.learn_start) * self.beta_grad, 1)
        return beta, self.batch_size


class ExperienceBuffer(object):
    """ The class represents prioritized experience replay buffer.

    The class has functions: store samples, pick samples with
    probability in proportion to sample's priority, update
    each sample's priority, reset alpha.

    see https://arxiv.org/pdf/1511.05952.pdf .

    """

    def __init__(self, conf={}):
        """ Prioritized experience replay buffer initialization.

        Parameters
        ----------
        memory_size : int
            sample size to be stored
        batch_size : int
            batch size to be selected by `select` method
        alpha: float
            exponent determine how much prioritization.
            Prob_i \sim priority_i**alpha/sum(priority**alpha)
        """
        self.beta_sched = BetaSchedule(conf)
        self._max_priority = 1.0

        self.index = 0
        self.record_size = 0
        self.isFull = False
        self.prioritized_replay_eps = 1e-6

        self.largest_obs_dim = conf['largest_obs_dim']
        self.largest_act_dim = conf['largest_act_dim']

        if conf is not None:
            memory_size = int(conf['size']) if 'size' in conf else 10000
            self.memory_size = memory_size
            self.tree = sum_tree.SumTree(memory_size)
            # self.batch_size = batch_size
            self.alpha = conf['alpha'] if 'alpha' in conf else 0.7
            if 'prioritized_replay_eps' in conf:
                self.prioritized_replay_eps = float(conf['prioritized_replay_eps'])

    def save(self, filename):
        assert False, "proportional.experience.save() is not implemented!"
        pass

    def load(self, filename):
        assert False, "proportional.experience.load() is not implemented!"
        pass

    def fix_index(self):
        """
        get next insert index
        :return: index, int
        """
        if self.record_size <= self.memory_size:
            self.record_size += 1
        if self.index % self.memory_size == 0:
            self.index = 1
            return self.index
        else:
            self.index += 1
            return self.index

    def add(self, state: np.ndarray, action: np.ndarray, reward: float, new_state: np.ndarray, done: bool, priority=None):
        """ Add new sample.

        Parameters
        ----------
        data : object
            new sample
        priority : float
            sample's priority, i.e., the td error
        """
        data = (state, action, reward, new_state, done)
        if priority is None:
            priority = self._max_priority
        self.fix_index()
        self.tree.add(data, priority ** self.alpha)

    def get_batch(self, global_step, num_samples=None):
        beta, batch_size = self.beta_sched.get_beta(global_step)
        if num_samples is None:
            num_samples = batch_size
        return self.select(beta, batch_size=num_samples)

    @property
    def n_entries(self):
        return self.tree.filled_size()

    def select(self, beta, batch_size=32):
        """ The method return samples randomly.

        Parameters
        ----------
        beta : float

        Returns
        -------
        out :
            list of samples
        weights:
            list of weight
        indices:
            list of sample indices
            The indices indicate sample positions in a sum tree.
        """

        if self.tree.filled_size() < batch_size:
            batch_size = self.tree.filled_size()

        out = {'states0': np.empty(shape=(0, self.largest_obs_dim), dtype=np.float32),
               'actions': np.empty(shape=(0, self.largest_act_dim), dtype=np.float32),
               'rewards': np.empty(shape=0, dtype=np.float32),
               'states1': np.empty(shape=(0, self.largest_obs_dim), dtype=np.float32),
               'terminals1': np.empty(shape=0, dtype=np.float32)}

        indices = []
        weights = []
        priorities = []
        rand_vals = np.random.uniform(low=0.0, high=0.9999, size=batch_size)
        for r in rand_vals:  # range(batch_size):
            # r = random.random()
            data, priority, index = self.tree.find(r)
            if data is None:
                print("data", data)
                print("beta", beta)
                print("priorities", priority)
                print("index", index)
                print("rand", r)
                print("batchsize", batch_size)
                print("trying to get a new one...")
                tries = 0
                while data is None:
                    tries = tries + 1
                    rt = np.random.random(1)
                    data, priority, index = self.tree.find(rt)
                print("done in %d tries." % tries)
            priorities.append(priority)
            weights.append((1. / (self.memory_size * priority)) ** beta if priority > 1e-16 else 0)
            indices.append(index)
            for i, k in enumerate(out.keys()):
                if k in ['states0', 'actions', 'states1']:
                    d = np.reshape(data[i], newshape=(1, len(data[i])))
                else:
                    d = [data[i]]
                out[k] = np.append(out[k], d, axis=0)
            # self.update_priority([index], [0]) # To avoid duplicating

        self.update_priority(indices, priorities)  # Revert priorities
        # weights /= max(weights) # Normalize for stability
        w = np.array(weights)
        w = np.divide(w, max(w))
        return out, w, indices

    def update_priority(self, indices, priorities):
        """ The methods update sample priority.

        Parameters
        ----------
        indices :
            list of sample indices
        """
        for i, p in zip(indices, priorities):
            self.tree.val_update(i, (p + self.prioritized_replay_eps) ** self.alpha)

    def reset_alpha(self, alpha):
        """ Reset a exponent alpha.

        Parameters
        ----------
        alpha : float
        """
        self.alpha, old_alpha = alpha, self.alpha
        priorities = [self.tree.get_val(i) ** -old_alpha for i in range(self.tree.filled_size())]
        self.update_priority(range(self.tree.filled_size()), priorities)

    def rebalance(self):
        pass



