import numpy as np
import gym
import contextlib
with contextlib.redirect_stdout(None):
    import pybulletgym
import pickle
import cloudpickle
from multiprocessing import Pipe, Process
import timeit
import os
import warnings
from tensorflow.python.util import deprecation
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
deprecation._PRINT_DEPRECATION_WARNINGS = False
warnings.filterwarnings("ignore")

class SubprocVecEnv():

    def __init__(self, env_fns, largest_obs_dim):
        self.largest_obs_dim = largest_obs_dim
        self.waiting = False
        self.closed = False
        self.no_of_envs = len(env_fns)
        self.remotes, self.work_remotes = \
            zip(*[Pipe() for _ in range(self.no_of_envs)])
        self.ps = []

        for wrk, rem, fn in zip(self.work_remotes, self.remotes, env_fns):
            proc = Process(target=worker,
                           args=(wrk, rem, CloudpickleWrapper(fn), self.largest_obs_dim))
            self.ps.append(proc)

        for p in self.ps:
            p.daemon = True
            p.start()

        for remote in self.work_remotes:
            remote.close()

    def step_async(self, actions):
        if self.waiting:
            raise NotImplementedError
        self.waiting = True

        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))

    def step_wait(self):
        if not self.waiting:
            raise NotImplementedError
        self.waiting = False

        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)
        obs = np.stack(obs)
        rews = np.stack(rews)
        dones = np.stack(dones)
        return obs, rews, dones, infos

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def get_rms(self):
        for remote in self.remotes:
            remote.send(('get_rms', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def get_original_reward(self):
        for remote in self.remotes:
            remote.send(('get_original_reward', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True


class CloudpickleWrapper(object):
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        self.x = pickle.loads(ob)

    def __call__(self):
        return self.x()


def worker(remote, parent_remote, env_fn, largest_obs_dim):
    parent_remote.close()
    env = env_fn()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            if len(np.shape(ob)) > 1:
                ob = ob[0, :]
                reward = reward[0]
                done = done[0]
            ob = np.pad(ob, pad_width=(0, largest_obs_dim - np.shape(ob)[-1]), mode="constant")
            ob = ob.astype(np.float32)


            remote.send((ob, reward, done, info))
        elif cmd == 'render':
            remote.send(env.render())

        elif cmd == 'close':
            remote.close()
            break

        elif cmd == 'get_rms':
            rms_mean = env.obs_rms.mean
            rms_var = env.obs_rms.var
            remote.send((rms_mean, rms_var))

        elif cmd == 'get_original_reward':
            r = env.get_original_reward()
            remote.send(r)

        elif cmd == 'reset':
            ob = env.reset()
            if len(np.shape(ob)) > 1:
                ob = ob[0, :]
            ob = np.pad(ob, pad_width=(0, largest_obs_dim - np.shape(ob)[-1]), mode="constant")
            ob = ob.astype(np.float32)
            remote.send(ob)

        else:
            raise NotImplementedError


def make_mp_envs(env_id, num_env, norm_reward=True, seed=1234, start_idx=0, largest_obs_dim=28):
    def make_env(rank):
        def fn():
            env = DummyVecEnv([lambda: gym.make(env_id)])
            env = VecNormalize(env, norm_obs=False, norm_reward=norm_reward, clip_obs=1000., clip_reward=10000)
            if seed is not None:
                env.seed(seed + rank)
            return env

        return fn

    return SubprocVecEnv([make_env(i + start_idx) for i in range(num_env)], largest_obs_dim)

# envs = make_mp_envs("Walker2DPyBulletEnv-v0", 128, 1234)
# start = timeit.default_timer()
# envs.reset()
# samples = []
# goal_n_samples = 4000
# while len(samples) <= goal_n_samples:
#     actions = np.random.uniform(low=0, high=1.0, size=(16, 6))
#     s, r, d, _ = envs.step(actions)
#     print(s.shape, d)
#     samples.extend(s)
# end = timeit.default_timer()
# print("took %.2f seconds for %d samples" % (end-start, len(samples)))