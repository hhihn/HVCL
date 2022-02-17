import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
######### Configuration files #########
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
#######################################

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False
import tensorflow as tf
import gym
import numpy as np
import pybulletgym.envs
from SAC_agent import Agent
from SAC_utils import ReplayBuffer
from ExperienceBuffer import ExperienceBuffer
from SAC_rla import SAC
from parallel_gym import make_mp_envs
import argparse

if __name__ == "__main__":
    print('TensorFlow version: %s' % tf.__version__)
    print('Keras version: %s' % tf.keras.__version__)
    # tf.random.set_seed(seed)
    # np.random.seed(seed)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--hvcl',
        action='store_true',
        dest='hvcl',
        help='Use HVCL')
    parser.add_argument(
        '--no-hvcl',
        action='store_false',
        dest='hvcl',
        help='Use HVCL')
    parser.add_argument(
        '--replay',
        action='store_true',
        dest='replay',
        help='Use replay buffer')
    parser.add_argument(
        '--no-replay',
        action='store_false',
        dest='replay',
        help='Use replay buffer')
    parser.add_argument(
        '--batch_s',
        type=int,
        default=256,
        help='Mini-Batch Size for SAC Updates')
    parser.add_argument(
        '--hidden_u',
        type=int,
        default=32,
        help='Number of hidden units per layer')
    parser.add_argument(
        '--lr',
        type=float,
        default=3e-4,
        help='Learning rate')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='Discount factor')
    parser.add_argument(
        '--e_beta',
        type=float,
        default=1.0,
        help='Expert beta')
    parser.add_argument(
        '--g_beta',
        type=float,
        default=1.0,
        help='Gate beta')
    parser.add_argument(
        '--polyak_coef',
        type=float,
        default=0.995,
        help='Polyak Coefficient for running mean updates of target networks')
    parser.add_argument(
        '--train_steps',
        type=int,
        default=1000000,
        help='Total number of Environment Interactions')
    parser.add_argument(
        '--replay_start_size',
        type=int,
        default=10000,
        help='Number of samples to collect before training starts')
    parser.add_argument(
        '--buffer_size',
        type=int,
        default=1000000,
        help='Size of replay buffer')
    parser.add_argument(
        '--train_interval',
        type=int,
        default=5000,
        help='Update Model every n steps')
    parser.add_argument(
        '--n_exp',
        type=int,
        default=2,
        help='Number of Experts')
    parser.add_argument(
        '--k',
        type=int,
        default=1,
        help='Gating top-k selection parameter')
    parser.add_argument(
        '--save_pref',
        default="",
        help='savename for npy files')

    environments = ['Walker2DPyBulletEnv-v0', 'HalfCheetahPyBulletEnv-v0', 'AntPyBulletEnv-v0',
                    'InvertedDoublePendulumPyBulletEnv-v0', 'HopperPyBulletEnv-v0']
    args = parser.parse_args()
    gamma = args.gamma
    polyak_coef = args.polyak_coef
    train_n_steps = args.train_steps
    batch_size = args.batch_s
    save_pref = args.save_pref
    n_experts = args.n_exp
    train_interval = args.train_interval
    n_hidden_units = args.hidden_u
    buffer_size = args.buffer_size
    replay_start_size = args.replay_start_size
    lr = args.lr
    vmoe = args.hvcl
    exp_beta = args.e_beta
    gate_beta = args.g_beta
    replay = False  # args.replay
    k = min(n_experts, args.k)
    # Creating a ReplayBuffer for the training process to store old samples in
    if replay:
        continual_replay_buffer = ExperienceBuffer({'size': 50000 * len(environments),
                                                    'batch_size': batch_size,
                                                    'total_steps': train_n_steps * (len(environments) - 1)})
    else:
        continual_replay_buffer = None
    old_models = []
    largest_obs_dim = -1
    largest_act_dim = -1
    obs_env = ""
    act_env = ""
    for env_id in environments:
        env = gym.make(env_id)
        obs_dim = env.observation_space.shape[0]
        n_actions = env.action_space.shape[0]
        if obs_dim > largest_obs_dim:
            largest_obs_dim = obs_dim
            obs_env = env_id
        if n_actions > largest_act_dim:
            largest_act_dim = n_actions
            act_env = env_id

    print("Largest Obs Dim: %d, from Env %s" % (largest_obs_dim, obs_env))
    print("Largest Act Dim: %d, from Env %s" % (largest_act_dim, act_env))

    eval_envs = []
    eval_baselines = []
    for env_itr, env_id in enumerate(environments):
        min_len = 1e10
        # We first choose a model
        model = SAC(save_dir='./',
                    discount=gamma, lr=lr, polyak_coef=polyak_coef, largest_act_dim=largest_act_dim,
                    largest_obs_dim=largest_obs_dim, n_hidden_units=n_hidden_units,
                    n_experts=n_experts, vmoe=vmoe, task=env_itr, total_episodes=train_n_steps // train_interval,
                    gate_beta=gate_beta, exp_beta=exp_beta, k=k)

        # Creating a ReplayBuffer for the training process
        replay_buffer = ExperienceBuffer(
            {'size': buffer_size, 'batch_size': batch_size, 'total_steps': train_n_steps,
             'largest_obs_dim': largest_obs_dim, 'largest_act_dim': largest_act_dim})
        # create an Agent to train / test the model
        agent = Agent(model=model, replay_buffer=replay_buffer,
                      replay_start_size=replay_start_size, batch_size=batch_size,
                      train_n_steps=train_n_steps, largest_act_dim=largest_act_dim, largest_obs_dim=largest_obs_dim,
                      vmoe=vmoe, task=env_itr, train_interval=train_interval)
        env = DummyVecEnv([lambda: gym.make(env_id)])  # gym.make(env_id)#
        # Automatically normalize the input features
        train_env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=100000., clip_reward=10000.)
        parallel_eval_envs = make_mp_envs(env_id=env_id, num_env=10, norm_reward=False, start_idx=(1 + env_itr) * 100,
                                          largest_obs_dim=largest_obs_dim)
        eval_envs.append(parallel_eval_envs)

        obs_dim = train_env.observation_space.shape[0]
        print("Obs dim:", obs_dim, env_id)
        n_actions = train_env.action_space.shape[0]
        print("Acts dim:", n_actions, env_id)
        act_lim = train_env.action_space.high
        model.reset_state(actions_dim=train_env.action_space.shape, obs_dim=obs_dim, n_actions=n_actions,
                          act_lim=act_lim, env=train_env)
        agent.reset_state(train_env=train_env, n_actions=n_actions, eval_envs=eval_envs,
                          continual_replay_buffer=continual_replay_buffer)
        if len(old_models) and vmoe:
            model.copy_weights(old_models=old_models)
        e_reward, continual_replay_buffer = agent.train()
        if vmoe:
            old_models = model.get_models()
        eval_baselines.append(e_reward)
        del model
        del agent

    np.save(arr=eval_baselines,
            file="%s_%s_contreplay_prio_replay_weighted_%s_%d_exp_%d_k_%d_units_%.3f_gb_%.3f_eb_1M_soft_deep_norms_eval_baselines.npy" % (
                save_pref,
                "with" if replay else "no",
                "vmoe" if vmoe else "dense",
                n_experts,
                k,
                n_hidden_units,
                gate_beta,
                exp_beta))
