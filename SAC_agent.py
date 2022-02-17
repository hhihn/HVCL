from SAC_utils import *
from timeit import default_timer as timer
import time
from parallel_gym import make_mp_envs
import gym
import pybulletgym.envs
from multiprocessing import Pool

class Agent:
    def __init__(self, model, replay_buffer, replay_start_size,
                 batch_size, train_n_steps, largest_act_dim, largest_obs_dim, vmoe, task, train_interval):
        self.model = model
        self.replay_buffer = replay_buffer
        self.replay_start_size = replay_start_size
        self.batch_size = batch_size
        self.largest_act_dim = largest_act_dim
        self.largest_obs_dim = largest_obs_dim
        self.batch_size = batch_size
        self.train_n_steps = train_n_steps
        self.n_timesteps = 100
        self.train_interval = train_interval
        self.eval_interval = 10000
        self.total_steps = 0
        self.total_episodes = 0
        self.vmoe = vmoe
        self.task = task

    def reset_state(self, train_env, n_actions, eval_envs, continual_replay_buffer):
        self.train_env = train_env
        self.n_actions = n_actions
        self.eval_envs = eval_envs
        self.continual_replay_buffer = continual_replay_buffer

    def train(self):
        pre_training_phase = 1
        train_step = 0

        # Noise + epsilon parameters
        epsilon = 1
        train_interval = self.train_interval
        eval_interval = self.eval_interval
        num_env = int(train_interval / self.n_timesteps)
        print("Collecting with %d envs" % num_env)
        sample_eval_ctr = 0
        actor_losses = []
        softq_losses = []
        action_logprob_means = []
        env_evals_r = [[] for _ in range(len(self.eval_envs))]
        durations = deque(maxlen=20)
        parallel_envs = make_mp_envs(env_id=self.train_env.envs[0].spec.id, num_env=num_env, norm_reward=True,
                                     largest_obs_dim=self.largest_obs_dim)
        train_sample_ctr = 0
        duration_sample_ctr = 0
        duration = .0
        while train_step < self.train_n_steps:
            start = timer()

            state = parallel_envs.reset().astype(np.float32)
            for k in range(self.n_timesteps):
                action = self.model.actions(state)
                # get action into right format
                ext_action = action[:, :self.n_actions]
                ext_action = np.reshape(ext_action, newshape=(np.shape(ext_action)[0], 1, np.shape(ext_action)[-1]))

                new_state, reward, done, _ = parallel_envs.step(ext_action)
                for s, a, r, ss, d in zip(state, action, reward, new_state, done):
                    self.replay_buffer.add(state=s, action=a, reward=r, new_state=ss, done=d)

                state = new_state
                train_step += np.shape(state)[0]
                train_sample_ctr += np.shape(state)[0]
                sample_eval_ctr += np.shape(state)[0]
                duration_sample_ctr += np.shape(state)[0]
                # if not pre_training_phase and epsilon > epsilon_min:
                #     epsilon = epsilon * epsilon_dk
                if self.replay_buffer.n_entries > self.replay_start_size:
                    if pre_training_phase:
                        print("Started Training with %d Samples in Buffer" % self.replay_buffer.n_entries)
                    pre_training_phase = 0
                if train_sample_ctr >= train_interval:
                    self.total_episodes += 1
                    break
            if not pre_training_phase and train_sample_ctr >= train_interval:
                K.set_learning_phase(True)
                mean_a_loss = 0.0
                mean_sq_loss = 0.0
                mean_alp_loss = 0.0
                mean_soft_targets_mean = 0.0
                mean_q_values_mean = 0.0
                mean_Qs_log_targets = 0.0
                mean_alpha = 0.0
                train_sample_ctr = 0
                mean_a_model_loss = 0.0
                mean_sq_model_loss = 0.0
                num_updates = 50# train_interval #np.minimum(100, episode_length)
                w_idxs = []
                new_prios = []
                old_w_idxs = []
                old_prios = []
                for nu in range(num_updates):
                    sample, w, w_idx = self.replay_buffer.get_batch(global_step=train_step,
                                                                    num_samples=self.batch_size)
                    if self.continual_replay_buffer is not None and self.vmoe and self.task > 0:
                        old_sample, old_w, old_w_idx = self.continual_replay_buffer.get_batch(
                            global_step=train_step + self.train_n_steps * (self.task + 1),
                            num_samples=self.batch_size)
                    else:
                        old_sample, old_w, old_w_idx = None, None, None
                    softq_loss, actor_loss, action_logprob_mean, soft_targets_mean, q_values_mean, \
                    Qs_log_targets, actor_model_losses, qmodel_losses, alpha, means, stds, td_error, old_td_error = self.model.train(
                        sample,
                        self.batch_size,
                        experience_weights=w,
                        old_sample=old_sample,
                        old_experience_weights=old_w,
                        task=self.task)
                    w_idxs.extend(w_idx)
                    td_error = td_error.numpy()[:, 0]
                    new_prios.extend(td_error)
                    if old_td_error is not None:
                        old_w_idxs.extend(old_w_idx)
                        old_td_error = old_td_error.numpy()[:, 0]
                        old_prios.extend(old_td_error)

                    mean_a_loss += np.array(actor_loss)
                    mean_a_model_loss += np.array(actor_model_losses)
                    mean_sq_model_loss += np.array(qmodel_losses)
                    mean_sq_loss += np.array(softq_loss)
                    mean_alp_loss += np.array(action_logprob_mean)
                    mean_soft_targets_mean += np.array(soft_targets_mean)
                    mean_q_values_mean += np.array(q_values_mean)
                    mean_Qs_log_targets += np.array(Qs_log_targets)
                    mean_alpha += np.array(alpha)
                self.model.updates_performed += 1
                self.replay_buffer.update_priority(indices=w_idxs, priorities=new_prios)
                if self.continual_replay_buffer is not None and len(old_prios):
                    self.continual_replay_buffer.update_priority(indices=old_w_idxs, priorities=old_prios)
                mean_a_loss /= train_interval
                mean_sq_loss /= train_interval
                mean_alp_loss /= train_interval
                mean_soft_targets_mean /= train_interval
                mean_q_values_mean /= train_interval
                mean_Qs_log_targets /= train_interval
                mean_alpha /= train_interval
                actor_losses.append(mean_a_loss)
                softq_losses.append(mean_sq_loss)
                action_logprob_means.append(mean_alp_loss)

                print("Environment is", self.train_env.envs[0].spec.id)
                print("Model Updates:", self.model.updates_performed)
                print("Actor loss is", mean_a_loss)
                print("Actor model loss is", mean_a_model_loss)
                print("Action log-p mean is", mean_alp_loss)
                print("Q loss is", mean_sq_loss)
                print("Q model loss is", mean_sq_model_loss)
                print("Soft Q Targets mean", mean_soft_targets_mean)
                print("Q Values mean", mean_q_values_mean)
                print("Q Log Targets mean", mean_Qs_log_targets)
                print("Alpha", mean_alpha)
                print("Epsilon", epsilon)
                print("Beta:", self.model.actor_exp_beta, self.model.actor_gate_beta, self.model.qnet_exp_beta,
                      self.model.qnet_gate_beta, self.model.beta_schedule[self.model.updates_performed])
                if len(durations):
                    mean_duration = np.mean(durations)
                    time_str = time.strftime('%H:%M:%S',
                                             time.gmtime(mean_duration * (self.train_n_steps - (train_step + 1.0))))
                    print("duration", duration)
                    std_duration = np.std(durations)
                    print("T per Step: %.4f +/- %.4f" % (mean_duration, std_duration))
                    time_str_std = time.strftime('%H:%M:%S',
                                                 time.gmtime(std_duration * (self.train_n_steps - (train_step + 1.0))))
                    print("ETA: %s, +/- %s" % (time_str, time_str_std))
            eval_t = 0
            last_eval_rewars = []
            if sample_eval_ctr >= eval_interval:
                K.set_learning_phase(False)
                sample_eval_ctr = 0
                # eval
                env_actions = [(0, 6), (1, 6), (2, 8), (3, 1), (4, 3)]
                last_eval_rewars = []
                for env_idx, env in enumerate(self.eval_envs):
                    eval_n_actions = env_actions[env_idx][1]
                    # for ei in range(eval_runs):
                    env_running_mask = np.ones(shape=env.no_of_envs)
                    state = env.reset().astype(np.float32)
                    eval_t = 0
                    eval_rewards = [[] for _ in range(env.no_of_envs)]
                    while np.any(env_running_mask):
                        action = self.model.actions(state)
                        ext_action = action[:, :eval_n_actions]
                        ext_action = np.reshape(ext_action,
                                                newshape=(np.shape(ext_action)[0], 1, np.shape(ext_action)[-1]))
                        new_state, reward, eval_done, _ = env.step(ext_action)
                        state = new_state
                        eval_t += 1
                        for di, d in enumerate(env_running_mask):
                            if d:
                                eval_rewards[di].append(reward[di])
                            else:
                                eval_rewards[di].append(0)
                        for di, d in enumerate(eval_done):
                            if d:
                                env_running_mask[di] = 0
                    eval_rewards = np.array(eval_rewards)
                    eval_rewards = np.sum(eval_rewards, axis=-1)
                    mean_eval_episode_reward = np.mean(eval_rewards, axis=-1)
                    env_evals_r[env_idx].append(mean_eval_episode_reward)
                    last_eval_rewars.append(mean_eval_episode_reward)
            end = timer()
            duration = end - start
            durations.append(duration / duration_sample_ctr)
            duration_sample_ctr = 0
            print("env:", self.train_env.envs[0].spec.id
                  , "Episode n.", self.total_episodes, "ended! Steps:", train_step, "The eval rewards are",
                    last_eval_rewars,
                  ", number of steps:", eval_t, end="\r")
        if self.vmoe and self.continual_replay_buffer is not None:
            samples, _, _ = self.replay_buffer.get_batch(global_step=train_step, num_samples=50000)
            states = samples['states0']
            actions = samples['actions']
            new_states = samples['states1']
            dones = samples['terminals1']
            rewards = samples['rewards']
            for s, a, r, ss, d in zip(states, actions, rewards, new_states, dones):
                self.continual_replay_buffer.add(s, a, r, ss, d)
        parallel_envs.close()
        return env_evals_r, self.continual_replay_buffer

    def test(self, model_path):
        self.model.load(model_path)
        while True:
            obs, done = self.test_env.reset(), False
            while not done:
                action = self.model.action(obs.astype(np.float32))
                obs, reward, done, info = self.test_env.step(action)
                self.test_env.render()
