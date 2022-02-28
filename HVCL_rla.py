import tensorflow_addons as tfa
from typing import Sequence
from numbers import Number
from HVCL_rl_utils import *

class SAC:
    def __init__(self, largest_obs_dim, largest_act_dim, discount, polyak_coef, lr,
                 n_hidden_units, save_dir, task, total_episodes, n_experts=1, k=1, vmoe=True, m_rl=True,
                 exp_beta=1.0, gate_beta=1.0):

        self.largest_obs_dim = largest_obs_dim
        self.largest_act_dim = largest_act_dim
        self.n_hidden_units = n_hidden_units
        self.discount = discount
        self.polyak_coef = polyak_coef
        self.lr = lr
        self.save_dir = save_dir
        self.gamma = discount
        self.reward_scale = 1.0
        self.exp_beta = exp_beta
        self.gate_beta = gate_beta
        self.vmoe = vmoe
        self.n_experts = n_experts
        self.k = k
        self.task = task
        self.munchausen_rl = m_rl
        self.munch_alpha = 0.9
        self.munch_tau = 0.03
        self.munch_lo = -1
        self.total_episodes = total_episodes

    def reset_state(self, actions_dim, obs_dim, n_actions, act_lim, env):
        if hasattr(self, 'actor_network'):
            del self.actor_network
            del self.actor_optimizer
            del self.Qs
            del self.Q_targets
            del self.Q_optimizers
            del self.alpha_optimizer

        self.act_lim = act_lim
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.actions_dim = actions_dim
        self.env = env
        self.actor_exp_beta = 1e-5 if self.task == 0 else self.exp_beta
        self.actor_gate_beta = 1e-5 if self.task == 0 else self.gate_beta
        self.qnet_exp_beta = 1e-5 if self.task == 0 else self.exp_beta
        self.qnet_gate_beta = 1e-5 if self.task == 0 else self.gate_beta
        # alpha optimizer
        self.alpha_lr = self.lr
        self.target_entropy = -np.prod(self.largest_act_dim)
        self.log_alpha = tf.Variable(0.0)
        self.alpha = tfp.util.DeferredTensor(self.log_alpha, tf.exp)

        self.alpha_optimizer = tfa.optimizers.LazyAdam(self.alpha_lr, name='alpha_optimizer')
        ### Creating networks and optimizers ###
        # Policy network
        # action_output are the squashed actions and action_original those straight from the normal distribution
        logprob_epsilon = 1e-6  # For numerical stability when computing tf.log
        self.actor_network = ActorNetwork(self.n_hidden_units, self.largest_act_dim,
                                          logprob_epsilon,
                                          vmoe=self.vmoe, n_experts=self.n_experts,
                                          expert_beta=self.actor_exp_beta,
                                          gating_beta= self.actor_gate_beta,
                                          inputdim=self.largest_obs_dim,
                                          k=self.k)

        # 2 Soft q-functions networks + targets
        self.softq_network = SoftQNetwork(self.n_hidden_units, qi="source_0", vmoe=self.vmoe,
                                          n_experts=self.n_experts,
                                          expert_beta=self.qnet_exp_beta,
                                          gating_beta=self.qnet_gate_beta,
                                          inputdim=self.largest_obs_dim + self.largest_act_dim,
                                          k=self.k)
        self.softq_target_network = SoftQNetwork(self.n_hidden_units, qi="target_0", vmoe=self.vmoe,
                                                 n_experts=self.n_experts,
                                                 expert_beta=self.qnet_exp_beta,
                                                 gating_beta=self.qnet_gate_beta,
                                                 inputdim=self.largest_obs_dim + self.largest_act_dim,
                                                 k=self.k)

        self.softq_network2 = SoftQNetwork(self.n_hidden_units, qi="source_1", vmoe=self.vmoe,
                                           n_experts=self.n_experts, expert_beta=0.0,
                                           gating_beta=0.0,
                                           inputdim=self.largest_obs_dim + self.largest_act_dim,
                                           k=self.k)
        self.softq_target_network2 = SoftQNetwork(self.n_hidden_units, qi="target_1",
                                                  vmoe=self.vmoe,
                                                  n_experts=self.n_experts, expert_beta=0.0,
                                                  gating_beta=0.0,
                                                  inputdim=self.largest_obs_dim + self.largest_act_dim,
                                                  k=self.k)

        # input1 = np.zeros(shape=(1, self.largest_obs_dim), dtype="float32")
        # input2 = np.zeros(shape=(1, self.largest_act_dim), dtype="float32")
        #
        # self.actor_network(input1)
        # self.softq_network(input1, input2)
        # self.softq_target_network(input1, input2)
        # self.softq_network2(input1, input2)
        # self.softq_target_network2(input1, input2)
        # print(self.softq_network.softq.summary())
        # print(self.softq_target_network.softq.summary())
        # print(self.softq_network2.softq.summary())
        # print(self.softq_target_network2.softq.summary())

        # Optimizers for the networks
        self.softq_optimizer = tfa.optimizers.LazyAdam(learning_rate=self.lr)
        self.softq_optimizer2 = tfa.optimizers.LazyAdam(learning_rate=self.lr)
        self.actor_optimizer = tfa.optimizers.LazyAdam(learning_rate=self.lr)

        self.Qs = [self.softq_network, self.softq_network2]
        self.Q_targets = [self.softq_target_network, self.softq_target_network2]
        self.Q_optimizers = [self.softq_optimizer, self.softq_optimizer2]
        self._update_target(tau=0.0)

        # for Q, Qtarget in zip(self.Qs, self.Q_targets):
        #     source_w = Q.trainable_variables
        #     target_w = Qtarget.trainable_variables
        #     for sw, tw in zip(source_w, target_w):
        #         tf.debugging.assert_equal(sw, tw)
        #
        # pred_q2 = self.softq_network(input1, input2)
        # target_pred_q2 = self.softq_target_network(input1, input2)
        # tf.debugging.assert_equal(pred_q2, target_pred_q2)
        #
        # pred_q2 = self.softq_network2(input1, input2)
        # target_pred_q2 = self.softq_target_network2(input1, input2)
        # tf.debugging.assert_equal(pred_q2, target_pred_q2)

        self.updates_performed = 0
        self.frange_cycle_sigmoid(start=0.0, stop=1.0, n_epoch=self.total_episodes)

    def softq_value(self, states: np.ndarray, actions: np.ndarray):
        return self.softq_network(states, actions)

    def softq_value2(self, states: np.ndarray, actions: np.ndarray):
        return self.softq_network2(states, actions)

    def actions(self, states: np.ndarray) -> np.ndarray:
        """Get the actions for a batch of states."""
        if len(np.shape(states)) == 1:
            states = states[None, :]
        return self.actor_network(states)[0]

    def action(self, state: np.ndarray) -> np.ndarray:
        """Get the action for a single state."""
        return self.actor_network(state[None, :])[0][0]

    def step(self, obs):
        return self.actor_network(obs)[0]

    def get_models(self):
        return [self.actor_network, self.softq_network, self.softq_network2, self.softq_target_network,
                self.softq_target_network2]

    @tf.function(experimental_relax_shapes=True)
    def _update_alpha(self, observations):
        if not isinstance(self.target_entropy, Number):
            return 0.0

        actions, log_pis, _, _ = self.actor_network(observations)
        with tf.GradientTape() as tape:
            alpha_losses = -1.0 * (
                    self.alpha * tf.stop_gradient(log_pis + self.target_entropy))
            alpha_loss = tf.nn.compute_average_loss(alpha_losses)

        alpha_gradients = tape.gradient(alpha_loss, [self.log_alpha])
        self.alpha_optimizer.apply_gradients(zip(
            alpha_gradients, [self.log_alpha]))

        return alpha_losses

    @tf.function(experimental_relax_shapes=True)
    def td_targets(self, rewards, discounts, next_values):
        return rewards + discounts * next_values

    def frange_cycle_sigmoid(self, start, stop, n_epoch, n_cycle=4, ratio=0.9):
        L = np.ones(n_epoch)
        # period = n_epoch / n_cycle
        # step = (stop - start) / (period * ratio)  # step is in [0,1]
        #
        # # transform into [-6, 6] for plots: v*12.-6.
        #
        # for c in range(n_cycle):
        #
        #     v, i = start, 0
        #     while v <= stop:
        #         L[int(i + c * period)] = 1.0 / (1.0 + np.exp(- (v * 12. - 6.)))
        #         v += step
        #         i += 1

        self.beta_schedule = L.astype(dtype="float32")

    # @tf.function(experimental_relax_shapes=True)
    def copy_weights(self, old_models, num_train_samples=1):
        for old_model, new_model in zip(old_models, [self.actor_network, self.softq_network, self.softq_network2,
                                                     self.softq_target_network, self.softq_target_network2]):
            if old_model is not None:
                for new_layer, old_layer in zip(new_model.model_layers.layers, old_model.model_layers.layers):
                    test_weights = []
                    old_weights = old_layer.get_weights()
                    new_weights = new_layer.get_weights()
                    old_weight_obj = old_layer.weights
                    new_weight_obj = new_layer.weights
                    prior_weights = [[] for _ in range(20)]
                    # print("looking at layer %s out of %d layers" % (old_layer.name, len(old_model.layers)))
                    for wobj, weight in zip(old_weight_obj, old_weights):
                        # experts
                        if "post_expert_mu_kernel" in wobj.name:
                            prior_weights[0] = weight
                        elif "post_expert_rho_kernel" in wobj.name:
                            prior_weights[1] = weight
                        elif "post_expert_bias" in wobj.name:
                            prior_weights[2] = weight
                        # gating
                        elif "post_gating_kernel" in wobj.name:
                            prior_weights[3] = weight
                        elif "post_gating_bias" in wobj.name:
                            prior_weights[4] = weight
                        elif "post_gating_out_kernel" in wobj.name:
                            prior_weights[5] = weight
                        elif "post_gating_out_bias" in wobj.name:
                            prior_weights[6] = weight
                        # batchnorms
                        elif "batch_norm" in wobj.name:
                            if "gamma" in wobj.name:
                                prior_weights[7] = weight
                            elif "beta" in wobj.name:
                                prior_weights[8] = weight
                            elif "moving_mean" in wobj.name:
                                prior_weights[9] = weight
                            elif "moving_variance" in wobj.name:
                                prior_weights[10] = weight
                        # transpose convs
                        elif "transpose" in wobj.name:
                            if "kernel" in wobj.name:
                                prior_weights[11] = weight
                            elif "bias" in wobj.name:
                                prior_weights[12] = weight
                        elif "layer_norm" in wobj.name:
                            if "gamma" in wobj.name:
                                prior_weights[13] = weight
                            elif "beta" in wobj.name:
                                prior_weights[14] = weight

                    updated_new_weights = []
                    for wobj, weight in zip(new_weight_obj, new_weights):
                        print(wobj.name)
                        # experts
                        if "expert_mu_kernel" in wobj.name:
                            updated_new_weights.append(prior_weights[0])
                        elif "expert_rho_kernel" in wobj.name:
                            updated_new_weights.append(prior_weights[1])
                        elif "expert_bias" in wobj.name:
                            updated_new_weights.append(prior_weights[2])
                        # gating
                        elif "gating_kernel" in wobj.name:
                            updated_new_weights.append(prior_weights[3])
                        elif "gating_bias" in wobj.name:
                            updated_new_weights.append(prior_weights[4])
                        elif "gating_out_kernel" in wobj.name:
                            updated_new_weights.append(prior_weights[5])
                        elif "gating_out_bias" in wobj.name:
                            updated_new_weights.append(prior_weights[6])
                        # batchnorms
                        elif "batch_norm" in wobj.name:
                            if "gamma" in wobj.name:
                                updated_new_weights.append(prior_weights[7])
                            elif "beta" in wobj.name:
                                updated_new_weights.append(prior_weights[8])
                            elif "moving_mean" in wobj.name:
                                updated_new_weights.append(prior_weights[9])
                            elif "moving_variance" in wobj.name:
                                updated_new_weights.append(prior_weights[10])
                        # transpose convs
                        elif "transpose" in wobj.name:
                            if "kernel" in wobj.name:
                                updated_new_weights.append(prior_weights[11])
                            elif "bias" in wobj.name:
                                updated_new_weights.append(prior_weights[12])
                        elif "layer_norm" in wobj.name:
                            if "gamma" in wobj.name:
                                updated_new_weights.append(prior_weights[13])
                            elif "beta" in wobj.name:
                                updated_new_weights.append(prior_weights[14])
                        else:
                            updated_new_weights.append(weight)
                            print("could not find %s and copied" % wobj.name)
                    if len(updated_new_weights):
                        new_layer.set_weights(updated_new_weights)
        # return new_models

    @tf.function(experimental_relax_shapes=True)
    def _update_target(self, tau):
        for Q, Q_target in zip(self.Qs, self.Q_targets):
            for source_weight, target_weight in zip(
                    Q.trainable_variables, Q_target.trainable_variables):
                target_weight.assign(
                    tau * target_weight + (1.0 - tau) * source_weight)

    @tf.function(experimental_relax_shapes=True)
    def compute_Q_targets(self, next_Q_values,
                          next_log_pis,
                          rewards,
                          terminals,
                          discount,
                          entropy_scale,
                          reward_scale):
        next_values = next_Q_values - entropy_scale * next_log_pis
        terminals = tf.cast(terminals, next_values.dtype)

        Q_targets = self.td_targets(
            rewards=reward_scale * rewards,
            discounts=discount,
            next_values=(1.0 - terminals) * next_values)
        return Q_targets

    @tf.function(experimental_relax_shapes=True)
    def _compute_Q_targets(self, batch):
        next_observations = batch['states1']
        rewards = batch['rewards']
        terminals = batch['terminals1']
        observations = batch['states0']
        if self.munchausen_rl:
            actions = batch['actions']
            _, _, mu_m, log_std_m = self.actor_network(observations)
            log_pi_a = self.munch_tau * self.actor_network.gaussian_likelihood(mu=mu_m, input=actions, log_std=log_std_m)
            print("logpi", log_pi_a.shape)
            print("rewards", rewards.shape)
            assert log_pi_a.shape == rewards.shape
            munchausen_reward = (rewards + self.munch_alpha * tf.clip_by_value(log_pi_a,
                                                                               clip_value_min=self.munch_lo,
                                                                               clip_value_max=0))
            assert munchausen_reward.shape == rewards.shape
            rewards = munchausen_reward

        entropy_scale = tf.convert_to_tensor(self.alpha)
        reward_scale = tf.convert_to_tensor(self.reward_scale)
        discount = tf.convert_to_tensor(self.gamma)

        next_actions, next_log_pis, _, _ = self.actor_network(next_observations)
        next_Qs_values = []
        for Q in self.Q_targets:
            next_Qs_values.append(Q(next_observations, next_actions))
        next_Qs_values = tf.concat(next_Qs_values, axis=-1)
        next_Qs_values = tf.math.reduce_min(next_Qs_values, axis=-1)

        Q_targets = self.compute_Q_targets(
            next_Qs_values,
            next_log_pis,
            rewards,
            terminals,
            discount,
            entropy_scale,
            reward_scale)
        tf.debugging.assert_all_finite(Q_targets, "q targets not finite")
        return tf.stop_gradient(Q_targets)

    def measure_graph_size(self, f, *args):
        g = f.get_concrete_function(*args).graph
        print("{}({}) contains {} nodes in its graph".format(
            f.__name__, ', '.join(map(str, args)), len(g.as_graph_def().node)))


    @tf.function
    def train(self, sample, batch_size, experience_weights=None, old_sample=None, old_experience_weights=None, task=1):
        observations = sample["states0"]
        actions = sample["actions"]

        # Computing target for q-functions
        softq_targets = self._compute_Q_targets(sample)
        softq_targets = tf.reshape(softq_targets, [batch_size, 1])
        tf.debugging.assert_all_finite(softq_targets, "q values not finite")
        q_losses = []
        avg_td_errors = tf.zeros_like(softq_targets)
        avg_old_td_errors = tf.zeros_like(softq_targets)
        avg_qmodel_losses = tf.zeros(shape=())
        if task >= 1 and old_sample is not None:
            new_weight = 0.5
            old_weight = 0.5
        else:
            new_weight = 1.0
            old_weight = 1.0
        for Q, optimizer in zip(self.Qs, self.Q_optimizers):
            with tf.GradientTape() as tape:
                Q_values, q_model_losses = Q.loss_call(observations, actions)
                avg_qmodel_losses = avg_qmodel_losses + tf.multiply(0.5, q_model_losses)
                pred_Q_losses = tf.keras.losses.huber(y_true=softq_targets, y_pred=Q_values)
                Q_losses = tf.nn.compute_average_loss(pred_Q_losses, sample_weight=experience_weights)
                Q_loss = Q_losses
                avg_td_errors = avg_td_errors + 0.5 * tf.abs(softq_targets - Q_values)
                if self.vmoe:
                    Q_loss += (self.beta_schedule[self.updates_performed] * q_model_losses)
                q_losses.append(Q_loss)

                if old_sample is not None:
                    old_observations = old_sample["states0"]
                    old_actions = old_sample["actions"]
                    old_softq_targets = self._compute_Q_targets(old_sample)
                    old_softq_targets = tf.reshape(old_softq_targets, [batch_size, 1])
                    tf.debugging.assert_all_finite(old_softq_targets, "old q values not finite")
                    old_Q_values, old_q_model_losses = Q.loss_call(old_observations, old_actions)
                    pred_old_Q_losses = tf.keras.losses.huber(y_true=old_softq_targets, y_pred=old_Q_values)
                    old_Q_losses = tf.nn.compute_average_loss(pred_old_Q_losses, sample_weight=old_experience_weights)
                    old_Q_loss = old_Q_losses
                    avg_old_td_errors = avg_old_td_errors + 0.5 * tf.abs(old_softq_targets - old_Q_values)
                    if self.vmoe:
                        old_Q_loss += (self.beta_schedule[self.updates_performed] * old_q_model_losses)
                    total_Q_loss = new_weight * Q_loss + old_weight * old_Q_loss
                else:
                    total_Q_loss = Q_loss
                    pred_old_Q_losses = None
            total_gradients = tape.gradient(total_Q_loss, Q.trainable_variables)
            [tf.debugging.assert_all_finite(g, "q fun grads not finite") for g in total_gradients]
            total_gradients, _ = tf.clip_by_global_norm(total_gradients, 1.0)
            optimizer.apply_gradients(zip(total_gradients, Q.trainable_variables))
        if old_sample is None:
            avg_old_td_errors = None
        # Gradient ascent for the policy (actor)
        entropy_scale = tf.convert_to_tensor(self.alpha)
        with tf.GradientTape() as actor_tape:
            actions, log_pis, mean, std, actor_model_loss = self.actor_network.loss_call(observations)
            Qs_log_targets = []
            for Q in self.Qs:
                Qs_log_targets.append(Q(observations, actions))
            Qs_log_targets = tf.concat(Qs_log_targets, axis=-1)
            Qs_log_targets = tf.math.reduce_min(Qs_log_targets, axis=-1)
            actor_loss = (entropy_scale * log_pis) - Qs_log_targets
            actor_loss = tf.nn.compute_average_loss(actor_loss, sample_weight=experience_weights)
            actor_loss = actor_loss
            if self.vmoe:
                actor_loss += (self.beta_schedule[self.updates_performed] * actor_model_loss)

            if old_sample is not None:
                old_actions, old_log_pis, old_mean, old_std, actor_model_loss = self.actor_network.loss_call(old_observations)
                old_Qs_log_targets = []
                for Q in self.Qs:
                    old_Qs_log_targets.append(Q(old_observations, old_actions))
                old_Qs_log_targets = tf.concat(old_Qs_log_targets, axis=-1)
                old_Qs_log_targets = tf.math.reduce_min(old_Qs_log_targets, axis=-1)
                old_actor_loss = (entropy_scale * old_log_pis) - old_Qs_log_targets
                old_actor_loss = tf.nn.compute_average_loss(old_actor_loss, sample_weight=old_experience_weights)
                old_actor_loss = old_actor_loss
                if self.vmoe:
                    old_actor_loss += (self.beta_schedule[self.updates_performed] * actor_model_loss)
                total_actor_loss = new_weight * actor_loss + old_weight * old_actor_loss
            else:
                total_actor_loss = actor_loss
        total_actor_gradients = actor_tape.gradient(total_actor_loss, self.actor_network.trainable_weights)
        [tf.debugging.assert_all_finite(g, "actor grads not finite") for g in total_actor_gradients]
        total_actor_gradients, _ = tf.clip_by_global_norm(total_actor_gradients, 1.0)
        # Minimize gradients wrt weights
        self.actor_optimizer.apply_gradients(zip(total_actor_gradients, self.actor_network.trainable_weights))

        # Update the weights of the soft q-function target networks
        self._update_target(tau=self.polyak_coef)

        self._update_alpha(observations=observations)

        return tf.reduce_mean(total_Q_loss), tf.reduce_mean(total_actor_loss), tf.reduce_mean(log_pis), tf.reduce_mean(
            softq_targets), tf.reduce_mean(Q_values), tf.reduce_mean(Qs_log_targets), tf.reduce_mean(actor_model_loss), \
               tf.reduce_mean(avg_qmodel_losses), self.alpha, mean, std, avg_td_errors, avg_old_td_errors

    # def save(self):
    #     self.actor_network.save_weights(self.save_dir + "/actor.ckpt")
    #     print("Model saved!")
    #
    # def load(self, filepath):
    #     self.actor_network.load_weights(filepath + "/actor.ckpt")
    #     print("Model loaded!")
