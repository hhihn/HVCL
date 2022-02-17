#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import activations, initializers, regularizers, constraints
from tensorflow.keras.initializers import RandomUniform, glorot_uniform
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.ops import math_ops, array_ops, state_ops
from tensorflow.python.keras.utils import tf_utils
from tensorflow.keras.layers import concatenate, Reshape
from tensorflow.python.framework import ops
import tensorflow_probability as tfp
from SparseDispatcher import SparseDispatcher
from initializer import *
from tensorflow.python.keras import backend as K

ds = tfp.distributions
#
class BayesianDenseMoE(Layer):
    """Mixture-of-experts layer.
    Implements: y = sum_{k=1}^K g(v_k * x) f(W_k * x)
        # Arguments
        units: Positive integer, dimensionality of the output space.
        n_experts: Positive integer, number of experts (K).
        expert_activation: Activation function for the expert model (f).
        gating_activation: Activation function for the gating model (g).
        use_expert_bias: Boolean, whether to use biases in the expert model.
        use_gating_bias: Boolean, whether to use biases in the gating model.
        expert_bias_initializer: Initializer for the expert biases.
        gating_bias_initializer: Initializer fot the gating biases.
        expert_kernel_regularizer: Regularizer for the expert model weights.
        gating_kernel_regularizer: Regularizer for the gating model weights.
        expert_bias_regularizer: Regularizer for the expert model biases.
        gating_bias_regularizer: Regularizer for the gating model biases.
        expert_kernel_constraint: Constraints for the expert model weights.
        gating_kernel_constraint: Constraints for the gating model weights.
        expert_bias_constraint: Constraints for the expert model biases.
        gating_bias_constraint: Constraints for the gating model biases.
        activity_regularizer: Activity regularizer.
    # Input shape
        nD tensor with shape: (batch_size, ..., input_dim).
        The most common situation would be a 2D input with shape (batch_size, input_dim).
    # Output shape
        nD tensor with shape: (batch_size, ..., units).
        For example, for a 2D input with shape (batch_size, input_dim), the output would have shape (batch_size, units).
    """

    def __init__(self, units,
                 n_experts,
                 expert_activation=None,
                 gating_activation='softmax',
                 use_expert_bias=True,
                 use_gating_bias=True,
                 expert_bias_initializer='zeros',
                 gating_bias_initializer='zeros',
                 expert_kernel_regularizer=None,  # tf.keras.regularizers.l2(l=1e-6),
                 gating_kernel_regularizer=None,  # tf.keras.regularizers.l2(l=1e-6),
                 attn_kernel_regularizer=None,  # tf.keras.regularizers.l1(l=0.05),
                 gating_noise_kernel_regularizer=None,  # tf.keras.regularizers.l1(l=0.05),
                 expert_bias_regularizer=None,
                 gating_bias_regularizer=None,
                 expert_kernel_constraint=None,
                 gating_kernel_constraint=None,
                 expert_bias_constraint=None,
                 gating_bias_constraint=None,
                 activity_regularizer=None,
                 gating_prior_momentum=0.99,
                 gating_beta=0.01,
                 expert_beta=0.01,
                 kernel_width=10.0,
                 gating_entropy_beta=1.0,
                 k=1,
                 n_monte_carlo=100,
                 attn=True,
                 kl_div_fun=None,
                 entropy_fun=None,
                 **kwargs):

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(BayesianDenseMoE, self).__init__(**kwargs)

        self.gating_beta = gating_beta
        self.gating_entropy_beta = gating_entropy_beta
        self.expert_beta = expert_beta
        self.units = units
        self.gating_units = int(units * 1.5)
        self.n_experts = n_experts
        self.intermediate_gating_units = units
        self.kernel_width = kernel_width
        # self.attn = attn
        self.kl_div_fun = kl_div_fun
        if type(k) is str and k == 'all':
            self.k = n_experts
        else:
            self.k = k
        self.n_monte_carlo = n_monte_carlo
        self.expert_kernel_initializer = HeUniformExpertInitializer(numexp=self.n_experts)
        rho_init = -2.6 #self._softplus_inverse(0.006)
        self.rho_initializer = tf.keras.initializers.random_normal(mean=rho_init, stddev=0.01)

        self.expert_activation = activations.get(expert_activation)
        self.gating_activation = activations.get(gating_activation)
        self.entropy_fun = entropy_fun
        self.use_expert_bias = use_expert_bias
        self.use_gating_bias = use_gating_bias

        self.expert_bias_initializer = initializers.get(expert_bias_initializer)
        self.gating_bias_initializer = initializers.get(gating_bias_initializer)

        self.expert_kernel_regularizer = expert_kernel_regularizer
        self.gating_kernel_regularizer = gating_kernel_regularizer
        self.attn_kernel_regularizer = attn_kernel_regularizer
        self.gating_noise_kernel_regularizer = gating_noise_kernel_regularizer

        self.expert_bias_regularizer = regularizers.get(expert_bias_regularizer)
        self.gating_bias_regularizer = regularizers.get(gating_bias_regularizer)

        self.expert_kernel_constraint = constraints.get(expert_kernel_constraint)
        self.gating_kernel_constraint = constraints.get(gating_kernel_constraint)

        self.expert_bias_constraint = constraints.get(expert_bias_constraint)
        self.gating_bias_constraint = constraints.get(gating_bias_constraint)

        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.gating_prior_momentum = gating_prior_momentum

        self.gating_reduction_axis = -1
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.input_dim = input_shape[-1]
        self.sel_input_dim = input_shape[-1]
        self.data_dim = self.input_dim
        self.expert_mu_kernel = self.add_weight(shape=(self.data_dim, self.units, self.n_experts),
                                                initializer=self.expert_kernel_initializer,
                                                name='post_expert_mu_kernel',
                                                regularizer=self.expert_kernel_regularizer,
                                                constraint=self.expert_kernel_constraint)

        self.expert_rho_kernel = self.add_weight(shape=(self.units, self.n_experts),
                                                 initializer=self.rho_initializer,
                                                 name='post_expert_rho_kernel',
                                                 regularizer=self.expert_kernel_regularizer,
                                                 constraint=self.expert_kernel_constraint)

        self.prior_expert_mu_kernel = self.add_weight(shape=(self.data_dim, self.units, self.n_experts),
                                                      initializer='zeros',
                                                      name='prior_expert_mu_kernel',
                                                      regularizer=self.expert_kernel_regularizer,
                                                      constraint=self.expert_kernel_constraint,
                                                      trainable=False)

        self.prior_expert_rho_kernel = self.add_weight(shape=(self.units, self.n_experts),
                                                       initializer=self.rho_initializer,
                                                       name='prior_expert_rho_kernel',
                                                       regularizer=self.expert_kernel_regularizer,
                                                       constraint=self.expert_kernel_constraint,
                                                       trainable=False)

        if self.n_experts > 1:

            self.gating_kernel = self.add_weight(shape=(self.sel_input_dim, self.n_experts),
                                                 initializer="he_uniform",
                                                 name='post_gating_kernel',
                                                 regularizer=self.gating_kernel_regularizer,
                                                 constraint=self.gating_kernel_constraint)

            self.prior_gating_kernel = self.add_weight(shape=(self.sel_input_dim, self.n_experts),
                                                       initializer="zeros",
                                                       name='prior_gating_kernel',
                                                       regularizer=self.gating_kernel_regularizer,
                                                       constraint=self.gating_kernel_constraint,
                                                       trainable=False)
        else:
            self.gating_noise = None
            self.gating_kernel = None
            self.prior_gating_kernel = None
            self.prior_gating = None
            self.old_prior_gating = None

        if self.use_expert_bias:
            self.expert_bias = self.add_weight(shape=(self.units, self.n_experts),
                                               initializer=self.expert_bias_initializer,
                                               name='post_expert_bias',
                                               regularizer=self.expert_bias_regularizer,
                                               constraint=self.expert_bias_constraint)
            self.prior_expert_bias = self.add_weight(shape=(self.units, self.n_experts),
                                                     initializer=self.expert_bias_initializer,
                                                     name='prior_expert_bias',
                                                     regularizer=self.expert_bias_regularizer,
                                                     constraint=self.expert_bias_constraint,
                                                     trainable=False)
        else:
            self.expert_bias = None
            self.prior_expert_bias = None

        if self.use_gating_bias and self.n_experts > 1:
            self.gating_bias = self.add_weight(shape=(self.n_experts,),
                                               initializer=self.gating_bias_initializer,
                                               name='post_gating_bias',
                                               regularizer=self.gating_bias_regularizer,
                                               constraint=self.gating_bias_constraint)
            self.prior_gating_bias = self.add_weight(shape=(self.n_experts,),
                                                     initializer=self.gating_bias_initializer,
                                                     name='prior_gating_bias',
                                                     regularizer=self.gating_bias_regularizer,
                                                     constraint=self.gating_bias_constraint,
                                                     trainable=False)
        else:
            self.gating_bias = None
            self.prior_gating_bias = None

        self.kl_weight = self.add_weight(shape=(),
                                         initializer=tf.initializers.constant(self.expert_beta),
                                         name='kl_weights',
                                         regularizer=self.gating_bias_regularizer,
                                         constraint=self.gating_bias_constraint,
                                         trainable=False)
        self.gating_kl_weight = self.add_weight(shape=(),
                                                initializer=tf.initializers.constant(self.gating_beta),
                                                name='gating_kl_weight',
                                                regularizer=self.gating_bias_regularizer,
                                                constraint=self.gating_bias_constraint,
                                                trainable=False)

        self.input_spec = InputSpec(min_ndim=2, axes={-1: self.input_dim})
        self.built = True

    def load_balanced_loss(self, router_probs, expert_mask):
        num_experts = tf.shape(expert_mask)[-1]
        density = tf.reduce_mean(expert_mask, axis=0)
        density_proxy = tf.reduce_mean(router_probs, axis=0)
        loss = tf.reduce_mean(density_proxy * density) * tf.cast(
            (num_experts ** 2), tf.dtypes.float32
        )
        return loss

    def call(self, inputs, **kwargs):
        if self.n_experts > 1:
            gating_logits, activated_gating_outputs, \
            activated_prior_gating_outputs, gating_cond_entropy = self.apply_gating_kernels(inputs)

            # gating entropy penalty
            batch_gating_prior = ds.Categorical(probs=tf.reduce_mean(activated_gating_outputs, axis=0, keepdims=True))
            batch_activated_gating_entropy = tf.reduce_mean(self.entropy_fun(batch_gating_prior))
            self.add_loss(self.gating_entropy_beta * batch_activated_gating_entropy)
            self.add_loss(self.gating_entropy_beta * gating_cond_entropy)

            # gating KL div penalty
            gating_posterior_dist = tfp.distributions.Categorical(probs=activated_gating_outputs)
            gating_prior_dist = tfp.distributions.Categorical(probs=activated_prior_gating_outputs)
            kl_loss = tf.reduce_mean(self.kl_div_fun(gating_posterior_dist, gating_prior_dist), axis=0)
            self.add_loss(self.gating_kl_weight * kl_loss)
            #
            gates = self.top_k_gating(gating_logits)
            # balance_loss = self.load_balanced_loss(router_probs=activated_gating_outputs, expert_mask=gates)
            # self.add_loss(self.gating_entropy_beta * balance_loss)
            # expert diversity bonus
            dkl_mat = [[tf.ones(shape=()) for _ in range(self.n_experts)] for _ in range(self.n_experts)]
            kernel_width = self.kernel_width
            for iexp in range(self.n_experts):
                for jexp in range(iexp, self.n_experts):
                    if iexp != jexp:
                        dkl_mat[iexp][jexp] = self.wasserstein_dist(iexp, jexp)
                        dkl_mat[iexp][jexp] = tf.exp(-self.wasserstein_dist(iexp, jexp) / kernel_width)
                        dkl_mat[jexp][iexp] = dkl_mat[iexp][jexp]
                    else:
                        dkl_mat[iexp][jexp] = tf.ones(shape=())

            exp_diversity = tf.linalg.det(dkl_mat)
            div_bonus = -exp_diversity
            self.add_loss(self.gating_entropy_beta * div_bonus)
        else:
            gates = tf.ones(shape=(tf.shape(inputs)[0], 1))
        dispatcher = SparseDispatcher(self.n_experts, gates)
        e_inputs = dispatcher.dispatch(inputs)
        dispatcher.expert_to_gates()

        expert_outputs, approx_expert_dkl = self.apply_kernels(expert_input_tensors=e_inputs)
        self.add_loss(self.kl_weight * tf.reduce_mean(approx_expert_dkl))
        # Combine Experts
        if self.n_experts > 1:
            output = dispatcher.combine(expert_outputs)
        else:
            output = expert_outputs[0]
        return output  # , activated_gating_outputs

    def norm(self, x):
        n = tf.sqrt(tf.reduce_sum(tf.square(x)) + 1e-4)
        return n

    def wasserstein_dist(self, i, j):
        # Hellinger Distance between diagonal Covariance Matrices
        cov_i = tf.math.sqrt(tf.nn.softplus(self.expert_rho_kernel[:, i]) + 1e-6)
        cov_j = tf.math.sqrt(tf.nn.softplus(self.expert_rho_kernel[:, j]) + 1e-6)
        cov_trace = self.norm(cov_i - cov_j)
        # mean over units
        diff = self.expert_mu_kernel[:, :, i] - self.expert_mu_kernel[:, :, j]
        mu_norm = tf.square(tf.norm(diff))
        w = mu_norm + cov_trace
        return w

    def entropy(self, exp):
        cov_post = tf.nn.softplus(self.expert_rho_kernel[:, exp])
        mu_post = self.expert_mu_kernel[:, :, exp]
        expert_kernel_dist = tfp.distributions.MultivariateNormalDiag(mu_post, cov_post)
        return tf.reduce_sum(self.entropy_fun(expert_kernel_dist))

    def apply_attention(self, data):
        query = K.dot(data, self.gating_attn)
        key = K.dot(data, self.gating_attn)
        value = key
        scores = math_ops.matmul(query, key, transpose_b=True)
        scores *= self.scale
        weights = tf.nn.softmax(scores)
        self_attn = math_ops.matmul(weights, value)
        query_encoding = tf.reduce_mean(query, axis=-1, keepdims=True)
        query_value_attention = tf.reduce_mean(self_attn, axis=-1, keepdims=True)
        self_attn = concatenate(axis=-1, inputs=[query_encoding, query_value_attention])
        return self_attn

    def apply_gating_kernels(self, inputs):

        gating_outputs = tf.matmul(inputs, self.gating_kernel)
        prior_gating_outputs = tf.matmul(inputs, self.prior_gating_kernel)

        if self.use_gating_bias:
            gating_outputs = K.bias_add(gating_outputs, self.gating_bias)
            prior_gating_outputs = K.bias_add(prior_gating_outputs, self.prior_gating_bias)

        activated_gating_outputs = self.gating_activation(gating_outputs)
        activated_prior_gating_outputs = self.gating_activation(prior_gating_outputs)
        gating_dist = ds.Categorical(probs=activated_gating_outputs)
        gating_entropy = tf.reduce_mean(self.entropy_fun(gating_dist))
        return gating_outputs, activated_gating_outputs, activated_prior_gating_outputs, gating_entropy,

    def approx_expert_dkl(self, exp, q=None):
        """
        Computes DKL between p and q
        :param exp: index of expert distribution
        :param q: if none, then q is exp's prior
        :return: DKL as tensor
        """
        cov_post = tf.nn.softplus(self.expert_rho_kernel[:, exp])
        mu_post = self.expert_mu_kernel[:, :, exp]

        if q is not None:
            exp = q
        cov_prior = tf.nn.softplus(self.prior_expert_rho_kernel[:, exp])
        mu_prior = self.prior_expert_mu_kernel[:, :, exp]
        expert_kernel_dist = tfp.distributions.MultivariateNormalDiag(mu_post, cov_post)
        expert_kernel_dist_prior = tfp.distributions.MultivariateNormalDiag(mu_prior, cov_prior)
        kl = self.kl_div_fun(expert_kernel_dist, expert_kernel_dist_prior)
        kl_loss = tf.reduce_mean(kl, axis=-1)
        return kl_loss

    def top_k_gating(self, gating_logits):
        top_logits, top_indices = self._my_top_k(gating_logits, min(self.k, self.n_experts))
        top_k_logits = tf.slice(top_logits, [0, 0], [-1, self.k])
        top_k_indices = tf.slice(top_indices, [0, 0], [-1, self.k])
        top_k_gates = tf.nn.softmax(top_k_logits)
        gates = self._rowwise_unsorted_segment_sum(top_k_gates, top_k_indices, self.n_experts)
        return gates

    def apply_kernels(self, expert_input_tensors):

        def random_rademacher(shape, dtype=tf.float64, seed=None):
            int_dtype = tf.int64 if tf.as_dtype(dtype) != tf.int32 else tf.int32
            random_bernoulli = tf.random.uniform(
                shape, minval=0, maxval=2, dtype=int_dtype, seed=seed)
            return tf.cast(2 * random_bernoulli - 1, dtype)

        expert_outputs = []
        expert_kls = []
        for i in range(self.n_experts):
            expert_input = expert_input_tensors[i]
            input_shape = tf.shape(expert_input)
            batch_shape = input_shape[:-1]
            posterior_affine_tensor = tfp.distributions.Normal(
                loc=tf.zeros_like(self.expert_mu_kernel[:, :, i]),
                scale=tf.nn.softplus(self.expert_rho_kernel[:, i])).sample()
            sign_input = random_rademacher(
                input_shape,
                dtype=expert_input.dtype)
            sign_output = random_rademacher(
                tf.concat([batch_shape,
                           tf.expand_dims(self.units, 0)], 0),
                dtype=expert_input.dtype)
            perturbed_inputs = tf.matmul(expert_input * sign_input, posterior_affine_tensor) * sign_output

            out = tf.matmul(expert_input, self.expert_mu_kernel[:, :, i])
            out += perturbed_inputs
            if self.use_expert_bias:
                out = out + self.expert_bias[:, i]
            if self.expert_activation is not None:
                out = self.expert_activation(out)
            expert_outputs.append(out)
            exp_kl = self.approx_expert_dkl(exp=i)
            expert_kls.append(exp_kl)
        return expert_outputs, expert_kls

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def _assign_moving_average(self, variable, value, momentum, inputs_size=None):
        with K.name_scope('AssignMovingAvg') as scope:
            with ops.colocate_with(variable):
                decay = 1.0 - momentum
                update_delta = (variable - math_ops.cast(value, variable.dtype)) * decay
                if inputs_size is not None:
                    update_delta = array_ops.where(inputs_size > 0, update_delta,
                                                   K.zeros_like(update_delta))
                return state_ops.assign_sub(variable, update_delta, name=scope)

    def get_config(self):
        config = {
            'units': self.units,
            'n_experts': self.n_experts,
            'expert_activation': activations.serialize(self.expert_activation),
            'gating_activation': activations.serialize(self.gating_activation),
            'use_expert_bias': self.use_expert_bias,
            'use_gating_bias': self.use_gating_bias,
            'expert_bias_initializer': initializers.serialize(self.expert_bias_initializer),
            'gating_bias_initializer': initializers.serialize(self.gating_bias_initializer),
            'expert_kernel_regularizer': regularizers.serialize(self.expert_kernel_regularizer),
            'gating_kernel_regularizer': regularizers.serialize(self.gating_kernel_regularizer),
            'expert_bias_regularizer': regularizers.serialize(self.expert_bias_regularizer),
            'gating_bias_regularizer': regularizers.serialize(self.gating_bias_regularizer),
            'expert_kernel_constraint': constraints.serialize(self.expert_kernel_constraint),
            'gating_kernel_constraint': constraints.serialize(self.gating_kernel_constraint),
            'expert_bias_constraint': constraints.serialize(self.expert_bias_constraint),
            'gating_bias_constraint': constraints.serialize(self.gating_bias_constraint),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'gating_beta': self.gating_beta,
            'expert_beta': self.expert_beta,
            'gating_entropy_beta': self.gating_entropy_beta,
            'k': self.k,
            'n_monte_carlo': self.n_monte_carlo,
        }
        base_config = super(BayesianDenseMoE, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _softplus_inverse(self, x):
        """Helper which computes the function inverse of `tf.nn.softplus`."""
        return np.log(np.exp(x - 1))

    def _my_top_k(self, x, k, soft=True):
        """GPU-compatible version of top-k that works for very small constant k.
        Calls argmax repeatedly.
        tf.nn.top_k is implemented for GPU, but the gradient, sparse_to_dense,
        seems not to be, so if we use tf.nn.top_k, then both the top_k and its
        gradient go on cpu.  Once this is not an issue, this function becomes
        obsolete and should be replaced by tf.nn.top_k.
        Args:
        x: a 2d Tensor.
        k: a small integer.
        soft: sample or use argmax
        Returns:
        values: a Tensor of shape [batch_size, k]
        indices: a int32 Tensor of shape [batch_size, k]
        """
        if k > 10:
            return tf.math.top_k(x, k)
        values = []
        indices = []
        depth = tf.shape(x)[1]
        for i in range(k):
            if not soft:
                idx = tf.argmax(x, 1)
                values.append(tf.reduce_max(x, 1))
            else:
                dist = ds.Categorical(logits=x)
                idx = dist.sample()
                values.append(dist.log_prob(idx))
            indices.append(idx)
            if i + 1 < k:
                x += tf.one_hot(idx, depth, -1e9)
        return tf.stack(values, axis=1), tf.cast(tf.stack(indices, axis=1), dtype=tf.int32)

    def _rowwise_unsorted_segment_sum(self, values, indices, n):
        """UnsortedSegmentSum on each row.
        Args:
        values: a `Tensor` with shape `[batch_size, k]`.
        indices: an integer `Tensor` with shape `[batch_size, k]`.
        n: an integer.
        Returns:
        A `Tensor` with the same type as `values` and shape `[batch_size, n]`.
        """
        batch, k = tf.unstack(tf.shape(indices), num=2)
        indices_flat = tf.reshape(indices, [-1]) + tf.cast(tf.divide(tf.range(batch * k), k), dtype=tf.int32) * n
        ret_flat = tf.math.unsorted_segment_sum(tf.reshape(values, [-1]), indices_flat, batch * n)
        return tf.reshape(ret_flat, [batch, n])

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values, k):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
          clean_values: a `Tensor` of shape [batch, n].
          noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
            normally distributed noise with standard deviation noise_stddev.
          noise_stddev: a `Tensor` of shape [batch, n], or None
          noisy_top_values: a `Tensor` of shape [batch, m].
             "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
          k: an integer.
        Returns:
          a `Tensor` of shape [batch, n].
        """
        batch = tf.shape(clean_values)[0]
        m = tf.shape(noisy_top_values)[1]
        top_values_flat = tf.reshape(noisy_top_values, [-1])
        # we want to compute the threshold that a particular value would have to
        # exceed in order to make the top k.  This computation differs depending
        # on whether the value is already in the top k.
        threshold_positions_if_in = tf.range(batch) * m + k
        threshold_if_in = tf.expand_dims(
            tf.gather(top_values_flat, threshold_positions_if_in), 1)
        is_in = tf.greater(noisy_values, threshold_if_in)
        if noise_stddev is None:
            return tf.cast(is_in, dtype=tf.float64)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = tf.expand_dims(
            tf.gather(top_values_flat, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        prob_if_in = self._normal_distribution_cdf(clean_values - threshold_if_in,
                                                   noise_stddev)
        prob_if_out = self._normal_distribution_cdf(clean_values - threshold_if_out,
                                                    noise_stddev)
        prob = tf.where(is_in, prob_if_in, prob_if_out)
        return prob

    def _normal_distribution_cdf(self, x, stddev):
        """Evaluates the CDF of the normal distribution.
      Normal distribution with mean 0 and standard deviation stddev,
      evaluated at x=x.
      input and output `Tensor`s have matching shapes.
      Args:
        x: a `Tensor`
        stddev: a `Tensor` with the same shape as `x`.
      Returns:
        a `Tensor` with the same shape as `x`.
      """
        return 0.5 * (1.0 + tf.math.erf(x / (tf.math.sqrt(2.0) * stddev + 1e-20)))

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
          x: a `Tensor`.
        Returns:
          a `Scalar`.
        """
        epsilon = 1e-10
        float_size = tf.cast(tf.size(x), dtype=tf.float64) + epsilon
        mean = tf.reduce_sum(x) / float_size
        variance = tf.reduce_sum(tf.math.squared_difference(x, mean)) / float_size
        return variance / (tf.square(mean) + epsilon)

