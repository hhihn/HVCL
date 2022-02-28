# -*- coding: utf-8 -*-
"""Convolutional MoE layers. The code here is based on the implementation of the standard convolutional layers in Keras.
"""
from tensorflow.python.keras.utils import conv_utils
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras import activations, initializers, regularizers, constraints
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.ops import math_ops, array_ops, state_ops
import tensorflow_probability as tfp
from tensorflow.python.layers import utils as tf_layers_util  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.ops import nn_ops, nn  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.framework import ops
from tensorflow.keras.layers import ELU, LeakyReLU, BatchNormalization, MaxPooling2D
from SparseDispatcher import SparseDispatcher
from tensorflow.keras.layers import SpatialDropout2D, SpatialDropout1D
from tensorflow.keras.regularizers import l2
from initializer import *
ds = tfp.distributions

class _ConvMoE(Layer):
    """Abstract nD convolution layer mixture of experts (private, used as implementation base).
    """

    def __init__(self, rank,
                 n_filters,
                 n_experts,
                 kernel_size,
                 strides=1,
                 padding='same',
                 data_format='channels_last',
                 dilation_rate=1,
                 expert_activation=None,
                 gating_activation=tf.nn.softmax,
                 use_expert_bias=True,
                 use_gating_bias=True,
                 expert_bias_initializer='zeros',
                 gating_bias_initializer='zeros',
                 expert_kernel_regularizer=None,
                 gating_kernel_regularizer=None,
                 expert_bias_regularizer=None,
                 gating_bias_regularizer=None,
                 expert_kernel_constraint=None,
                 gating_kernel_constraint=None,
                 expert_bias_constraint=None,
                 gating_bias_constraint=None,
                 activity_regularizer=None,
                 gating_noise_sigma=0.0001,
                 gating_prior_momentum=0.9995,
                 gating_beta=0.001,
                 expert_beta=0.001,
                 diversity_bonus=1.0,
                 k=1,
                 n_monte_carlo=10,
                 kernel_width=10,
                 kl_div_fun=None,
                 entropy_fun=None,
                 **kwargs):
        super(_ConvMoE, self).__init__(**kwargs)
        self.rank = rank
        self.n_filters = n_filters
        self.n_experts = n_experts
        if type(k) is str and k == 'all':
            self.k = n_experts
        else:
            self.k = min(n_experts, k)
        self.kernel_width = kernel_width
        self.n_monte_carlo = n_monte_carlo
        self.gating_prior_momentum = gating_prior_momentum
        self.gating_noise_sigma = gating_noise_sigma
        self.kl_div_fun = kl_div_fun
        self.gating_beta = gating_beta
        self.expert_beta = expert_beta
        self.entropy_fun = entropy_fun
        self.diversity_bonus = diversity_bonus
        self.n_total_filters = self.n_filters * self.n_experts
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')

        self.expert_kernel_initializer = HeUniformExpertInitializer(numexp=self.n_experts)
        rho_init = -6.0 #self._softplus_inverse(0.006)
        self.rho_initializer = tf.keras.initializers.random_normal(mean=rho_init, stddev=0.01)
        self.expert_activation = activations.get(expert_activation)
        self.gating_activation = activations.get(gating_activation)

        self.use_expert_bias = use_expert_bias
        self.use_gating_bias = use_gating_bias

        self.expert_bias_initializer = initializers.get(expert_bias_initializer)
        self.gating_bias_initializer = initializers.get(gating_bias_initializer)

        self.expert_kernel_regularizer = regularizers.get(expert_kernel_regularizer)
        self.gating_kernel_regularizer = regularizers.get(gating_kernel_regularizer)

        self.expert_bias_regularizer = regularizers.get(expert_bias_regularizer)
        self.gating_bias_regularizer = regularizers.get(gating_bias_regularizer)

        self.expert_kernel_constraint = constraints.get(expert_kernel_constraint)
        self.gating_kernel_constraint = constraints.get(gating_kernel_constraint)

        self.expert_bias_constraint = constraints.get(expert_bias_constraint)
        self.gating_bias_constraint = constraints.get(gating_bias_constraint)

        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.input_spec = InputSpec(ndim=self.rank + 2)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
            self.tf_data_format = 'NCHW'
        else:
            channel_axis = -1
            self.tf_data_format = 'NHWC'

        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs should be defined. Found `None`.')

        input_dim = input_shape[channel_axis]

        expert_kernel_shape = self.kernel_size + (input_dim, self.n_filters, self.n_experts)
        expert_rho_shape = self.kernel_size[1:] + (input_dim, self.n_filters, self.n_experts)

        self.expert_mu_kernel = self.add_weight(shape=expert_kernel_shape,
                                                initializer=self.expert_kernel_initializer,
                                                name='post_expert_mu_kernel',
                                                regularizer=self.expert_kernel_regularizer,
                                                constraint=self.expert_kernel_constraint)

        self.expert_rho_kernel = self.add_weight(shape=expert_rho_shape,
                                                 initializer=self.rho_initializer,
                                                 name='post_expert_rho_kernel',
                                                 regularizer=self.expert_kernel_regularizer,
                                                 constraint=self.expert_kernel_constraint)

        self.prior_expert_mu_kernel = self.add_weight(shape=expert_kernel_shape,
                                                      initializer="glorot_uniform",
                                                      name='prior_expert_mu_kernel',
                                                      regularizer=self.expert_kernel_regularizer,
                                                      constraint=self.expert_kernel_constraint,
                                                      trainable=False)

        self.prior_expert_rho_kernel = self.add_weight(shape=expert_rho_shape,
                                                       initializer=self.rho_initializer,
                                                       name='prior_expert_rho_kernel',
                                                       regularizer=self.expert_kernel_regularizer,
                                                       constraint=self.expert_kernel_constraint,
                                                       trainable=False)
        self.n_gating_filters = self.n_filters
        gating_kernel_shape = self.kernel_size + (input_dim, self.n_gating_filters)
        output_size_offset = 1  # elf.rank - 1
        padding = int(np.ceil((self.kernel_size[
                                   0] - 1) / 2.0))  # (self.strides[0] - 1) * input_shape[1] - self.strides[0] + self.kernel_size[0] / 2.0)
        gating_out_kernel_size = int(
            np.floor((input_shape[1] + 2 * padding - self.kernel_size[0]) / self.strides[0]) + output_size_offset)
        gating_out_kernel_size = int(gating_out_kernel_size * 0.5)  # apply max pooling factor
        self.gating_out_kernel_shape = (self.n_gating_filters * gating_out_kernel_size ** self.rank)
        if self.n_experts > 1:
            self.gating_kernel = self.add_weight(shape=gating_kernel_shape,
                                                 initializer="he_uniform",
                                                 name='post_gating_kernel',
                                                 regularizer=self.gating_kernel_regularizer,
                                                 constraint=self.gating_kernel_constraint)
            self.gating_out_kernel = self.add_weight(shape=(self.gating_out_kernel_shape, self.n_experts),
                                                     initializer="he_uniform",
                                                     name='post_gating_out_kernel',
                                                     regularizer=self.gating_kernel_regularizer,
                                                     constraint=self.gating_kernel_constraint)

            self.prior_gating_kernel = self.add_weight(shape=gating_kernel_shape,
                                                       initializer="he_uniform",
                                                       name='prior_gating_kernel',
                                                       regularizer=self.gating_kernel_regularizer,
                                                       constraint=self.gating_kernel_constraint,
                                                       trainable=False)
            self.prior_gating_out_kernel = self.add_weight(shape=(self.gating_out_kernel_shape, self.n_experts),
                                                           initializer="he_uniform",
                                                           name='prior_gating_out_kernel',
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
            expert_bias_shape = (self.n_filters, self.n_experts)
            self.expert_bias = self.add_weight(shape=expert_bias_shape,
                                               initializer=self.expert_bias_initializer,
                                               name='post_expert_bias',
                                               regularizer=self.expert_bias_regularizer,
                                               constraint=self.expert_bias_constraint)
            self.prior_expert_bias = self.add_weight(shape=expert_bias_shape,
                                                     initializer=self.expert_bias_initializer,
                                                     name='prior_expert_bias',
                                                     regularizer=self.expert_bias_regularizer,
                                                     constraint=self.expert_bias_constraint,
                                                     trainable=False)
        else:
            self.expert_bias = None
            self.prior_expert_bias = None

        if self.use_gating_bias and self.n_experts > 1:
            self.gating_bias = self.add_weight(shape=(self.n_gating_filters,),
                                               initializer=self.gating_bias_initializer,
                                               name='post_gating_bias',
                                               regularizer=self.gating_bias_regularizer,
                                               constraint=self.gating_bias_constraint)
            self.prior_gating_bias = self.add_weight(shape=(self.n_gating_filters,),
                                                     initializer=self.gating_bias_initializer,
                                                     name='prior_gating_bias',
                                                     regularizer=self.gating_bias_regularizer,
                                                     constraint=self.gating_bias_constraint,
                                                     trainable=False)

            self.gating_out_bias = self.add_weight(shape=(self.n_experts,),
                                                   initializer=self.gating_bias_initializer,
                                                   name='post_gating_out_bias',
                                                   regularizer=self.gating_bias_regularizer,
                                                   constraint=self.gating_bias_constraint)
            self.prior_gating_out_bias = self.add_weight(shape=(self.n_experts,),
                                                         initializer=self.gating_bias_initializer,
                                                         name='prior_gating_out_bias',
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
        self.new_gating_outputs_shape = (-1,)
        for i in range(self.rank):
            self.new_gating_outputs_shape = self.new_gating_outputs_shape + (1,)
        self.new_gating_outputs_shape = self.new_gating_outputs_shape + (self.n_filters, self.n_experts)

        kernel_shape = self.kernel_size + (input_dim, self.n_filters)
        self._convolution_op = nn_ops.Convolution(
            input_shape,
            filter_shape=tf.TensorShape(kernel_shape),
            dilation_rate=self.dilation_rate,
            strides=self.strides,
            padding=self.padding.upper(),
            data_format=tf_layers_util.convert_data_format(
                self.data_format, self.rank + 2))

        self.gating_convolution_op = nn_ops.Convolution(
            input_shape,
            filter_shape=tf.TensorShape(gating_kernel_shape),
            dilation_rate=self.dilation_rate,
            strides=self.strides,
            padding=self.padding.upper(),
            data_format=tf_layers_util.convert_data_format(
                self.data_format, self.rank + 2))
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs, **kwargs):

        if self.n_experts > 1:
            noisy_gating_outputs, activated_gating_outputs, \
            activated_prior_gating_outputs, gating_cond_entropy, attn_inuts = self.apply_gating_kernels(inputs,
                                                                                                        **kwargs)

            # gating entropy penalty
            batch_gating_prior = tfp.distributions.Categorical(
                probs=tf.reduce_mean(activated_gating_outputs, axis=0, keepdims=True))
            batch_activated_gating_entropy = tf.reduce_mean(self.entropy_fun(batch_gating_prior))
            self.add_loss(self.diversity_bonus * batch_activated_gating_entropy)
            self.add_loss(self.diversity_bonus * gating_cond_entropy)

            gates = self.top_k_gating(noisy_gating_outputs)

            # gating KL div penalty
            gating_posterior_dist = tfp.distributions.Categorical(probs=activated_gating_outputs,
                                                                  validate_args=True)
            gating_prior_dist = tfp.distributions.Categorical(probs=activated_prior_gating_outputs,
                                                              validate_args=True)
            kl_loss = tf.reduce_mean(self.kl_div_fun(gating_posterior_dist, gating_prior_dist), axis=0)
            self.add_loss(self.gating_kl_weight * kl_loss)
            # balance_loss = self.load_balanced_loss(router_probs=activated_gating_outputs, expert_mask=gates)
            # self.add_loss(self.diversity_bonus * balance_loss)
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

            # exp_diversity = 1.0 - tf.math.square(dkl_mat[0][1])  # tf.linalg.det(dkl_mat)
            exp_diversity = tf.linalg.det(dkl_mat)
            div_bonus = -exp_diversity
            self.add_loss(self.diversity_bonus * div_bonus)
        else:
            activated_gating_outputs = None
            gates = tf.ones(shape=(tf.shape(inputs)[0], 1))

        dispatcher = SparseDispatcher(self.n_experts, gates)
        e_inputs = dispatcher.dispatch(inputs)
        dispatcher.expert_to_gates()

        expert_outputs, approx_expert_dkl, output_shape = self.apply_kernels(e_inputs)
        self.add_loss(self.kl_weight * tf.reduce_mean(approx_expert_dkl))
        if self.n_experts > 1:
            outputs = dispatcher.combine(expert_outputs)
        else:
            outputs = expert_outputs[0]
        outputs = tf.keras.layers.Reshape(output_shape)(outputs)
        return outputs

    def load_balanced_loss(self, router_probs, expert_mask):
        # router_probs [tokens_per_batch, num_experts] is the probability assigned for
        # each expert per token. expert_mask [tokens_per_batch, num_experts] contains
        # the expert with the highest router probability in one−hot format.

        num_experts = tf.shape(expert_mask)[-1]
        # Get the fraction of tokens routed to each expert.
        # density is a vector of length num experts that sums to 1.
        density = tf.reduce_mean(expert_mask, axis=0)
        # Get fraction of probability mass assigned to each expert from the router
        # across all tokens. density_proxy is a vector of length num experts that sums to 1.
        density_proxy = tf.reduce_mean(router_probs, axis=0)
        # Want both vectors to have uniform allocation (1/num experts) across all
        # num_expert elements. The two vectors will be pushed towards uniform allocation
        # when the dot product is minimized.
        loss = tf.reduce_mean(density_proxy * density) * tf.cast(
            (num_experts ** 2), tf.dtypes.float32
        )
        return loss

    def apply_kernels(self, expert_input_tensors):

        def random_rademacher(shape, dtype=tf.float32, seed=None):
            int_dtype = tf.int64 if tf.as_dtype(dtype) != tf.int32 else tf.int32
            random_bernoulli = tf.random.uniform(
                shape, minval=0, maxval=2, dtype=int_dtype, seed=seed)
            return tf.cast(2 * random_bernoulli - 1, dtype)

        expert_outputs = []
        expert_kls = []
        for i in range(self.n_experts):
            expert_input = expert_input_tensors[i]
            input_shape = tf.shape(expert_input)
            batch_shape = tf.expand_dims(input_shape[0], 0)
            if self.data_format == 'channels_first':
                channels = input_shape[1]
            else:
                channels = input_shape[-1]

            if self.rank == 2:
                posterior_affine_tensor = tfp.distributions.Normal(
                    loc=tf.zeros_like(self.expert_mu_kernel[:, :, :, :, i]),
                    scale=tf.nn.softplus(self.expert_rho_kernel[:, :, :, i])).sample()
            if self.rank == 1:
                posterior_affine_tensor = tfp.distributions.Normal(
                    loc=tf.zeros_like(self.expert_mu_kernel[:, :, :, i]),
                    scale=tf.nn.softplus(self.expert_rho_kernel[:, :, i])).sample()

            if self.rank == 2:
                outputs = self._convolution_op(
                    expert_input, self.expert_mu_kernel[:, :, :, :, i])
            if self.rank == 1:
                outputs = self._convolution_op(
                    expert_input, self.expert_mu_kernel[:, :, :, i])

            sign_input = random_rademacher(
                tf.concat([batch_shape,
                           tf.expand_dims(channels, 0)], 0),
                dtype=expert_input.dtype)
            sign_output = random_rademacher(
                tf.concat([batch_shape,
                           tf.expand_dims(self.n_filters, 0)], 0),
                dtype=expert_input.dtype)

            if self.data_format == 'channels_first':
                for _ in range(self.rank):
                    sign_input = tf.expand_dims(sign_input, -1)  # 2D ex: (B, C, 1, 1)
                    sign_output = tf.expand_dims(sign_output, -1)
            else:
                for _ in range(self.rank):
                    sign_input = tf.expand_dims(sign_input, 1)  # 2D ex: (B, 1, 1, C)
                    sign_output = tf.expand_dims(sign_output, 1)

            perturbed_inputs = self._convolution_op(
                expert_input * sign_input, posterior_affine_tensor) * sign_output

            outputs += perturbed_inputs
            if self.use_expert_bias:
                outputs = nn.bias_add(outputs, self.expert_bias[:, i], data_format=self.tf_data_format)
            if self.expert_activation is not None:
                outputs = self.expert_activation(outputs)
            flat_expert_outputs = tf.keras.layers.Flatten()(outputs)
            expert_outputs.append(flat_expert_outputs)
            exp_kl = self.approx_expert_dkl(exp=i)
            expert_kls.append(exp_kl)
        return expert_outputs, expert_kls, outputs.shape[1:]

    def approx_expert_dkl(self, exp, q=None):
        """
        Computes DKL between p and q
        :param exp: index of expert distribution
        :param q: if none, then q is exp's prior
        :return: DKL as tensor
        """
        if self.rank == 2:
            cov_post = tf.nn.softplus(self.expert_rho_kernel[:, :, :, exp])
            mu_post = self.expert_mu_kernel[:, :, :, :, exp]

        if self.rank == 1:
            cov_post = tf.nn.softplus(self.expert_rho_kernel[:, :, exp])
            mu_post = self.expert_mu_kernel[:, :, :, exp]

        if q is not None:
            exp = q

        if self.rank == 2:
            cov_prior = tf.nn.softplus(self.prior_expert_rho_kernel[:, :, :, exp])
            mu_prior = self.prior_expert_mu_kernel[:, :, :, :, exp]

        if self.rank == 1:
            cov_prior = tf.nn.softplus(self.prior_expert_rho_kernel[:, :, exp])
            mu_prior = self.prior_expert_mu_kernel[:, :, :, exp]

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
        # update gating prior
        # avg_gates = tf.reduce_sum(gates, axis=0, keepdims=True)
        # avg_gates = avg_gates / tf.reduce_sum(avg_gates)
        return gates

    def apply_gating_kernels(self, inputs, **kwargs):
        gating_outputs = self.gating_convolution_op(inputs, self.gating_kernel)
        prior_gating_outputs = self.gating_convolution_op(inputs, self.prior_gating_kernel)
        if self.use_gating_bias:
            gating_outputs = K.bias_add(gating_outputs, self.gating_bias, data_format=self.data_format)
            prior_gating_outputs = K.bias_add(prior_gating_outputs, self.prior_gating_bias,
                                              data_format=self.data_format)
        gating_outputs = tf.nn.leaky_relu(gating_outputs)
        gating_outputs = MaxPooling2D()(gating_outputs)
        gating_outputs = tf.keras.layers.Flatten()(gating_outputs)

        prior_gating_outputs = tf.nn.leaky_relu(prior_gating_outputs)
        prior_gating_outputs = MaxPooling2D()(prior_gating_outputs)
        prior_gating_outputs = tf.keras.layers.Flatten()(prior_gating_outputs)

        gating_outputs = K.dot(gating_outputs, self.gating_out_kernel)
        gating_outputs = K.bias_add(gating_outputs, self.gating_out_bias)
        prior_gating_outputs = K.dot(prior_gating_outputs, self.prior_gating_out_kernel)
        prior_gating_outputs = K.bias_add(prior_gating_outputs, self.prior_gating_out_bias)

        if self.gating_activation is not None:
            activated_gating_outputs = self.gating_activation(gating_outputs)
            activated_prior_gating_outputs = self.gating_activation(prior_gating_outputs)
        return gating_outputs, activated_gating_outputs, activated_prior_gating_outputs, 0.0, inputs

    def norm(self, x):
        n = tf.sqrt(tf.reduce_sum(tf.square(x)))
        return n

    def wasserstein_dist(self, i, j):
        if self.rank == 2:
            diff = tf.reduce_mean(self.expert_mu_kernel[:, :, :, :, i] - self.expert_mu_kernel[:, :, :, :, j], axis=0)
        elif self.rank == 1:
            diff = tf.reduce_mean(self.expert_mu_kernel[:, :, :, i] - self.expert_mu_kernel[:, :, :, j], axis=0)
        mu_norm = tf.square(tf.norm(diff))
        # Hellinger Distance between diagonal Covariance Matrices
        if self.rank == 2:
            cov_i = tf.math.sqrt(tf.nn.softplus(self.expert_rho_kernel[:, :, :, i]) + 1e-6)
            cov_j = tf.math.sqrt(tf.nn.softplus(self.expert_rho_kernel[:, :, :, j]) + 1e-6)
        else:
            cov_i = tf.math.sqrt(tf.nn.softplus(self.expert_rho_kernel[:, :, i]) + 1e-6)
            cov_j = tf.math.sqrt(tf.nn.softplus(self.expert_rho_kernel[:, :, j]) + 1e-6)
        cov_trace = self.norm(cov_i - cov_j)
        w = mu_norm + cov_trace
        return w

    def get_config(self):
        config = {
            'rank': self.rank,
            'n_filters': self.n_filters,
            'n_experts': self.n_experts,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
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
            'activity_regularizer': regularizers.serialize(self.activity_regularizer)
        }
        base_config = super(_ConvMoE, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _softplus_inverse(self, x):
        """Helper which computes the function inverse of `tf.nn.softplus`."""
        return tf.math.log(tf.math.expm1(x))

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
        Returns:
        values: a Tensor of shape [batch_size, k]
        indices: a int32 Tensor of shape [batch_size, k]
        """
        if k > 10:
            return tf.nn.top_k(x, k)
        values = []
        indices = []
        depth = tf.shape(x)[1]
        for i in range(k):
            if not soft:
                values.append(tf.reduce_max(x, 1))
                idx = tf.argmax(x, 1)
                indices.append(idx)
            else:
                dist = ds.Categorical(logits=x)
                idx = dist.sample()
                values.append(dist.log_prob(idx))
                indices.append(idx)
            if i + 1 < k:
                x += tf.one_hot(idx, depth, -1e9)
        return tf.stack(values, axis=1), tf.cast(tf.stack(indices, axis=1), dtype=tf.int32)

    def _assign_moving_average(self, variable, value, momentum, inputs_size=None):
        with K.name_scope('AssignMovingAvg') as scope:
            with ops.colocate_with(variable):
                decay = 1.0 - momentum
                update_delta = (variable - math_ops.cast(value, variable.dtype)) * decay
                if inputs_size is not None:
                    update_delta = array_ops.where(inputs_size > 0, update_delta,
                                                   K.zeros_like(update_delta))
                return state_ops.assign_sub(variable, update_delta, name=scope)

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
            return tf.cast(is_in, dtype=tf.float32)
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
        float_size = tf.cast(tf.size(x), dtype=tf.float32) + epsilon
        mean = tf.reduce_sum(x) / float_size
        variance = tf.reduce_sum(tf.math.squared_difference(x, mean)) / float_size
        return variance / (tf.square(mean) + epsilon)


class Conv1DMoE(_ConvMoE):
    """1D convolution layer (e.g. temporal convolution).

    # Input shape
        3D tensor with shape: `(batch_size, steps, input_dim)`

    # Output shape
        3D tensor with shape: `(batch_size, new_steps, n_filters)`
        `steps` value might have changed due to padding or strides.
    """

    def __init__(self,
                 n_filters,
                 n_experts,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format='channels_last',
                 dilation_rate=1,
                 expert_activation=None,
                 gating_activation=None,
                 use_expert_bias=True,
                 use_gating_bias=True,
                 expert_bias_initializer='zeros',
                 gating_bias_initializer='zeros',
                 expert_kernel_regularizer=None,
                 gating_kernel_regularizer=None,
                 expert_bias_regularizer=None,
                 gating_bias_regularizer=None,
                 expert_kernel_constraint=None,
                 gating_kernel_constraint=None,
                 expert_bias_constraint=None,
                 gating_bias_constraint=None,
                 activity_regularizer=None,
                 **kwargs):
        if padding == 'causal':
            if data_format != 'channels_last':
                raise ValueError(
                    'When using causal padding in `Conv1DMoE`, `data_format` must be "channels_last" (temporal data).')
        super(Conv1DMoE, self).__init__(
            rank=1,
            n_filters=n_filters,
            n_experts=n_experts,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            expert_activation=expert_activation,
            gating_activation=gating_activation,
            use_expert_bias=use_expert_bias,
            use_gating_bias=use_gating_bias,
            expert_bias_initializer=expert_bias_initializer,
            gating_bias_initializer=gating_bias_initializer,
            expert_kernel_regularizer=expert_kernel_regularizer,
            gating_kernel_regularizer=gating_kernel_regularizer,
            expert_bias_regularizer=expert_bias_regularizer,
            gating_bias_regularizer=gating_bias_regularizer,
            expert_kernel_constraint=expert_kernel_constraint,
            gating_kernel_constraint=gating_kernel_constraint,
            expert_bias_constraint=expert_bias_constraint,
            gating_bias_constraint=gating_bias_constraint,
            activity_regularizer=activity_regularizer,
            **kwargs)
        self.input_spec = InputSpec(ndim=3)

    def get_config(self):
        config = super(Conv1DMoE, self).get_config()
        config.pop('rank')
        return config


class Conv2DMoE(_ConvMoE):
    """2D convolution layer (e.g. spatial convolution over images).

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)`
        if `data_format` is `"channels_first"`
        or 4D tensor with shape:
        `(samples, rows, cols, channels)`
        if `data_format` is `"channels_last"`.

    # Output shape
        4D tensor with shape:
        `(samples, n_filters, new_rows, new_cols)`
        if `data_format` is `"channels_first"`
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, n_filters)`
        if `data_format` is `"channels_last"`.
        `rows` and `cols` values might have changed due to padding.
    """

    def __init__(self,
                 n_filters,
                 n_experts,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format='channels_last',
                 dilation_rate=(1, 1),
                 expert_activation=None,
                 gating_activation=None,
                 use_expert_bias=True,
                 use_gating_bias=True,
                 expert_bias_initializer='zeros',
                 gating_bias_initializer='zeros',
                 expert_kernel_regularizer=None,
                 gating_kernel_regularizer=None,
                 expert_bias_regularizer=None,
                 gating_bias_regularizer=None,
                 expert_kernel_constraint=None,
                 gating_kernel_constraint=None,
                 expert_bias_constraint=None,
                 gating_bias_constraint=None,
                 activity_regularizer=None,
                 **kwargs):
        super(Conv2DMoE, self).__init__(
            rank=2,
            n_filters=n_filters,
            n_experts=n_experts,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            expert_activation=expert_activation,
            gating_activation=gating_activation,
            use_expert_bias=use_expert_bias,
            use_gating_bias=use_gating_bias,
            expert_bias_initializer=expert_bias_initializer,
            gating_bias_initializer=gating_bias_initializer,
            expert_kernel_regularizer=expert_kernel_regularizer,
            gating_kernel_regularizer=gating_kernel_regularizer,
            expert_bias_regularizer=expert_bias_regularizer,
            gating_bias_regularizer=gating_bias_regularizer,
            expert_kernel_constraint=expert_kernel_constraint,
            gating_kernel_constraint=gating_kernel_constraint,
            expert_bias_constraint=expert_bias_constraint,
            gating_bias_constraint=gating_bias_constraint,
            activity_regularizer=activity_regularizer,
            **kwargs)
        self.input_spec = InputSpec(ndim=4)

    def get_config(self):
        config = super(Conv2DMoE, self).get_config()
        config.pop('rank')
        return config


class Conv3DMoE(_ConvMoE):
    """3D convolution layer (e.g. spatial convolution over volumes).

    # Input shape
        5D tensor with shape:
        `(samples, channels, conv_dim1, conv_dim2, conv_dim3)`
        if `data_format` is `"channels_first"`
        or 5D tensor with shape:
        `(samples, conv_dim1, conv_dim2, conv_dim3, channels)`
        if `data_format` is `"channels_last"`.

    # Output shape
        5D tensor with shape:
        `(samples, n_filters, new_conv_dim1, new_conv_dim2, new_conv_dim3)`
        if `data_format` is `"channels_first"`
        or 5D tensor with shape:
        `(samples, new_conv_dim1, new_conv_dim2, new_conv_dim3, n_filters)`
        if `data_format` is `"channels_last"`.
        `new_conv_dim1`, `new_conv_dim2` and `new_conv_dim3` values might have changed due to padding.
    """

    def __init__(self,
                 n_filters,
                 n_experts,
                 kernel_size,
                 strides=(1, 1, 1),
                 padding='valid',
                 data_format='channels_last',
                 dilation_rate=(1, 1, 1),
                 expert_activation=None,
                 gating_activation=None,
                 use_expert_bias=True,
                 use_gating_bias=True,
                 expert_bias_initializer='zeros',
                 gating_bias_initializer='zeros',
                 expert_kernel_regularizer=None,
                 gating_kernel_regularizer=None,
                 expert_bias_regularizer=None,
                 gating_bias_regularizer=None,
                 expert_kernel_constraint=None,
                 gating_kernel_constraint=None,
                 expert_bias_constraint=None,
                 gating_bias_constraint=None,
                 activity_regularizer=None,
                 **kwargs):
        super(Conv3DMoE, self).__init__(
            rank=3,
            n_filters=n_filters,
            n_experts=n_experts,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            expert_activation=expert_activation,
            gating_activation=gating_activation,
            use_expert_bias=use_expert_bias,
            use_gating_bias=use_gating_bias,
            expert_bias_initializer=expert_bias_initializer,
            gating_bias_initializer=gating_bias_initializer,
            expert_kernel_regularizer=expert_kernel_regularizer,
            gating_kernel_regularizer=gating_kernel_regularizer,
            expert_bias_regularizer=expert_bias_regularizer,
            gating_bias_regularizer=gating_bias_regularizer,
            expert_kernel_constraint=expert_kernel_constraint,
            gating_kernel_constraint=gating_kernel_constraint,
            expert_bias_constraint=expert_bias_constraint,
            gating_bias_constraint=gating_bias_constraint,
            activity_regularizer=activity_regularizer,
            **kwargs)
        self.input_spec = InputSpec(ndim=5)

    def get_config(self):
        config = super(Conv3DMoE, self).get_config()
        config.pop('rank')
        return config


# Aliases
Convolution1DMoE = Conv1DMoE
Convolution2DMoE = Conv2DMoE
Convolution3DMoE = Conv3DMoE
