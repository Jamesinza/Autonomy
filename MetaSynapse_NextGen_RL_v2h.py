# Silencing annoying warnings
import shutup
shutup.please()

import os
import tensorflow as tf
# import tensorflow_probability as tfp
import pandas as pd
import numpy as np
import math
import random
import time

# Set seeds for reproducibility.
np.random.seed(42)
tf.random.set_seed(42)

class IdentityLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return inputs

class ConcreteDropout(tf.keras.layers.Layer):
    def __init__(self, layer, weight_regularizer, dropout_regularizer, init_dropout=0.1, temperature=0.1, **kwargs):
        """
        Args:
          layer: The layer to wrap. For dropout-only behavior you can use an IdentityLayer.
          weight_regularizer: Coefficient for weight regularization (set to 0 if not needed).
          dropout_regularizer: Coefficient for dropout regularization.
          init_dropout: Initial dropout probability.
          temperature: Temperature parameter for the Concrete distribution.
        """
        super(ConcreteDropout, self).__init__(**kwargs)
        self.layer = layer
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.temperature = temperature
        # Initialize dropout probability in logit space.
        p = init_dropout
        self.p_logit = tf.Variable(tf.math.log(p) - tf.math.log(1.0 - p), trainable=True, dtype=tf.float32)

    def call(self, inputs, training=None):
        p = tf.sigmoid(self.p_logit)
        if training:
            eps = 1e-7
            unif_noise = tf.random.uniform(tf.shape(inputs))
            drop_prob = tf.math.log(p + eps) - tf.math.log(1.0 - p + eps) \
                        + tf.math.log(unif_noise + eps) - tf.math.log(1.0 - unif_noise + eps)
            drop_prob = tf.sigmoid(drop_prob / self.temperature)
            random_tensor = 1.0 - drop_prob
            # Scale inputs by (1-p) to maintain expectation.
            outputs = self.layer(inputs) * random_tensor / (1.0 - p)
        else:
            outputs = self.layer(inputs)
        return outputs

    def get_regularization_loss(self):
        p = tf.sigmoid(self.p_logit)
        # If the wrapped layer has a kernel, add weight regularization.
        if hasattr(self.layer, 'kernel') and self.layer.kernel is not None:
            weight_reg = self.weight_regularizer * tf.reduce_sum(tf.square(self.layer.kernel))
        else:
            weight_reg = 0.0
        # Dropout regularization encourages an optimal dropout probability.
        dropout_reg = self.dropout_regularizer * tf.cast(tf.size(self.p_logit), tf.float32) * (
            p * tf.math.log(p + 1e-7) + (1.0 - p) * tf.math.log(1.0 - p + 1e-7))
        return weight_reg + dropout_reg

# =========================================================================
# Learned Activations combined with Quantum-Inspired Stochastic Activation
# Dynamically adjust weights based on input context
# =========================================================================
class LearnedQuantumActivation(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LearnedQuantumActivation, self).__init__(**kwargs)

    def build(self, input_shape):
        # We have 9 candidate activation functions (8 computed + identity)
        self.act_w = self.add_weight(
            shape=(9,),
            initializer=tf.keras.initializers.Ones(),
            trainable=True,
            name='activation_weights')
        self.act_temperature = self.add_weight(
            shape=(),
            initializer=tf.keras.initializers.Constant(0.5),
            trainable=True,
            name='activation_temperature')
        super(LearnedQuantumActivation, self).build(input_shape)

    def call(self, inputs, training=True):
        # Compute candidate activations.
        sig = tf.keras.activations.sigmoid(inputs)
        elu = tf.keras.activations.elu(inputs)
        tanh = tf.keras.activations.tanh(inputs)
        relu = tf.keras.activations.relu(inputs)
        silu = tf.keras.activations.silu(inputs)
        gelu = tf.keras.activations.gelu(inputs)
        selu = tf.keras.activations.selu(inputs)
        mish = tf.keras.activations.mish(inputs)
        linear = inputs  # Identity activation

        # Stack all candidate activations along a new dimension.
        # If inputs has shape (batch, ...), the stacked tensor will have shape (batch, ..., 9).
        acts = tf.stack([sig, elu, tanh, relu, silu, gelu, selu, mish, linear], axis=-1)
        
        # Compute softmax of learnable weights to ensure they form a valid convex combination.
        weights = tf.nn.softmax(self.act_w)  # Shape: (9,)
        
        # Compute weighted sum across candidate activations.
        base_activation = tf.reduce_sum(acts * weights, axis=-1)
        del acts
        # Generate stochastic phase noise.
        phase_noise = tf.random.normal(tf.shape(inputs)) * self.act_temperature
        
        # Apply noise: add noise for positive activations, subtract absolute noise for negative ones.
        noisy_activation = tf.where(base_activation > 0,
                                    base_activation + phase_noise,
                                    base_activation - tf.abs(phase_noise)) if training else base_activation
        
        return noisy_activation

    def compute_output_shape(self, input_shape):
        return input_shape

# =============================================================================
# Unsupervised Feature Extractor (Autoencoder with U-Net inspired architecture)
# =============================================================================
class UnsupervisedFeatureExtractor(tf.keras.Model):
    def __init__(self, units=256, **kwargs):
        super(UnsupervisedFeatureExtractor, self).__init__(**kwargs)
        self.units = units
        # Encoder layers
        self.enc_cnn1 = tf.keras.layers.Conv1D(self.units, 3, activation="relu", padding="same", name="enc_cnn1")
        self.enc_pool1 = tf.keras.layers.MaxPooling1D(2, padding="same", name="enc_pool1")
        self.enc_cnn2 = tf.keras.layers.Conv1D(self.units*2, 3, activation="relu", padding="same", name="enc_cnn2")
        self.enc_pool2 = tf.keras.layers.MaxPooling1D(2, padding="same", name="enc_pool2")
        self.enc_latent = tf.keras.layers.Conv1D(self.units*4, 3, activation="relu", padding="same", name="enc_latent")
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling1D(name="global_avg_pool")
        # Decoder layers
        self.dec_upconv1 = tf.keras.layers.Conv1DTranspose(self.units*2, 3, strides=2, activation="relu", padding="same", name="dec_upconv1")
        self.dec_concat1 = tf.keras.layers.Concatenate(name="dec_concat1")
        self.dec_upconv2 = tf.keras.layers.Conv1DTranspose(self.units, 3, strides=2, activation="relu", padding="same", name="dec_upconv2")
        self.dec_output = tf.keras.layers.Conv1D(16, 3, activation="sigmoid", padding="same", name="dec_output")        

    def build(self, input_shape):
        super(UnsupervisedFeatureExtractor, self).build(input_shape)        

    def encode(self, x):
        x = self.enc_cnn1(x)
        x = self.enc_pool1(x)
        x1 = self.enc_cnn2(x)
        x = self.enc_pool2(x1)
        latent = self.enc_latent(x)
        flat_latent = self.global_avg_pool(latent)
        return x1, latent, flat_latent

    def decode(self, x1, latent):
        y1 = self.dec_upconv1(latent)
        y1 = self.dec_concat1([x1, y1])
        y = self.dec_upconv2(y1)
        return self.dec_output(y)

    def call(self, inputs):
        x1, latent, flat_latent = self.encode(inputs)
        reconstruction = self.decode(x1, latent)
        return flat_latent, reconstruction

    def compute_output_shape(self, input_shape):
        return tuple(input_shape)        

class DynamicPlasticDenseDendritic(tf.keras.layers.Layer):
    def __init__(self, max_units, initial_units, plasticity_controller, num_branches=4, **kwargs):
        super(DynamicPlasticDenseDendritic, self).__init__(name='DynamicPlasticDenseDendritic', **kwargs)
        self.max_units = max_units
        self.initial_units = initial_units
        self.plasticity_controller = plasticity_controller
        self.num_branches = num_branches
        self.act = LearnedQuantumActivation()
        self.dynamic_connectivity = DynamicConnectivity(max_units=self.max_units)
        
        # Make formerly fixed hyperparameters trainable.
        self.prune_threshold = tf.Variable(0.01, trainable=True, dtype=tf.float32, name='prune_threshold')
        self.add_prob = tf.Variable(0.01, trainable=True, dtype=tf.float32, name='add_prob')
        # Running statistics (non-trainable, used for monitoring)
        self.sparsity = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.avg_weight_magnitude = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.avg_delay_magnitude = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        # Neuron mask and activation average for dynamic architecture.
        init_mask = [1.0] * initial_units + [0.0] * (max_units - initial_units)
        self.neuron_mask = tf.Variable(init_mask, trainable=False, dtype=tf.float32)
        self.neuron_activation_avg = tf.Variable(tf.zeros([max_units], dtype=tf.float32), trainable=False)
        # Gating layer for combining outputs from dendritic branches.
        self.branch_gating = tf.keras.layers.Dense(self.num_branches, activation='softmax', name="branch_gating")

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        # Weight tensor: shape (input_dim, max_units, num_branches)
        self.w = self.add_weight(
            shape=(input_dim, self.max_units, self.num_branches),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True, name='dpd_kernel')
        # Bias: shape (max_units, num_branches)
        self.b = self.add_weight(
            shape=(self.max_units, self.num_branches),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True, name='dpd_bias')
        # Delay for each branch.
        self.delay = self.add_weight(
            shape=(input_dim, self.max_units, self.num_branches),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True, name='delay')
        # Trainable parameters for homeostatic scaling and structural thresholds.
        self.decay_factor = self.add_weight(
            shape=(), initializer=tf.keras.initializers.Constant(0.9),
            trainable=True, name="decay_factor")
        self.prune_activation_threshold = self.add_weight(
            shape=(), initializer=tf.keras.initializers.Constant(0.01),
            trainable=True, name="prune_activation_threshold")
        self.growth_activation_threshold = self.add_weight(
            shape=(), initializer=tf.keras.initializers.Constant(0.8),
            trainable=True, name="growth_activation_threshold")
        self.target_avg = self.add_weight(
            shape=(), initializer=tf.keras.initializers.Constant(0.2),
            trainable=True, name="target_avg")
        self.target_delay = self.add_weight(
            shape=(), initializer=tf.keras.initializers.Constant(0.2),
            trainable=True, name="target_delay")
        self.alpha = self.add_weight(
            shape=(), initializer=tf.keras.initializers.Constant(0.9),
            trainable=True, name="alpha")
        # Meta scaling parameters for structural updates.
        self.prune_min = self.add_weight(
            shape=(), initializer=tf.keras.initializers.Constant(1.05),
            trainable=True, name="prune_min")
        self.prune_max = self.add_weight(
            shape=(), initializer=tf.keras.initializers.Constant(0.95),
            trainable=True, name="prune_max")
        self.prob_min = self.add_weight(
            shape=(), initializer=tf.keras.initializers.Constant(1.05),
            trainable=True, name="prob_min")
        self.prob_max = self.add_weight(
            shape=(), initializer=tf.keras.initializers.Constant(0.95),
            trainable=True, name="prob_max")
        super(DynamicPlasticDenseDendritic, self).build(input_shape)

    def call(self, inputs, meta_signal=None):
        # Compute branch outputs.
        w_mod = self.w * tf.math.sigmoid(self.delay)
        branch_outputs = tf.tensordot(inputs, w_mod, axes=[[1], [0]])
        branch_outputs += self.b
        gating_weights = self.branch_gating(inputs)
        gating_weights = tf.expand_dims(gating_weights, axis=1)
        z_combined = tf.reduce_sum(branch_outputs * gating_weights, axis=-1)
        
        # Apply dynamic connectivity and neuron mask.
        connectivity = self.dynamic_connectivity(tf.expand_dims(self.neuron_activation_avg, axis=0))
        connectivity = tf.cast(connectivity, z_combined.dtype)
        mask = tf.cast(tf.expand_dims(self.neuron_mask, axis=0), z_combined.dtype)
        z_modulated = z_combined * connectivity * mask
        a = self.act(z_modulated)
        
        # Update running average (exponential moving average)
        batch_mean = tf.reduce_mean(a, axis=0)  # Shape: (max_units,)
        new_avg = self.alpha * self.neuron_activation_avg + (1 - self.alpha) * tf.cast(batch_mean, self.neuron_activation_avg.dtype)
        
        # If a meta signal is provided, squeeze it to a scalar before modulation.
        if meta_signal is not None:
            # Reduce meta_signal to a scalar.
            meta_scalar = tf.reduce_mean(meta_signal)
            new_avg *= meta_scalar
        
        self.neuron_activation_avg.assign(new_avg)
        return a

    def vectorized_plasticity_update(self, pre_activity, post_activity, reward, meta_signal=None):
        """
        Compute plasticity updates using the meta-plasticity controller.
        meta_signal (if provided) scales the update magnitude.
        """
        pre_mean = tf.reduce_mean(pre_activity)
        post_mean = tf.reduce_mean(post_activity)
        weight_mean = tf.reduce_mean(self.w)
        features = tf.expand_dims(tf.stack([pre_mean, post_mean, weight_mean]), axis=0)
        scale_factor, bias_adjustment, update_gate = self.plasticity_controller(features)
        meta_mod = meta_signal if meta_signal is not None else 1.0
        delta_w = tf.cast(reward, tf.float32) * meta_mod * (update_gate * (scale_factor + bias_adjustment * self.w))
        delta_delay = tf.cast(reward, tf.float32) * meta_mod * (update_gate * (scale_factor + bias_adjustment * self.delay))
        delta_w = tf.clip_by_norm(delta_w, clip_norm=0.05)
        delta_delay = tf.clip_by_norm(delta_delay, clip_norm=0.05)
        return delta_w, delta_delay

    def apply_homeostatic_scaling(self, meta_modulation=1.0):
        """
        Adjust weights and delays to move toward target averages.
        The meta_modulation factor (default 1.0) can dynamically adjust the scaling.
        """
        avg_w = tf.reduce_mean(tf.abs(self.w))
        new_target = self.decay_factor * self.target_avg + (1 - self.decay_factor) * avg_w
        scaling_factor = new_target / (avg_w + 1e-6)
        scaling_factor *= meta_modulation
        self.w.assign(self.w * scaling_factor)
        self.avg_weight_magnitude.assign(avg_w)
        
        avg_delay = tf.reduce_mean(tf.abs(self.delay))
        new_target_delay = self.decay_factor * self.target_delay + (1 - self.decay_factor) * avg_delay
        scaling_factor_delay = new_target_delay / (avg_delay + 1e-6)
        scaling_factor_delay *= meta_modulation
        self.delay.assign(self.delay * scaling_factor_delay)
        self.avg_delay_magnitude.assign(avg_delay)

    def apply_structural_plasticity(self, meta_modulation=1.0):
        """
        Apply structural plasticity (pruning and connection growth) with meta modulation.
        The effective pruning threshold and add probability are modulated by meta_modulation.
        """
        effective_threshold = self.prune_threshold * meta_modulation
        pruned_w = tf.where(tf.abs(self.w) < effective_threshold, tf.zeros_like(self.w), self.w)
        self.w.assign(pruned_w)
        
        random_matrix = tf.random.uniform(tf.shape(self.w))
        effective_add_prob = self.add_prob * meta_modulation
        add_mask = tf.cast(random_matrix < effective_add_prob, tf.float32)
        new_connections = tf.where(
            tf.logical_and(tf.equal(self.w, 0.0), tf.equal(add_mask, 1.0)),
            tf.random.uniform(tf.shape(self.w), minval=0.01, maxval=0.05),
            self.w)
        self.w.assign(new_connections)
        self.sparsity.assign(tf.reduce_mean(tf.cast(tf.equal(self.w, 0.0), tf.float32)))
        
        self.adjust_prune_threshold(meta_modulation)
        self.adjust_add_prob(meta_modulation)
        
        # Update delays similarly.
        pruned_delay = tf.where(tf.abs(self.delay) < effective_threshold, tf.zeros_like(self.delay), self.delay)
        self.delay.assign(pruned_delay)
        random_matrix_delay = tf.random.uniform(tf.shape(self.delay))
        add_mask_delay = tf.cast(random_matrix_delay < effective_add_prob, tf.float32)
        new_delays = tf.where(
            tf.logical_and(tf.equal(self.delay, 0.0), tf.equal(add_mask_delay, 1.0)),
            tf.random.uniform(tf.shape(self.delay), minval=0.01, maxval=0.05),
            self.delay)
        self.delay.assign(new_delays)

    def adjust_prune_threshold(self, meta_modulation=1.0):
        """
        Dynamically adjust the pruning threshold based on current sparsity.
        The meta_modulation factor further scales the adjustment.
        """
        self.prune_threshold.assign(tf.cond(
            tf.greater(self.sparsity, 0.8),
            lambda: self.prune_threshold * self.prune_min * meta_modulation,
            lambda: tf.cond(tf.less(self.sparsity, 0.3),
                            lambda: self.prune_threshold * self.prune_max * meta_modulation,
                            lambda: self.prune_threshold)
        ))

    def adjust_add_prob(self, meta_modulation=1.0):
        """
        Adjust the probability of adding new connections based on weight magnitude.
        """
        self.add_prob.assign(tf.cond(
            tf.less(self.avg_weight_magnitude, 0.01),
            lambda: self.add_prob * self.prob_min * meta_modulation,
            lambda: tf.cond(tf.greater(self.avg_weight_magnitude, 0.1),
                            lambda: self.add_prob * self.prob_max * meta_modulation,
                            lambda: self.add_prob)
        ))

    def apply_architecture_modification(self, meta_threshold=None):
        """
        Update the neuron mask (i.e. architecture) based on neuron activation averages.
        Optionally, a meta_threshold can override the learned prune_activation_threshold.
        """
        threshold = meta_threshold if meta_threshold is not None else self.prune_activation_threshold
        mask_after_prune = tf.where(
            tf.logical_and(tf.equal(self.neuron_mask, 1.0),
                           tf.less(self.neuron_activation_avg, threshold)),
            tf.zeros_like(self.neuron_mask),
            self.neuron_mask)
        active_values = tf.boolean_mask(self.neuron_activation_avg, tf.equal(mask_after_prune, 1.0))
        growth_cond = tf.cond(
            tf.greater(tf.size(active_values), 0),
            lambda: tf.greater(tf.reduce_mean(active_values), self.growth_activation_threshold),
            lambda: tf.constant(False)
        )
        def grow_neuron():
            inactive_indices = tf.squeeze(tf.where(tf.equal(mask_after_prune, 0.0)), axis=1)
            def activate_random():
                random_idx = tf.random.shuffle(inactive_indices)[0]
                new_mask = tf.tensor_scatter_nd_update(mask_after_prune, [[random_idx]], [1.0])
                return new_mask
            new_mask = tf.cond(tf.greater(tf.shape(inactive_indices)[0], 0),
                               activate_random,
                               lambda: mask_after_prune)
            return new_mask
        new_mask = tf.cond(growth_cond, grow_neuron, lambda: mask_after_prune)
        self.neuron_mask.assign(new_mask)

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.max_units
        return tuple(output_shape)

import tensorflow as tf

class AdaptiveMetaController(tf.keras.layers.Layer):
    def __init__(self, hidden_units=16, **kwargs):
        super(AdaptiveMetaController, self).__init__(**kwargs)
        self.shared_dense = tf.keras.layers.Dense(hidden_units, activation='relu')
        # Now output 10 signals:
        # 0: (unused raw offset if desired)
        # 1: plasticity_mod (sigmoid, in (0,1))
        # 2: homeostasis_mod (sigmoid, in (0,1))
        # 3: structural_mod (sigmoid, in (0,1))
        # 4: arch_threshold_adj (sigmoid, in (0,1))
        # 5: lr_multiplier (0.5 + 0.5 * sigmoid, in (0.5,1.0))
        # 6: homeo_interval (softplus, positive, then clipped)
        # 7: structural_interval (softplus, positive, then clipped)
        # 8: architecture_interval (softplus, positive, then clipped)
        # 9: vectorized_interval (softplus, positive, then clipped)
        self.out_dense = tf.keras.layers.Dense(10, activation=None)

    def call(self, features):
        x = self.shared_dense(features)
        out = self.out_dense(x)
        # Unpack outputs.
        # We ignore output[0] (could be used for a raw offset if needed)
        plasticity_mod = tf.nn.sigmoid(out[:, 1:2])
        homeostasis_mod  = tf.nn.sigmoid(out[:, 2:3])
        structural_mod   = tf.nn.sigmoid(out[:, 3:4])
        arch_threshold_adj = tf.nn.sigmoid(out[:, 4:5])
        lr_multiplier    = 0.5 + 0.5 * tf.nn.sigmoid(out[:, 5:6])
        # For intervals, use softplus to ensure positivity.
        homeo_interval = tf.nn.softplus(out[:, 6:7])
        structural_interval = tf.nn.softplus(out[:, 7:8])
        architecture_interval = tf.nn.softplus(out[:, 8:9])
        vectorized_interval = tf.nn.softplus(out[:, 9:10])
        # Optionally, clip intervals to a reasonable range, e.g. [1, 1000].
        homeo_interval = tf.clip_by_value(homeo_interval, 1.0, 1000.0)
        structural_interval = tf.clip_by_value(structural_interval, 1.0, 1000.0)
        architecture_interval = tf.clip_by_value(architecture_interval, 1.0, 1000.0)
        vectorized_interval = tf.clip_by_value(vectorized_interval, 1.0, 1000.0)
        
        return {
            "plasticity_mod": plasticity_mod,
            "homeostasis_mod": homeostasis_mod,
            "structural_mod": structural_mod,
            "arch_threshold_adj": arch_threshold_adj,
            "lr_multiplier": lr_multiplier,
            "homeo_interval": homeo_interval,
            "structural_interval": structural_interval,
            "architecture_interval": architecture_interval,
            "vectorized_interval": vectorized_interval
        }

    # Utility functions remain similar if needed; for example:
    def get_learning_rate(self, current_lr, features):
        lr_multiplier = self.call(features)["lr_multiplier"]
        new_lr = current_lr * lr_multiplier[0, 0]
        return tf.maximum(new_lr, 1e-6)

class MoE_DynamicPlasticLayer(tf.keras.layers.Layer):
    def __init__(self, num_experts, max_units, initial_units, plasticity_controller, neural_ode_plasticity, **kwargs):
        super(MoE_DynamicPlasticLayer, self).__init__(**kwargs)
        self.num_experts = num_experts
        self.max_units = max_units
        self.initial_units = initial_units
        self.neural_ode_plasticity = neural_ode_plasticity
        self.plasticity_controller = plasticity_controller
        # Create experts using the modified DynamicPlasticDenseDendritic
        self.experts = [
            DynamicPlasticDenseDendritic(self.max_units, self.initial_units, self.plasticity_controller)
            for _ in range(self.num_experts)
        ]
        # Gating network to mix expert outputs.
        self.gating_network = tf.keras.Sequential([
            tf.keras.layers.Dense(self.num_experts * 4, activation='relu', name='moe_dens1'),
            tf.keras.layers.Dense(self.num_experts, activation='softmax', name='moe_dens2')
        ])

    def build(self, input_shape):
        self.clip1 = self.add_weight(
            shape=(), initializer=tf.keras.initializers.Constant(0.05),
            trainable=True, name='clip1_norm'
        )
        self.clip2 = self.add_weight(
            shape=(), initializer=tf.keras.initializers.Constant(0.05),
            trainable=True, name='clip2_norm'
        )
        super(MoE_DynamicPlasticLayer, self).build(input_shape)

    def call(self, inputs, latent=None, meta_forward=None):
        """
        Forward pass.
          - inputs: main input tensor.
          - latent: optional additional feature (e.g. from an episodic memory).
          - meta_forward: optional meta signal (e.g. from the meta-controller) that modulates the experts' forward pass.
        """
        expert_outputs = []
        for expert in self.experts:
            # Pass the meta signal to each expert if provided.
            if meta_forward is not None:
                out = expert(inputs, meta_signal=meta_forward)
            else:
                out = expert(inputs)
            expert_outputs.append(out)
        expert_stack = tf.stack(expert_outputs, axis=-1)  # shape: (batch, features, num_experts)
        gating_input = tf.concat([inputs, latent], axis=-1) if latent is not None else inputs
        gate_weights = self.gating_network(gating_input)  # shape: (batch, num_experts)
        gate_weights = tf.expand_dims(gate_weights, axis=1)  # shape: (batch, 1, num_experts)
        output = tf.reduce_sum(expert_stack * gate_weights, axis=-1)
        return output

    def batch_plasticity_update(self, pre_activity, post_activity, reward, neuromod_signal,
                                scale_factor, bias_adjustment, update_gate, gating_signal,
                                plasticity_meta=None):
        """
        Perform a batched plasticity update on all experts.
          - plasticity_meta: optional scalar modulation signal from the meta-controller.
        """
        # Gather weights and delays from all experts.
        expert_ws = tf.stack([expert.w for expert in self.experts], axis=0)
        expert_delays = tf.stack([expert.delay for expert in self.experts], axis=0)
        
        # Compute update deltas via the neural ODE plasticity module.
        delta_ws = (
            self.neural_ode_plasticity(expert_ws, context=neuromod_signal)
            * gating_signal *
            tf.cast(reward, tf.float32) *
            (update_gate * (scale_factor + bias_adjustment * expert_ws))
        )
        delta_delays = (
            self.neural_ode_plasticity(expert_delays, context=neuromod_signal)
            * gating_signal *
            tf.cast(reward, tf.float32) *
            (update_gate * (scale_factor + bias_adjustment * expert_delays))
        )
        # If provided, scale the updates with the meta plasticity modulation signal.
        if plasticity_meta is not None:
            delta_ws *= plasticity_meta
            delta_delays *= plasticity_meta

        # Clip the computed updates.
        delta_ws = tf.clip_by_norm(delta_ws, clip_norm=self.clip1, axes=[1, 2, 3])
        delta_delays = tf.clip_by_norm(delta_delays, clip_norm=self.clip2, axes=[1, 2, 3])
        
        # Apply the updates to each expert.
        new_expert_ws = expert_ws + delta_ws
        new_expert_delays = expert_delays + delta_delays
        new_ws_list = tf.unstack(new_expert_ws, axis=0)
        new_delays_list = tf.unstack(new_expert_delays, axis=0)
        for i, expert in enumerate(self.experts):
            expert.w.assign(new_ws_list[i])
            expert.delay.assign(new_delays_list[i])
        return delta_ws, delta_delays

class DynamicConnectivity(tf.keras.layers.Layer):
    def __init__(self, max_units, **kwargs):
        super(DynamicConnectivity, self).__init__(**kwargs)
        self.max_units = max_units
        self.act = LearnedQuantumActivation()
        self.attention_net = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation=self.act, name='dc_dens1'),
            tf.keras.layers.Dense(self.max_units, activation='sigmoid', name='dc_dens2')
        ])
        
    def build(self, input_shape):
        super(DynamicConnectivity, self).build(input_shape)

    def call(self, neuron_features, meta_signal=None):
        """
        neuron_features: Tensor containing features (e.g. neuron activation averages).
        meta_signal: Optional scalar or tensor that modulates connectivity.
        """
        connectivity = self.attention_net(neuron_features)
        if meta_signal is not None:
            connectivity = connectivity * meta_signal
        return connectivity

class NeuralODEPlasticity(tf.keras.layers.Layer):
    def __init__(self, n_steps=1, context_dim=1, **kwargs):
        """
        Evolve a state over time using RK4 integration.
        Args:
            n_steps: Number of RK4 steps.
            context_dim: Dimensionality of the context vector.
        """
        super(NeuralODEPlasticity, self).__init__(**kwargs)
        self.n_steps = n_steps
        self.context_dim = context_dim
        self.deriv_net = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='tanh', name='node_dens1'),
            tf.keras.layers.Dense(1, activation='linear', name='node_dens2')
        ])

    def build(self, input_shape):
        self.dt = self.add_weight(
            shape=(),
            initializer=tf.keras.initializers.Constant(0.1),
            trainable=True,
            name='integration_time')
        super(NeuralODEPlasticity, self).build(input_shape)

    def call(self, initial_state, context=None, meta_modulation=None):
        """
        Args:
            initial_state: Tensor representing the state to evolve.
            context: Optional tensor providing context (e.g., neuromodulatory signals).
            meta_modulation: Optional scalar to modulate the derivative.
        """
        orig_shape = tf.shape(initial_state)
        state_flat = tf.reshape(initial_state, [-1])  # Flatten the state.

        dt = self.dt
        h = dt / self.n_steps
        t = tf.constant(0.0, dtype=state_flat.dtype)
        y = state_flat

        if context is not None:
            context = tf.reshape(context, [1, self.context_dim])
            context_tiled = tf.tile(context, [tf.shape(y)[0], 1])
        else:
            context_tiled = None

        def ode_fn(t_val, y_val):
            y_reshaped = tf.reshape(y_val, [-1, 1])
            if context_tiled is not None:
                y_input = tf.concat([y_reshaped, context_tiled], axis=-1)
            else:
                y_input = y_reshaped
            dy = self.deriv_net(y_input)
            if meta_modulation is not None:
                dy = dy * meta_modulation
            return tf.reshape(dy, [-1])
        
        for _ in range(self.n_steps):
            k1 = ode_fn(t, y)
            k2 = ode_fn(t + h/2, y + h/2 * k1)
            k3 = ode_fn(t + h/2, y + h/2 * k2)
            k4 = ode_fn(t + h, y + h * k3)
            y = y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            t += h

        new_state = tf.reshape(y, orig_shape)
        return new_state

class MetacognitiveCritic(tf.keras.layers.Layer):
    def __init__(self, hidden_units=64, **kwargs):
        super(MetacognitiveCritic, self).__init__(**kwargs)
        # Process each modality separately.
        self.latent_dense = tf.keras.layers.Dense(hidden_units // 2, activation='relu', name='critic_episodic')
        self.pred_dense = tf.keras.layers.Dense(hidden_units // 2, activation='relu', name='critic_pred')
        self.hidden_dense = tf.keras.layers.Dense(hidden_units // 2, activation='relu', name='critic_hidden')
        # Fuse all features.
        self.fusion_dense = tf.keras.layers.Dense(hidden_units, activation='relu', name='critic_fusion')
        # Outputs: gating and alignment.
        self.out_dense = tf.keras.layers.Dense(2, activation='linear', name='critic_out')
    
    def build(self, input_shape):
        # A learnable threshold for gating.
        self.meta_threshold = self.add_weight(
            shape=(),
            initializer=tf.keras.initializers.Constant(0.5),
            trainable=True,
            name='meta_threshold'
        )
        super(MetacognitiveCritic, self).build(input_shape)

    def call(self, episodic, predictions, hidden_activation, meta_signal=None):
        """
        Args:
            episodic: Tensor from episodic memory.
            predictions: Model predictions.
            hidden_activation: Hidden layer activations.
            meta_signal: Optional scalar or tensor meta signal that modulates the outputs.
        Returns:
            gating_signal: Sigmoid-modulated gating signal.
            alignment_score: Sigmoid-modulated alignment score.
        """
        mem_feat = self.latent_dense(episodic)
        pred_feat = self.pred_dense(predictions)
        hidden_feat = self.hidden_dense(hidden_activation)
        joint_input = tf.concat([mem_feat, pred_feat, hidden_feat], axis=-1)
        fused = self.fusion_dense(joint_input)
        outputs = self.out_dense(fused)  # Shape: (batch, 2)
        gating_signal = tf.nn.sigmoid(outputs[:, 0:1] - self.meta_threshold)
        alignment_score = tf.nn.sigmoid(outputs[:, 1:2])
        if meta_signal is not None:
            # Optionally modulate both signals.
            gating_signal *= meta_signal
            alignment_score *= meta_signal
        return gating_signal, alignment_score

class ReinforcementSignalLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ReinforcementSignalLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.loss_weight = self.add_weight(
            shape=(),
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=True,
            name="loss_weight"
        )
        self.recon_weight = self.add_weight(
            shape=(),
            initializer=tf.keras.initializers.Constant(0.1),
            trainable=True,
            name="recon_weight"
        )
        self.novelty_weight = self.add_weight(
            shape=(),
            initializer=tf.keras.initializers.Constant(0.5),
            trainable=True,
            name="novelty_weight"
        )
        super(ReinforcementSignalLayer, self).build(input_shape)

    def call(self, loss, recon_loss, novelty, meta_modulation=None):
        """
        Args:
            loss: Primary loss signal.
            recon_loss: Reconstruction loss signal.
            novelty: Novelty measure (e.g., standard deviation of hidden activations).
            meta_modulation: Optional scalar to modulate the overall reward.
        Returns:
            combined_reward: Scalar reinforcement signal.
        """
        loss_signal = self.loss_weight * tf.sigmoid(-loss)
        recon_signal = self.recon_weight * tf.sigmoid(-recon_loss)
        novelty_signal = self.novelty_weight * tf.sigmoid(novelty)
        combined_reward = loss_signal + recon_signal + novelty_signal
        if meta_modulation is not None:
            combined_reward *= meta_modulation
        return combined_reward

class ChaoticModulation(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ChaoticModulation, self).__init__(**kwargs)

    def build(self, input_shape):
        self.r = self.add_weight(
            shape=(),
            initializer=tf.keras.initializers.Constant(3.8),
            trainable=True,
            name='chaotic_r'
        )
        self.plasticity_weight_multiplier = self.add_weight(
            shape=(),
            initializer=tf.keras.initializers.Constant(0.5),
            trainable=True,
            name='plas_w_multi'
        )
        self.reward_scaling_factor = self.add_weight(
            shape=(),
            initializer=tf.keras.initializers.Constant(0.1),
            trainable=True,
            name='reward_scale_fact'
        )
        super(ChaoticModulation, self).build(input_shape)

    def call(self, c, meta_modulation=None):
        """
        Compute chaotic modulation via a logistic map.
        Optionally modulate the result with a meta signal.
        
        Args:
            c: Input scalar or tensor.
            meta_modulation: Optional scalar to further modulate the chaotic factor.
            
        Returns:
            chaotic_factor: The modulated chaotic factor.
        """
        chaotic_factor = self.r * c * (1 - c)
        if meta_modulation is not None:
            chaotic_factor *= meta_modulation
        return chaotic_factor

class EpisodicMemory(tf.keras.layers.Layer):
    def __init__(self, memory_size, memory_dim, **kwargs):
        super(EpisodicMemory, self).__init__(**kwargs)
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        # A dense layer to generate candidate write vectors.
        self.write_layer = tf.keras.layers.Dense(self.memory_dim, activation='tanh', name='em_write')
        # A dense layer to compute attention weights for memory updates.
        self.attention_layer = tf.keras.layers.Dense(self.memory_size, activation='softmax', name='em_attention')        

    def build(self, input_shape):
        # Initialize memory matrix; non-trainable but updated differentiably.
        self.memory = self.add_weight(
            shape=(self.memory_size, self.memory_dim),
            initializer='glorot_uniform',
            trainable=False,
            name='episodic_memory'
        )
        # A learned update gate for memory blending.
        self.update_gate = self.add_weight(
            shape=(self.memory_size,),
            initializer=tf.keras.initializers.Constant(0.5),
            trainable=True, name="update_gate"
        )
        super(EpisodicMemory, self).build(input_shape)

    def call(self, inputs, meta_modulation=None):
        # Read phase: compute attention over memory.
        read_attention = self.attention_layer(inputs)  # shape: (batch, memory_size)
        read_vector = tf.matmul(read_attention, self.memory)  # shape: (batch, memory_dim)

        # Write phase: generate candidate write vectors.
        candidate_write = self.write_layer(inputs)  # shape: (batch, memory_dim)
        aggregated_write = tf.reduce_mean(candidate_write, axis=0)  # shape: (memory_dim,)
        # Reshape aggregated write to (1, memory_dim) so that it broadcasts correctly.
        aggregated_write = tf.reshape(aggregated_write, (1, self.memory_dim))

        # Compute update weights.
        write_attention = tf.reduce_mean(read_attention, axis=0)  # shape: (memory_size,)
        update_weights = write_attention * self.update_gate  # shape: (memory_size,)
        if meta_modulation is not None:
            # Reduce meta_modulation to a scalar.
            meta_modulation = tf.reduce_mean(meta_modulation)
            update_weights *= meta_modulation
        update_weights = tf.expand_dims(update_weights, axis=-1)  # shape: (memory_size, 1)

        # Compute new memory: broadcast aggregated_write to (memory_size, memory_dim).
        new_memory = self.memory * (1 - update_weights) + update_weights * aggregated_write
        # Ensure new_memory has shape (memory_size, memory_dim).
        new_memory = tf.reshape(new_memory, self.memory.shape)
        self.memory.assign(new_memory)

        return read_vector

class PlasticityModelMoE(tf.keras.Model):
    def __init__(self, h, w, units=128, num_experts=1, num_layers=1, max_units=128, 
                 initial_units=32, num_classes=10, memory_size=32, memory_dim=8, **kwargs):
        super(PlasticityModelMoE, self).__init__(**kwargs)
        self.units = units
        self.num_experts = num_experts
        self.max_units = max_units 
        self.initial_units = initial_units
        self.num_classes = num_classes
        self.memory_size = memory_size
        self.memory_dim = memory_dim

        # CNN branch.
        self.cnn1 = tf.keras.layers.Conv1D(units//4, 3, activation='relu', padding='same')
        self.cnn2 = tf.keras.layers.Conv1D(units//2, 3, activation='relu', padding='same')
        self.cnn3 = tf.keras.layers.Conv1D(units, 3, activation='relu', padding='same')
        # Replace fixed dropout with ConcreteDropout wrapping an IdentityLayer.
        self.concrete_dropout = ConcreteDropout(
            layer=IdentityLayer(),
            weight_regularizer=0.0,         # IdentityLayer has no kernel.
            dropout_regularizer=1e-5,
            init_dropout=0.1,
            temperature=0.1
        )
        self.flatten = tf.keras.layers.GlobalAveragePooling1D()

        # Dense layers.
        self.dens = tf.keras.layers.Dense(units, activation='relu', name='pm_dens1')
        self.classification_head = tf.keras.layers.Dense(self.num_classes, activation='softmax', name='pm_dens2')
        self.feature_combiner = tf.keras.layers.Dense(units, activation='relu', name='pm_feature_combiner')

        # Instantiate meta-controller.
        self.meta_controller = AdaptiveMetaController(hidden_units=units)

        # Plasticity control modules.
        self.neural_ode_plasticity = NeuralODEPlasticity(n_steps=10)
        self.hidden = MoE_DynamicPlasticLayer(self.num_experts, self.max_units, self.initial_units, 
                                               self.meta_controller, self.neural_ode_plasticity)
        self.chaotic_modulator = ChaoticModulation()

        # Unsupervised branch.
        self.unsupervised_extractor = UnsupervisedFeatureExtractor(self.units)
        self.reinforcement_layer = ReinforcementSignalLayer()

        # Episodic memory.
        self.episodic_memory = EpisodicMemory(self.memory_size, self.memory_dim)
        self.projected_memory = tf.keras.layers.Dense(10, kernel_initializer='glorot_uniform')

        # Critic module.
        self.critic = MetacognitiveCritic(hidden_units=64)

    def build(self, input_shape):
        # Add trainable parameters for loss weighting.
        self.log_sigma_class = self.add_weight(name="log_sigma_class", shape=(), 
                                               initializer=tf.keras.initializers.Constant(0.0),
                                               trainable=True)
        self.log_sigma_recon = self.add_weight(name="log_sigma_recon", shape=(), 
                                               initializer=tf.keras.initializers.Constant(0.0),
                                               trainable=True)
        self.log_sigma_critic = self.add_weight(name="log_sigma_critic", shape=(), 
                                                initializer=tf.keras.initializers.Constant(0.0),
                                                trainable=True)
        super(PlasticityModelMoE, self).build(input_shape)

    def call(self, x, training=False):
        # # Compute a flattened representation from the input.
        # x_flat = self.flatten(x)  # Shape: (batch, feature_dim)
        # meta_features = x_flat  # Use these features for meta-control.

        # CNN branch.
        cnn1 = self.cnn1(x)
        cnn2 = self.cnn2(cnn1)
        cnn3 = self.cnn3(cnn2)
        drop = self.concrete_dropout(cnn3, training=training)
        x_flat_cnn = self.flatten(drop)

        # Get meta signals from the meta-controller.
        meta_signals = self.meta_controller(x_flat_cnn)
        # Unpack meta signals.
        plasticity_mod = meta_signals["plasticity_mod"]
        homeostasis_mod = meta_signals["homeostasis_mod"]
        structural_mod = meta_signals["structural_mod"]
        arch_threshold_adj = meta_signals["arch_threshold_adj"]
        lr_multiplier = meta_signals["lr_multiplier"]
        homeo_interval = meta_signals["homeo_interval"]
        structural_interval = meta_signals["structural_interval"]
        architecture_interval = meta_signals["architecture_interval"]
        vectorized_interval = meta_signals["vectorized_interval"]       

        # Use the homeostasis_mod (averaged over batch) to modulate episodic memory updates.
        memory_read = self.episodic_memory(x_flat_cnn, meta_modulation=tf.reduce_mean(homeostasis_mod))        

        # Plasticity control branch.
        # combined_input = tf.concat([x_flat_cnn, latent], axis=-1)
        # Pass plasticity_mod to modulate the MoE hidden branch.
        hidden_act = self.hidden(memory_read, meta_forward=plasticity_mod)
        # combined2 = self.feature_combiner(hidden_act)

        # Main classification output.
        class_out = self.classification_head(hidden_act)

        # Critic outputs.
        gating_signal, alignment_score = self.critic(memory_read, class_out, hidden_act)

        # Pack all meta signals into a dictionary for the training loop.
        meta_dict = {
            "plasticity_mod": plasticity_mod,
            "homeostasis_mod": homeostasis_mod,
            "structural_mod": structural_mod,
            "arch_threshold_adj": arch_threshold_adj,
            "lr_multiplier": lr_multiplier,
            "homeo_interval": homeo_interval,
            "structural_interval": structural_interval,
            "architecture_interval": architecture_interval,
            "vectorized_interval": vectorized_interval
        }
        return class_out, gating_signal, alignment_score, meta_dict

# -------------------------
# Helper Functions
# -------------------------
@tf.function
def train_step(model, loss_fn, optimizer, images, labels,
               train_loss_metric, train_accuracy_metric,
               mean_alignment_metric, mean_gating_metric):
    with tf.GradientTape() as tape:
        (predictions, gating_signal, alignment_score, meta_dict) = model(images, training=True)
        classification_loss = loss_fn(labels, predictions)
        critic_loss = tf.reduce_mean(1.0 - alignment_score)
        # Sum regularization losses from all ConcreteDropout layers.
        reg_loss = 0.0
        for layer in model.layers:
            if isinstance(layer, ConcreteDropout):
                reg_loss += layer.get_regularization_loss()
        # Combine losses.
        # Use the learned log_sigma parameters to weight each loss.
        w_class = tf.exp(-model.log_sigma_class)
        w_critic = tf.exp(-model.log_sigma_critic)
        
        weighted_class_loss = w_class * classification_loss + model.log_sigma_class
        weighted_critic_loss = w_critic * critic_loss + model.log_sigma_critic
        
        total_loss = weighted_class_loss + weighted_critic_loss + reg_loss
        
    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Compute an aggregate gradient norm for monitoring.
    grad_norms = [tf.norm(g) for g in grads if g is not None]
    avg_grad_norm = tf.reduce_mean(grad_norms)    
    
    # Update metrics.
    train_loss_metric.update_state(total_loss)
    train_accuracy_metric.update_state(labels, predictions)
    mean_alignment_metric.update_state(alignment_score)
    mean_gating_metric.update_state(gating_signal)
    
    return total_loss, avg_grad_norm, meta_dict

# -------------------------
# Validation Step (Unchanged)
# -------------------------
@tf.function
def val_step(model, loss_fn, val_loss_metric, val_accuracy_metric, val_iter, val_steps):
    for _ in range(val_steps):
        val_images, val_labels = next(val_iter)
        val_predictions, _, _, _ = model(val_images, training=False)
        val_loss = loss_fn(val_labels, val_predictions)
        val_loss_metric.update_state(val_loss)
        val_accuracy_metric.update_state(val_labels, val_predictions)

# -------------------------
# Test Step (Unchanged)
# -------------------------
@tf.function
def test_step(model, loss_fn, test_loss_metric, test_accuracy_metric, test_iter, test_steps):
    for _ in range(test_steps):
        test_images, test_labels = next(test_iter)
        test_predictions, _, _, _ = model(test_images, training=False)
        test_loss = loss_fn(test_labels, test_predictions)
        test_loss_metric.update_state(test_loss)
        test_accuracy_metric.update_state(test_labels, test_predictions)

# -------------------------
# Unified Training Loop
# -------------------------
def train_model(model, model_name, ds_train, ds_val, ds_test, train_steps, val_steps, test_steps,
                num_epochs=1000, initial_lr=1e-3,
                base_homeo_interval=50, base_structural_interval=50, base_architecture_interval=50, base_vectorized_interval=50,
                early_stop_patience=100, early_stop_min_delta=1e-4,
                log_dir="./logs"):
    # Use a single optimizer for all parameters.
    optimizer = tf.keras.optimizers.AdamW(learning_rate=initial_lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    
    # Define metrics.
    train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
    val_loss_metric = tf.keras.metrics.Mean(name="val_loss")
    val_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="val_accuracy")
    test_loss_metric = tf.keras.metrics.Mean(name="test_loss")
    test_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")
    mean_alignment_metric = tf.keras.metrics.Mean(name="mean_alignment_score")
    mean_gating_metric = tf.keras.metrics.Mean(name="mean_gating_signal")
    
    best_val_loss = np.inf
    patience_counter = 0
    best_weights = None
    global_step = 0
    
    writer = tf.summary.create_file_writer(os.path.join(log_dir, model_name))
    
    for epoch in range(num_epochs):
        start_time = time.time()
        train_iter = iter(ds_train)
        val_iter = iter(ds_val)
        test_iter = iter(ds_test)
        
        epoch_grad_norms = []
        for _ in range(train_steps):
            images, labels = next(train_iter)
            total_loss, avg_grad_norm, meta_dict = train_step(model, loss_fn, optimizer, images, labels,
                                                              train_loss_metric, train_accuracy_metric,
                                                              mean_alignment_metric, mean_gating_metric)
            epoch_grad_norms.append(avg_grad_norm)
            global_step += 1        
            
            # Directly use learned dynamic intervals from meta_dict.
            dynamic_homeo_interval = tf.cast(tf.clip_by_value(tf.reduce_mean(meta_dict["homeo_interval"]), 1, 1000), tf.int32)
            dynamic_structural_interval = tf.cast(tf.clip_by_value(tf.reduce_mean(meta_dict["structural_interval"]), 1, 1000), tf.int32)
            dynamic_architecture_interval = tf.cast(tf.clip_by_value(tf.reduce_mean(meta_dict["architecture_interval"]), 1, 1000), tf.int32)
            dynamic_vectorized_interval = tf.cast(tf.clip_by_value(tf.reduce_mean(meta_dict["vectorized_interval"]), 1, 1000), tf.int32)
            lr_multiplier = tf.reduce_mean(meta_dict["lr_multiplier"])
            
            # Conditionally apply each plasticity update.
            if global_step % dynamic_homeo_interval == 0:
                for expert in model.hidden.experts:
                    expert.apply_homeostatic_scaling(meta_modulation=tf.reduce_mean(meta_dict["homeostasis_mod"]))
            
            if global_step % dynamic_structural_interval == 0:
                for expert in model.hidden.experts:
                    expert.apply_structural_plasticity(meta_modulation=tf.reduce_mean(meta_dict["structural_mod"]))
            
            if global_step % dynamic_architecture_interval == 0:
                for expert in model.hidden.experts:
                    expert.apply_architecture_modification(meta_threshold=tf.reduce_mean(meta_dict["arch_threshold_adj"]))
            
            if global_step % dynamic_vectorized_interval == 0:
                # Compute reinforcement signals from combined input and hidden activations.
                reward, pre_activity, post_activity = compute_reinforcement_signals(model, None, None, 0.0, 0.0, 1.0)
                # Also compute neuromodulatory signal from predictions.
                dummy_predictions = tf.zeros((1, model.num_classes))
                base_neuromod_signal = compute_neuromodulatory_signal(dummy_predictions, 0.0)
                # Use plasticity_mod as meta for vectorized update.
                plasticity_meta = tf.reduce_mean(meta_dict["plasticity_mod"])
                for expert in model.hidden.experts:
                    delta_ws, delta_delays = expert.vectorized_plasticity_update(pre_activity, post_activity, reward,
                                                                                  base_neuromod_signal,
                                                                                  meta_signal=plasticity_meta)
                    expert.w.assign_add(delta_ws)
                    expert.delay.assign_add(delta_delays)

            

        # Update learning rate.
        new_lr = initial_lr * lr_multiplier
        optimizer.learning_rate.assign(new_lr)

        epoch_grad_norm = tf.reduce_mean(epoch_grad_norms)                
        
        # Run validation.
        val_step(model, loss_fn, val_loss_metric, val_accuracy_metric, val_iter, val_steps)
        current_val_loss = val_loss_metric.result().numpy()
        
        # Run testing.
        test_step(model, loss_fn, test_loss_metric, test_accuracy_metric, test_iter, test_steps)
        
        tf.print(f"Epoch {epoch+1}/{num_epochs}\nVal Loss: {current_val_loss:.4f} | Val Accuracy: {val_accuracy_metric.result():.4f}")
        tf.print(f"Test Loss: {test_loss_metric.result():.4f} | Test Accuracy: {test_accuracy_metric.result():.4f}")
        tf.print(f"Updated LR: {new_lr:.6e} | Homeo Int: {dynamic_homeo_interval} | Struct Int: {dynamic_structural_interval}\n"
                 f"Arch Int: {dynamic_architecture_interval} | Vec Int: {dynamic_vectorized_interval} | Avg Grad Norm: {epoch_grad_norm:.4f}")
        
        # TensorBoard logging.
        with writer.as_default():
            tf.summary.scalar("train_loss", train_loss_metric.result(), step=epoch)
            tf.summary.scalar("train_accuracy", train_accuracy_metric.result(), step=epoch)
            tf.summary.scalar("val_loss", current_val_loss, step=epoch)
            tf.summary.scalar("val_accuracy", val_accuracy_metric.result(), step=epoch)
            tf.summary.scalar("test_loss", test_loss_metric.result(), step=epoch)
            tf.summary.scalar("test_accuracy", test_accuracy_metric.result(), step=epoch)
            tf.summary.scalar("avg_grad_norm", epoch_grad_norm, step=epoch)
            tf.summary.scalar("lr_multiplier", lr_multiplier, step=epoch)
            tf.summary.scalar("homeo_interval_offset", homeo_offset, step=epoch)
            tf.summary.scalar("structural_interval_offset", structural_offset, step=epoch)
            tf.summary.scalar("architecture_interval_offset", architecture_offset, step=epoch)
            tf.summary.scalar("vectorized_interval_offset", vectorized_offset, step=epoch)
            tf.summary.scalar("dynamic_homeo_interval", tf.cast(dynamic_homeo_interval, tf.float32), step=epoch)
            tf.summary.scalar("dynamic_structural_interval", tf.cast(dynamic_structural_interval, tf.float32), step=epoch)
            tf.summary.scalar("dynamic_architecture_interval", tf.cast(dynamic_architecture_interval, tf.float32), step=epoch)
            tf.summary.scalar("dynamic_vectorized_interval", tf.cast(dynamic_vectorized_interval, tf.float32), step=epoch)
            for layer in model.submodules:
                if isinstance(layer, ConcreteDropout):
                    p = tf.sigmoid(layer.p_logit)
                    tf.summary.scalar(layer.name + "_dropout_p", p, step=epoch)
            meta_names = ["interval_offset", "plasticity_mod", "homeostasis_mod", "structural_mod", "arch_threshold_adj", "lr_multiplier",
                          "homeo_interval_offset", "structural_interval_offset", "architecture_interval_offset"]
            for name in meta_names:
                tf.summary.scalar("meta_" + name, tf.reduce_mean(meta_dict[name]), step=epoch)
                
        # Reset metrics.
        train_loss_metric.reset_state()
        train_accuracy_metric.reset_state()
        val_loss_metric.reset_state()
        val_accuracy_metric.reset_state()
        test_loss_metric.reset_state()
        test_accuracy_metric.reset_state()
        mean_alignment_metric.reset_state()
        mean_gating_metric.reset_state()
        
        # Early stopping.
        if current_val_loss < best_val_loss - early_stop_min_delta:
            best_val_loss = current_val_loss
            patience_counter = 0
            best_weights = model.get_weights()
            tf.print("Validation loss improved; saving model.")
            model.save(f"checkpoints/{model_name}.keras")
        else:
            patience_counter += 1
            tf.print(f"No improvement in validation loss for {patience_counter} epoch(s).")
            if patience_counter >= early_stop_patience:
                if best_weights is not None:
                    tf.print("Restoring best model weights and terminating training.")
                    model.set_weights(best_weights)
                    model.save(f"models/{model_name}.keras")
                break
        
        tf.print(f"Epoch {epoch+1} completed in {time.time()-start_time:.2f}s\n")
# =========================================
# Data Loading and Preprocessing Functions
# =========================================
def get_real_data(num_samples):
    dataset = 'C4L'
    target_df = pd.read_csv(f'datasets/{dataset}_Full.csv')
    cols = ['A', 'B', 'C', 'D', 'E']
    target_df = target_df[cols].dropna().astype(np.int8)
    target_df = target_df.map(lambda x: f'{x:02d}')
    flattened = target_df.values.flatten()
    full_data = np.array([int(d) for num in flattened for d in str(num)], dtype=np.int8)
    return full_data[:num_samples]
    
def get_base_data(num_samples):    
    quick_df = pd.read_csv('datasets/Quick.csv')
    quick_df = quick_df.drop(columns=['Unnamed: 0']).dropna().reset_index(drop=True).astype(np.int8)
    quick_df = quick_df.map(lambda x: f'{x:02d}')
    flattened = quick_df.values.flatten()
    quick_df = np.array([int(d) for num in flattened for d in str(num)], dtype=np.int8)
    return quick_df[:num_samples]

def create_windows(sequence, window_size=65):
    num_windows = len(sequence) - window_size + 1
    windows = np.lib.stride_tricks.as_strided(
        sequence,
        shape=(num_windows, window_size),
        strides=(sequence.strides[0], sequence.strides[0])
    ).copy()
    inputs = windows[:, :-1]
    labels = windows[:, -1]
    return inputs, labels

def split_dataset(inputs, labels, train_frac=0.8, val_frac=0.1):
    total = inputs.shape[0]
    train_end = int(total * train_frac)
    val_end = int(total * (train_frac + val_frac))
    return (inputs[:train_end], labels[:train_end]), (inputs[train_end:val_end], labels[train_end:val_end]), (inputs[val_end:], labels[val_end:])

def create_tf_dataset(inputs, labels, h, w, batch_size=128, shuffle=False):
    inputs = inputs.astype(np.float32) / 9.0
    inputs = inputs.reshape((-1, h, w))
    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size).cache().repeat().prefetch(tf.data.AUTOTUNE)
    return dataset
    
# ============================================================================
# Main Function: Build dataset, instantiate controllers and model, then train
# ============================================================================
def main():
    batch_size = 64
    max_units = 64
    initial_units = 16
    epochs = 1000
    experts = 1
    num_layers = 1
    h = 16
    w = 16
    cnn_units = h * w
    window_size = h * w
    learning_rate = 1e-3
    memory_size = 32
    memory_dim = 8
    num_classes = 10
    
    # Load data.
    sequence = get_base_data(num_samples=10_000)
    input_data, labels = create_windows(sequence, window_size=window_size + 1)
    (inp_train, lbl_train), (inp_val, lbl_val), (inp_test, lbl_test) = split_dataset(input_data, labels)
    
    train_len = inp_train.shape[0]
    val_len = inp_val.shape[0]
    test_len = inp_test.shape[0]
    
    train_steps = math.ceil(train_len / batch_size)
    val_steps = math.ceil(val_len / batch_size)
    test_steps = math.ceil(test_len / batch_size)
    
    ds_train = create_tf_dataset(inp_train, lbl_train, h, w, batch_size, shuffle=False)
    ds_val = create_tf_dataset(inp_val, lbl_val, h, w, batch_size)
    ds_test = create_tf_dataset(inp_test, lbl_test, h, w, batch_size)
    
    # Warm-up dummy inputs.
    dummy_input1 = tf.zeros((1, h, w))
    dummy_input2 = tf.zeros((h, w))
    
    # Instantiate the model.
    model = PlasticityModelMoE(h, w, units=cnn_units, 
                               num_experts=experts, num_layers=num_layers, 
                               max_units=max_units, initial_units=initial_units,
                               num_classes=num_classes, memory_size=memory_size, memory_dim=memory_dim,)
    _ = model(dummy_input1, training=False)
    _ = model.critic(tf.zeros([1, memory_dim]), tf.zeros([1, num_classes]),
                     tf.zeros([1, max_units]), training=False)
    _ = model.meta_controller(tf.zeros([1,cnn_units]), training=False)  # Updated: use meta_controller.
    _ = model.neural_ode_plasticity(tf.zeros([1]), tf.zeros([1]), training=False)
    _ = model.reinforcement_layer(tf.zeros([2]), tf.zeros([2]), tf.zeros([2]), training=False)
    _ = model.chaotic_modulator(tf.zeros([2]), training=False)
    _ = model.hidden.experts[0](tf.random.normal((1, memory_dim)), training=False)
    
    model.summary()
    
    model_name = "MetaSynapse_NextGen_RL_v2g"
    # Start training using the simplified training loop.
    train_model(model, model_name, ds_train, ds_val, ds_test,
                train_steps, val_steps, test_steps,
                num_epochs=epochs, initial_lr=learning_rate)
    
if __name__ == '__main__':
    main()

