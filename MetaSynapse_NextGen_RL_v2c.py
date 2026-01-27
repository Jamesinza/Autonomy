# Silencing annoying warnings
import shutup
shutup.please()

import tensorflow as tf
import pandas as pd
import numpy as np
import math
import random
import time

# Set seeds for reproducibility.
np.random.seed(42)
tf.random.set_seed(42)

# ===================================
# Celestial Alignment inspired layer
# ===================================
class CelestialAlignmentLayer(tf.keras.layers.Layer):
    def __init__(self, units=256, **kwargs):
        """
        Args:
            units: The dimensionality of the modulation vector (e.g. matching feature size).
            period: The period of the cyclic signal (e.g. 12 to mimic zodiac cycles).
        """
        super(CelestialAlignmentLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        # Trainable phase shift and amplitude for each unit.
        self.phase_shift = self.add_weight(
            shape=(self.units,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            name="phase_shift")
        self.amplitude = self.add_weight(
            shape=(self.units,),
            initializer=tf.keras.initializers.Ones(),
            trainable=True,
            name="amplitude")
        self.period = self.add_weight(shape=(),
                                      initializer=tf.keras.initializers.Constant(12.0),
                                      trainable=True,
                                      name='period')
        super(CelestialAlignmentLayer, self).build(input_shape)

    def call(self, inputs):
        '''
        inputs: shape (batch, features)
        Generate a cyclic modulation signal. Instead of relying on tf.timestamp(),
        which can be unstable in training, we use a continuously updated constant.
        Here we simulate a time variable using the training step (or a fixed frequency signal).
        For demonstration, we use a non-trainable sine wave based on a fixed "time" parameter.
        '''
        time_scalar = tf.cast(tf.timestamp(), tf.float32)  # Real time in seconds
        # Create a cyclic signal: sin(2*pi*time/period + phase)
        cyclic_signal = tf.sin(2 * math.pi * time_scalar / self.period + self.phase_shift)
        # Scale it with the amplitude.
        modulation = cyclic_signal * self.amplitude
        # Expand dims to match inputs and modulate features element-wise.
        modulation = tf.reshape(modulation, (1, self.units))
        modulated = inputs * modulation  # Elementwise modulation
        return modulated

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

# ====================== Plasticity Inspired Control Mechanisms Section =======================

# ============================================================================
# Dynamic Connectivity: Using learned activation and unsupervised latent cues
# ============================================================================
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

    def call(self, neuron_features):
        connectivity = self.attention_net(neuron_features)
        return connectivity

# ====================================================
# Dynamic Plastic Dense Layer with Dendritic Branches
# ====================================================
class DynamicPlasticDenseDendritic(tf.keras.layers.Layer):
    def __init__(self, max_units, initial_units, plasticity_controller, num_branches=4, **kwargs):
        super(DynamicPlasticDenseDendritic, self).__init__(name='DynamicPlasticDenseDendritic', **kwargs)
        self.max_units = max_units
        self.initial_units = initial_units
        self.plasticity_controller = plasticity_controller
        self.num_branches = num_branches
        self.act = LearnedQuantumActivation()
        self.dynamic_connectivity = DynamicConnectivity(max_units=self.max_units)
        # Plasticity & homeostasis parameters (non-trainable running stats)
        self.prune_threshold = tf.Variable(0.01, trainable=False, dtype=tf.float32)
        self.add_prob = tf.Variable(0.01, trainable=False, dtype=tf.float32)
        self.sparsity = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.avg_weight_magnitude = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.avg_delay_magnitude = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        # Neuron mask and running average (float32 for stability)
        init_mask = [1.0] * initial_units + [0.0] * (max_units - initial_units)
        self.neuron_mask = tf.Variable(init_mask, trainable=False, dtype=tf.float32)
        self.neuron_activation_avg = tf.Variable(tf.zeros([max_units], dtype=tf.float32), trainable=False)
        # Circular buffer for meta-plasticity features has been removed for full vectorization.
        # Gating mechanism to combine dendritic branch outputs.
        self.branch_gating = tf.keras.layers.Dense(self.num_branches, activation='softmax', name="branch_gating")

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        # Weight: shape (input_dim, max_units, num_branches)
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
        # Make the hard-coded hyperparameters trainable now.
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

    def call(self, inputs):
        # Compute branch outputs.
        w_mod = self.w * tf.math.sigmoid(self.delay)  # (input_dim, max_units, num_branches)
        branch_outputs = tf.tensordot(inputs, w_mod, axes=[[1], [0]])  # (batch, max_units, num_branches)
        branch_outputs += self.b
        gating_weights = self.branch_gating(inputs)  # (batch, num_branches)
        gating_weights = tf.expand_dims(gating_weights, axis=1)  # (batch, 1, num_branches)
        z_combined = tf.reduce_sum(branch_outputs * gating_weights, axis=-1)  # (batch, max_units)
        # Apply dynamic connectivity and neuron mask.
        connectivity = self.dynamic_connectivity(tf.expand_dims(self.neuron_activation_avg, axis=0))
        connectivity = tf.cast(connectivity, z_combined.dtype)
        mask = tf.cast(tf.expand_dims(self.neuron_mask, axis=0), z_combined.dtype)
        z_modulated = z_combined * connectivity * mask
        a = self.act(z_modulated)
        # Update running average (exponential moving average)
        batch_mean = tf.reduce_mean(a, axis=0)
        new_avg = self.alpha * self.neuron_activation_avg + (1 - self.alpha) * tf.cast(batch_mean, self.neuron_activation_avg.dtype)
        self.neuron_activation_avg.assign(new_avg)
        return a

    @tf.function
    def vectorized_plasticity_update(self, pre_activity, post_activity, reward):
        # Compute global statistics.
        pre_mean = tf.reduce_mean(pre_activity)
        post_mean = tf.reduce_mean(post_activity)
        weight_mean = tf.reduce_mean(self.w)
        # Form feature vector: shape (1,3) for meta controller input.
        features = tf.expand_dims(tf.stack([pre_mean, post_mean, weight_mean]), axis=0)
        # Let the meta controller compute scale, bias, and update gate.
        scale_factor, bias_adjustment, update_gate = self.plasticity_controller(features)
        # Compute update deltas for weights and delay in a fully vectorized manner.
        delta_w = tf.cast(reward, tf.float32) * (update_gate * (scale_factor + bias_adjustment * self.w))
        delta_delay = tf.cast(reward, tf.float32) * (update_gate * (scale_factor + bias_adjustment * self.delay))
        # Clip updates.
        delta_w = tf.clip_by_norm(delta_w, clip_norm=0.05)
        delta_delay = tf.clip_by_norm(delta_delay, clip_norm=0.05)
        return delta_w, delta_delay

    # Old implementation no longer used
    # def apply_plasticity(self, plasticity_delta_w, plasticity_delta_delay):
    #     self.w.assign_add(plasticity_delta_w)
    #     self.delay.assign_add(plasticity_delta_delay)

    @tf.function
    def apply_homeostatic_scaling(self):
        avg_w = tf.reduce_mean(tf.abs(self.w))
        new_target = self.decay_factor * self.target_avg + (1 - self.decay_factor) * avg_w
        scaling_factor = new_target / (avg_w + 1e-6)
        self.w.assign(self.w * scaling_factor)
        self.avg_weight_magnitude.assign(avg_w)
        avg_delay = tf.reduce_mean(tf.abs(self.delay))
        new_target_delay = self.decay_factor * self.target_delay + (1 - self.decay_factor) * avg_delay
        scaling_factor_delay = new_target_delay / (avg_delay + 1e-6)
        self.delay.assign(self.delay * scaling_factor_delay)
        self.avg_delay_magnitude.assign(avg_delay)

    @tf.function
    def apply_structural_plasticity(self):
        pruned_w = tf.where(tf.abs(self.w) < self.prune_threshold, tf.zeros_like(self.w), self.w)
        self.w.assign(pruned_w)
        random_matrix = tf.random.uniform(tf.shape(self.w))
        add_mask = tf.cast(random_matrix < self.add_prob, tf.float32)
        new_connections = tf.where(
            tf.logical_and(tf.equal(self.w, 0.0), tf.equal(add_mask, 1.0)),
            tf.random.uniform(tf.shape(self.w), minval=0.01, maxval=0.05),
            self.w)
        self.w.assign(new_connections)
        self.sparsity.assign(tf.reduce_mean(tf.cast(tf.equal(self.w, 0.0), tf.float32)))
        self.adjust_prune_threshold()
        self.adjust_add_prob()
        pruned_delay = tf.where(tf.abs(self.delay) < self.prune_threshold, tf.zeros_like(self.delay), self.delay)
        self.delay.assign(pruned_delay)
        random_matrix_delay = tf.random.uniform(tf.shape(self.delay))
        add_mask_delay = tf.cast(random_matrix_delay < self.add_prob, tf.float32)
        new_delays = tf.where(
            tf.logical_and(tf.equal(self.delay, 0.0), tf.equal(add_mask_delay, 1.0)),
            tf.random.uniform(tf.shape(self.delay), minval=0.01, maxval=0.05),
            self.delay)
        self.delay.assign(new_delays)

    @tf.function
    def adjust_prune_threshold(self):
        self.prune_threshold.assign(tf.cond(
            tf.greater(self.sparsity, 0.8),
            lambda: self.prune_threshold * self.prune_min,
            lambda: tf.cond(tf.less(self.sparsity, 0.3),
                            lambda: self.prune_threshold * self.prune_max,
                            lambda: self.prune_threshold)))

    @tf.function
    def adjust_add_prob(self):
        self.add_prob.assign(tf.cond(
            tf.less(self.avg_weight_magnitude, 0.01),
            lambda: self.add_prob * self.prob_min,
            lambda: tf.cond(tf.greater(self.avg_weight_magnitude, 0.1),
                            lambda: self.add_prob * self.prob_max,
                            lambda: self.add_prob)))

    @tf.function
    def apply_architecture_modification(self):
        mask_after_prune = tf.where(
            tf.logical_and(tf.equal(self.neuron_mask, 1.0),
                           tf.less(self.neuron_activation_avg, self.prune_activation_threshold)),
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

# ==========================
# MoE Dynamic Plastic Layer
# ==========================
class MoE_DynamicPlasticLayer(tf.keras.layers.Layer):
    def __init__(self, num_experts, max_units, initial_units, plasticity_controller, neural_ode_plasticity, **kwargs):
        super(MoE_DynamicPlasticLayer, self).__init__(**kwargs)
        self.num_experts = num_experts
        self.max_units = max_units
        self.initial_units = initial_units
        self.neural_ode_plasticity = neural_ode_plasticity
        self.plasticity_controller = plasticity_controller
        self.experts = [DynamicPlasticDenseDendritic(self.max_units, self.initial_units, self.plasticity_controller)
                        for i in range(self.num_experts)]
        self.gating_network = tf.keras.Sequential([
            tf.keras.layers.Dense(self.num_experts*4, activation='relu', name='moe_dens1'),
            tf.keras.layers.Dense(self.num_experts, activation='softmax', name='moe_dens2')
        ])

    def build(self, input_shape):
        self.clip1 = self.add_weight(
            shape=(),
            initializer=tf.keras.initializers.Constant(0.05),
            trainable=True,
            name='clip1_norm')
        self.clip2 = self.add_weight(
            shape=(),
            initializer=tf.keras.initializers.Constant(0.05),
            trainable=True,
            name='clip2_norm')        
        super(MoE_DynamicPlasticLayer, self).build(input_shape)         

    def call(self, inputs, latent=None):
        # Compute expert outputs in parallel.
        expert_outputs = [expert(inputs) for expert in self.experts]
        expert_stack = tf.stack(expert_outputs, axis=-1)  # (batch, features, num_experts)
        gating_input = tf.concat([inputs, latent], axis=-1) if latent is not None else inputs
        gate_weights = self.gating_network(gating_input)  # (batch, num_experts)
        gate_weights = tf.expand_dims(gate_weights, axis=1)  # (batch, 1, num_experts)
        output = tf.reduce_sum(expert_stack * gate_weights, axis=-1)
        return output

    @tf.function
    def batch_plasticity_update(self, pre_activity, post_activity, reward, neuromod_signal,
                                scale_factor, bias_adjustment, update_gate, gating_signal):
        # Stack weights and delays from all experts.
        expert_ws = tf.stack([tf.convert_to_tensor(expert.w) for expert in self.experts], axis=0)  # shape: (num_experts, input_dim, max_units, num_branches)
        expert_delays = tf.stack([tf.convert_to_tensor(expert.delay) for expert in self.experts], axis=0)
        # Compute global features.
        pre_mean = tf.reduce_mean(pre_activity)
        post_mean = tf.reduce_mean(post_activity)
        weight_mean = tf.reduce_mean(expert_ws)
        # (MetaPlasticityController is applied externally.)
        # Compute update deltas using the provided meta parameters.
        delta_ws = (self.neural_ode_plasticity(expert_ws, context=neuromod_signal) * gating_signal *
                    tf.cast(reward, tf.float32) * (update_gate * (scale_factor + bias_adjustment * expert_ws)))
        delta_delays = (self.neural_ode_plasticity(expert_delays, context=neuromod_signal) * gating_signal *
                        tf.cast(reward, tf.float32) * (update_gate * (scale_factor + bias_adjustment * expert_delays)))
        # Clip updates.
        delta_ws = tf.clip_by_norm(delta_ws, clip_norm=self.clip1, axes=[1,2,3])
        delta_delays = tf.clip_by_norm(delta_delays, clip_norm=self.clip2, axes=[1,2,3])
        # Update experts' weights vectorized.
        new_expert_ws = expert_ws + delta_ws  # * neuromod_signal
        new_expert_delays = expert_delays + delta_delays  # * neuromod_signal
        new_ws_list = tf.unstack(new_expert_ws, axis=0)
        new_delays_list = tf.unstack(new_expert_delays, axis=0)
        for i, expert in enumerate(self.experts):
            expert.w.assign(new_ws_list[i])
            expert.delay.assign(new_delays_list[i])
        return delta_ws, delta_delays

# ===========================
# Meta-Plasticity Controller
# ===========================
class MetaPlasticityController(tf.keras.layers.Layer):
    def __init__(self, input_dim=3, units=32, **kwargs):
        super(MetaPlasticityController, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.units = units
        self.dense1 = tf.keras.layers.Dense(self.units, activation='relu', input_shape=(self.input_dim,), name='mpc_dens1')
        # Output three scalars:
        #   - scale_factor,
        #   - bias_adjustment,
        #   - raw_update_logit (to be gated with a learnable threshold)
        self.dense2 = tf.keras.layers.Dense(3, activation='linear', name='mpc_dens2')
    
    def build(self, input_shape):
        self.update_threshold = self.add_weight(
            shape=(),
            initializer=tf.keras.initializers.Constant(0.5),
            trainable=True,
            name='update_threshold')        
        super(MetaPlasticityController, self).build(input_shape)        

    def call(self, plasticity_features):
        x = self.dense1(plasticity_features)
        adjustments = self.dense2(x)
        # Split outputs:
        # adjustments[:, 0:1] -> scale_factor,
        # adjustments[:, 1:2] -> bias_adjustment,
        # adjustments[:, 2:3] -> raw_update_logit.
        scale_factor = adjustments[:, 0:1]
        bias_adjustment = adjustments[:, 1:2]
        raw_update_logit = adjustments[:, 2:3]
        # Compute a soft update gate: subtract the learnable threshold and apply sigmoid.
        update_gate = tf.sigmoid(raw_update_logit - self.update_threshold)
        return scale_factor, bias_adjustment, update_gate

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.input_dim
        return tuple(output_shape)

# ====================================================================
# Using Neural Ordinary Differential Equations to modulate Plasticity
# ====================================================================
class NeuralODEPlasticity(tf.keras.layers.Layer):
    def __init__(self, n_steps=10, context_dim=1, **kwargs):
        """
        Evolve a state over time using RK4 integration.
        Args:
            n_steps: Number of RK4 steps.
            context_dim: Dimensionality of the context vector.
        """
        super(NeuralODEPlasticity, self).__init__(**kwargs)
        self.n_steps = n_steps
        self.context_dim = context_dim
        # Derivative network: expects input shape (N, 1 + context_dim)
        self.deriv_net = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='tanh', name='node_dens1'),
            tf.keras.layers.Dense(1, activation='linear', name='node_dens2')
        ])

    def build(self, input_shape):
        # dt is now a trainable parameter.
        self.dt = self.add_weight(
            shape=(),
            initializer=tf.keras.initializers.Constant(0.1),
            trainable=True,
            name='integration_time')
        super(NeuralODEPlasticity, self).build(input_shape)

    def call(self, initial_state, context=None):
        # Save original shape and flatten the state.
        orig_shape = tf.shape(initial_state)
        state_flat = tf.reshape(initial_state, [-1])  # shape: (N,)

        dt = self.dt
        h = dt / self.n_steps
        t = tf.constant(0.0, dtype=state_flat.dtype)
        y = state_flat

        # Precompute tiled context if provided.
        if context is not None:
            context = tf.reshape(context, [1, self.context_dim])
            # Tile context for each element in y.
            context_tiled = tf.tile(context, [tf.shape(y)[0], 1])
        else:
            context_tiled = None

        # Define the derivative function.
        @tf.function
        def ode_fn(t_val, y_val):
            # Reshape y to (N, 1) for processing.
            y_reshaped = tf.reshape(y_val, [-1, 1])
            # If context is provided, concatenate it.
            if context_tiled is not None:
                y_input = tf.concat([y_reshaped, context_tiled], axis=-1)  # shape: (N, 1+context_dim)
            else:
                y_input = y_reshaped
            dy = self.deriv_net(y_input)  # shape: (N, 1)
            return tf.reshape(dy, [-1])
        
        # RK4 integration loop.
        for _ in range(self.n_steps):
            k1 = ode_fn(t, y)
            k2 = ode_fn(t + h/2, y + h/2 * k1)
            k3 = ode_fn(t + h/2, y + h/2 * k2)
            k4 = ode_fn(t + h, y + h * k3)
            y = y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            t += h

        # Reshape y back to the original state shape.
        new_state = tf.reshape(y, orig_shape)
        return new_state

# ====================== Quantum Inspired Control Mechanisms Section =======================

# ===================================================
# Differentiable Quantum-inspired Circuit simulation
# ===================================================
class DifferentiableQuantumCircuit(tf.keras.layers.Layer):
    def __init__(self, num_qubits=4, num_layers=1, **kwargs):
        """
        Args:
            num_qubits: Number of qubits in the simulated circuit.
            num_layers: How many layers (circuit blocks) to apply.
        """
        super(DifferentiableQuantumCircuit, self).__init__(**kwargs)
        self.num_qubits = num_qubits
        self.state_dim = 2 ** num_qubits
        self.num_layers = num_layers
        # For each layer, for each qubit, parameterize three angles (theta, phi, lambda).
        self.thetas = []
        self.phis = []
        self.lams = []

    def build(self, input_shape):
        for l in range(self.num_layers):
            self.thetas.append(self.add_weight(
                shape=(self.num_qubits,),
                initializer=tf.keras.initializers.RandomUniform(0, 2 * 3.1416),
                trainable=True,
                name=f"theta_{l}"))
            self.phis.append(self.add_weight(
                shape=(self.num_qubits,),
                initializer=tf.keras.initializers.RandomUniform(0, 2 * 3.1416),
                trainable=True,
                name=f"phi_{l}"))
            self.lams.append(self.add_weight(
                shape=(self.num_qubits,),
                initializer=tf.keras.initializers.RandomUniform(0, 2 * 3.1416),
                trainable=True,
                name=f"lambda_{l}"))
        # Optionally, we can add fixed entangling layers between single-qubit rotations.        
        super(DifferentiableQuantumCircuit, self).build(input_shape)         

    def single_qubit_gate(self, theta, phi, lam):
        """
        Construct a single-qubit U3 gate.
           U3(theta, phi, lambda) = 
             [[cos(theta/2), -exp(1j*lam)*sin(theta/2)],
              [exp(1j*phi)*sin(theta/2), exp(1j*(phi+lam))*cos(theta/2)]]
        """
        # Compute cos and sin in real domain.
        c = tf.math.cos(theta / 2)
        s = tf.math.sin(theta / 2)
        # Compute complex exponentials: use tf.complex to create complex numbers.
        exp_i_phi = tf.exp(tf.complex(tf.zeros_like(phi), phi))
        exp_i_lam = tf.exp(tf.complex(tf.zeros_like(lam), lam))
        exp_i_phi_plus_lam = tf.exp(tf.complex(tf.zeros_like(phi + lam), phi + lam))
        # Create complex representations of cos and sin.
        c_complex = tf.complex(c, tf.zeros_like(c))
        s_complex = tf.complex(s, tf.zeros_like(s))
        # Construct the U3 gate as a 2x2 complex matrix.
        row1 = tf.stack([c_complex, -exp_i_lam * s_complex])
        row2 = tf.stack([exp_i_phi * s_complex, exp_i_phi_plus_lam * c_complex])
        gate = tf.stack([row1, row2])
        return gate

    def build_full_gate(self, theta_vec, phi_vec, lam_vec):
        """Build the full circuit unitary for one layer by taking the tensor product
           of single-qubit gates.
        """
        gates = []
        for q in range(self.num_qubits):
            gate = self.single_qubit_gate(theta_vec[q], phi_vec[q], lam_vec[q])
            gates.append(gate)
        # Compute tensor (Kronecker) product across all qubits.
        full_gate = gates[0]
        for gate in gates[1:]:
            full_gate = tf.linalg.LinearOperatorKronecker([
                tf.linalg.LinearOperatorFullMatrix(full_gate),
                tf.linalg.LinearOperatorFullMatrix(gate)
            ]).to_dense()
        return full_gate

    def call(self, inputs):
        """
        Args:
            inputs: Tensor of shape (batch, state_dim). Expected to be real amplitudes.
        Returns:
            A tensor representing the output quantum state probabilities.
        """
        # Ensure input is a complex state.
        inputs_complex = tf.complex(inputs, tf.zeros_like(inputs))
        # Normalize each state to unit norm.
        norm = tf.linalg.norm(inputs_complex, axis=1, keepdims=True)
        state = inputs_complex / norm
        # Process through each circuit layer.
        for l in range(self.num_layers):
            full_gate = self.build_full_gate(self.thetas[l], self.phis[l], self.lams[l])
            # Apply the unitary transformation to each state in the batch.
            state = tf.map_fn(lambda x: tf.linalg.matvec(full_gate, x), state)
            # Optional: insert an entangling operation here (e.g., a fixed permutation gate).
        # Return measurement probabilities (|amplitudes|^2).
        output = tf.math.real(state * tf.math.conj(state))
        return output

    def compute_output_shape(self, input_shape):
        return tuple(input_shape)         

# ====================================
# Quantum Entanglement inspired Layer
# ====================================
class RefinedQuantumEntanglementLayer(tf.keras.layers.Layer):
    def __init__(self, num_qubits=4, **kwargs):
        """
        Quantum-inspired entanglement layer that operates purely in the real domain.
        
        Args:
            max_num_qubits: Maximum number of units.
                The unit size MUST be 2**max_num_qubits.
                (eg, 2,4,8,16,32,64...)                
        """
        super().__init__(**kwargs)
        self.max_state_size = 2 ** num_qubits

    def build(self, input_shape):
        if input_shape[-1] != self.max_state_size:
            raise ValueError(f"Input dimension {input_shape[-1]} must equal 2**max_num_qubits = {self.max_state_size}\n "
            f"Either match units to input shape of {input_shape[-1]} or adjust previous layers output dim at axis[-1]")
        # Real-valued representation of a complex matrix (split into real and imaginary parts)
        self.M_real = self.add_weight(
            shape=(self.max_state_size, self.max_state_size),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
            trainable=True,
            name="M_real",
            dtype=tf.float32)
        self.M_imag = self.add_weight(
            shape=(self.max_state_size, self.max_state_size),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
            trainable=True,
            name="M_imag",
            dtype=tf.float32)
        # Bias term (real-valued)
        self.bias = self.add_weight(
            shape=(self.max_state_size,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            name="bias",
            dtype=tf.float32)
        # Learnable gating vector for adaptive qubit usage
        self.qubit_gate = self.add_weight(
            shape=(self.max_state_size,),
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=True,
            name="qubit_gate",
            dtype=tf.float32)
        super().build(input_shape)

    def call(self, inputs):
        # Normalize inputs to unit norm (simulating a quantum state)
        norm_inputs = tf.linalg.l2_normalize(inputs, axis=-1)
        # Construct a skew-symmetric real-valued transformation matrix
        A = self.M_real - tf.transpose(self.M_real)  # Skew-symmetric
        B = self.M_imag - tf.transpose(self.M_imag)  # Skew-symmetric
        U = tf.linalg.expm(A) @ tf.linalg.expm(B)  # Exponential maps ensure a stable transformation
        # Transform the state and apply bias
        transformed = tf.matmul(norm_inputs, U) + self.bias
        # Compute squared magnitudes (simulating measurement probabilities)
        measured = tf.square(transformed)
        # Dynamic gating via softmax on the learned gate vector
        gate_probs = tf.nn.softmax(self.qubit_gate)
        gated_output = measured * gate_probs
        # Return the final real-valued output and gate probabilities
        return gated_output, gate_probs

    def compute_output_shape(self, input_shape):
        return tuple(input_shape)

# ============================
# Metacognitive Critic Module
# ============================
class MetacognitiveCritic(tf.keras.layers.Layer):
    def __init__(self, hidden_units=64, **kwargs):
        super(MetacognitiveCritic, self).__init__(**kwargs)
        # Process each modality separately.
        self.latent_dense = tf.keras.layers.Dense(hidden_units // 2, activation='relu', name='critic_latent')
        self.pred_dense = tf.keras.layers.Dense(hidden_units // 2, activation='relu', name='critic_pred')
        self.hidden_dense = tf.keras.layers.Dense(hidden_units // 2, activation='relu', name='critic_hidden')
        self.uncertainty_dense = tf.keras.layers.Dense(hidden_units // 2, activation='relu', name='critic_uncertainty')
        # Fuse all features.
        self.fusion_dense = tf.keras.layers.Dense(hidden_units, activation='relu', name='critic_fusion')
        # Outputs: gating and alignment.
        self.out_dense = tf.keras.layers.Dense(2, activation='linear', name='critic_out')
    
    def build(self, input_shape):
        # A learnable threshold for gating can be shared or separate.
        self.meta_threshold = self.add_weight(
            shape=(), initializer=tf.keras.initializers.Constant(0.5),
            trainable=True,
            name='meta_threshold')
        super(MetacognitiveCritic, self).build(input_shape)

    def call(self, latent, predictions, hidden_activation, uncertainty):
        # Process each modality.
        latent_feat = self.latent_dense(latent)
        pred_feat = self.pred_dense(predictions)
        hidden_feat = self.hidden_dense(hidden_activation)
        uncertainty_feat = self.uncertainty_dense(uncertainty)
        # Concatenate all features.
        joint_input = tf.concat([latent_feat, pred_feat, hidden_feat, uncertainty_feat], axis=-1)
        fused = self.fusion_dense(joint_input)
        outputs = self.out_dense(fused)  # (batch, 2)
        # Split outputs into gating and alignment signals.
        gating_signal = tf.nn.sigmoid(outputs[:, 0:1] - self.meta_threshold)
        alignment_score = tf.nn.sigmoid(outputs[:, 1:2])
        return gating_signal, alignment_score
        
# =================================
# Chaos-inspired Modulation Module
# =================================
class ChaoticModulation(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ChaoticModulation, self).__init__(**kwargs)

    def build(self, input_shape):
        # Register r as a trainable scalar parameter.
        self.r = self.add_weight(
            shape=(),
            initializer=tf.keras.initializers.Constant(3.8),
            trainable=True,
            name='chaotic_r')
        self.plasticity_weight_multiplier = self.add_weight(
            shape=(),
            initializer=tf.keras.initializers.Constant(0.5),
            trainable=True,
            name='plas_w_multi')
        self.reward_scaling_factor = self.add_weight(
            shape=(),
            initializer=tf.keras.initializers.Constant(0.1),
            trainable=True,
            name='reward_scale_fact')
        super(ChaoticModulation, self).build(input_shape)

    # @tf.function
    def call(self, c):
        # Logistic map: new value = r * c * (1 - c)
        chaotic_factor = self.r * c * (1 - c)
        return chaotic_factor

# ===================================
# Reinforcment style signal learning
# ===================================
class ReinforcementSignalLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ReinforcementSignalLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Initialize with your current hard-coded values.
        self.loss_weight = self.add_weight(
            shape=(), initializer=tf.keras.initializers.Constant(1.0),
            trainable=True, name="loss_weight")
        self.recon_weight = self.add_weight(
            shape=(), initializer=tf.keras.initializers.Constant(0.1),
            trainable=True, name="recon_weight")
        self.novelty_weight = self.add_weight(
            shape=(), initializer=tf.keras.initializers.Constant(0.5),
            trainable=True, name="novelty_weight")
        self.uncertainty_weight = self.add_weight(
            shape=(), initializer=tf.keras.initializers.Constant(0.05),
            trainable=True, name="uncertainty_weight")
        super(ReinforcementSignalLayer, self).build(input_shape)

    def call(self, loss, recon_loss, novelty, avg_uncertainty):
        # Compute each component using the trainable weights.
        loss_signal = self.loss_weight * tf.sigmoid(-loss)
        recon_signal = self.recon_weight * tf.sigmoid(-recon_loss)
        novelty_signal = self.novelty_weight * tf.sigmoid(novelty)
        uncertainty_signal = self.uncertainty_weight * tf.sigmoid(avg_uncertainty)
        combined_reward = loss_signal + recon_signal + novelty_signal + uncertainty_signal
        return combined_reward

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
        self.dec_output = tf.keras.layers.Conv1D(8, 3, activation="sigmoid", padding="same", name="dec_output")        

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

# =======================================
# Episodic Memory for Continual Learning
# =======================================
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

    def call(self, inputs):
        # Read: compute attention over memory and generate read vector.
        read_attention = self.attention_layer(inputs)  # shape: (batch, memory_size)
        read_vector = tf.matmul(read_attention, self.memory)  # shape: (batch, memory_dim)

        # Write: generate candidate write vector.
        candidate_write = self.write_layer(inputs)  # shape: (batch, memory_dim)
        # Aggregate write vector across the batch.
        aggregated_write = tf.reduce_mean(candidate_write, axis=0)  # shape: (memory_dim,)

        # Compute update weights for all memory slots based on input-derived attention.
        write_attention = tf.reduce_mean(read_attention, axis=0)  # shape: (memory_size,)
        # Combine with learned update gate.
        update_weights = write_attention * self.update_gate  # shape: (memory_size,)
        update_weights = tf.expand_dims(update_weights, axis=-1)  # shape: (memory_size, 1)

        # Perform a differentiable update: blend the current memory with the aggregated write.
        self.memory.assign(self.memory * (1 - update_weights) + update_weights * aggregated_write)

        return read_vector
        
# ===============================================================================
# Full Model with Integrated Unsupervised Branch and Dynamic Plastic Dense Layer
# ===============================================================================
class PlasticityModelMoE(tf.keras.Model):
    def __init__(self, h, w, units=128, num_experts=1, num_layers=1, max_units=128, 
                 initial_units=32, num_classes=10, memory_size=10, memory_dim=64, **kwargs):
        super(PlasticityModelMoE, self).__init__(**kwargs)
        self.units = units
        # self.plasticity_controller = plasticity_controller
        self.num_experts = num_experts
        self.max_units = max_units 
        self.initial_units = initial_units
        self.num_classes = num_classes
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.act = LearnedQuantumActivation()
        # Classic CNN based network layers
        self.cnn1 = tf.keras.layers.Conv1D(units//4, 3, activation='relu', padding='same')
        self.cnn2 = tf.keras.layers.Conv1D(units//2, 3, activation='relu', padding='same')
        self.cnn3 = tf.keras.layers.Conv1D(units, 3, activation='relu', padding='same')
        self.drop = tf.keras.layers.Dropout(0.0)
        self.flatten = tf.keras.layers.GlobalAveragePooling1D()
        self.dens = tf.keras.layers.Dense(units, activation='relu', name='pm_dens1')
        self.classification_head = tf.keras.layers.Dense(self.num_classes, activation='softmax', name='pm_dens2')
        # Instantiate the quantum inspired layers.
        # Project hidden activations to dimension 16 (for 4 qubits: 2**4 = 16)
        self.quantum_dense = tf.keras.layers.Dense(16, activation=self.act, name='quantum_dense')
        self.quantum_branch = RefinedQuantumEntanglementLayer(num_qubits=4, name="QuantumEntanglement")
        self.quantum_circuit = DifferentiableQuantumCircuit(num_qubits=4, num_layers=1, name="DifferentiableQuantumCircuit")
        # Instantiate plasticity control layers
        self.neural_ode_plasticity = NeuralODEPlasticity(n_steps=10)
        self.meta_plasticity_controller = MetaPlasticityController(units=self.units)
        self.hidden = MoE_DynamicPlasticLayer(self.num_experts, self.max_units, self.initial_units, self.meta_plasticity_controller, self.neural_ode_plasticity)        
        self.chaotic_modulator = ChaoticModulation()
        # Instantiate other custom layers
        self.unsupervised_extractor = UnsupervisedFeatureExtractor(self.units)
        self.reinforcement_layer = ReinforcementSignalLayer()
        self.celestial_alignment = CelestialAlignmentLayer(units=self.units, name="CelestialAlignment")
        self.feature_combiner = tf.keras.layers.Dense(self.units, activation='relu', name='pm_dens2')
        self.uncertainty_head = tf.keras.layers.Dense(1, activation='sigmoid', name='pm_dens3')
        # Project episodic_memory to dimension D.
        self.episodic_memory = EpisodicMemory(self.memory_size, self.memory_dim)
        self.projected_memory = tf.keras.layers.Dense(10, kernel_initializer='glorot_uniform')
        # Instantiate the Critic.
        self.critic = MetacognitiveCritic(hidden_units=64)

    def build(self, input_shape):
        # Initialize loss weights as trainable variables.
        self.recon_loss_weight = self.add_weight(
            shape=(),
            initializer=tf.keras.initializers.Constant(0.1),
            trainable=True,
            name="recon_loss_weight")
        self.uncertainty_loss_weight = self.add_weight(
            shape=(),
            initializer=tf.keras.initializers.Constant(0.05),
            trainable=True,
            name="uncertainty_loss_weight")
        super(PlasticityModelMoE, self).build(input_shape)        

    # @tf.function
    def call(self, x, training=False):
        latent, reconstruction = self.unsupervised_extractor(x)
        memory_read = self.episodic_memory(latent)
        projected_memory = self.projected_memory(memory_read)
        cnn1 = self.cnn1(x)
        cnn2 = self.cnn2(cnn1)
        cnn3 = self.cnn3(cnn2)
        drop = self.drop(cnn3, training=training)
        x_flat = self.flatten(cnn3)
        dens = self.dens(x_flat)

        # --- Plasticity control branch ---
        combined_input = tf.concat([x_flat, latent, memory_read], axis=-1)
        hidden_act = self.hidden(combined_input, latent)
        
        # --- Quantum Branch Processing ---
        quantum_input = self.quantum_dense(hidden_act)
        quantum_out1, gate_probs = self.quantum_branch(quantum_input)
        quantum_out2 = self.quantum_circuit(quantum_input)
        quantum_out = tf.concat([quantum_out1, quantum_out2], axis=-1)
        # Compute quantum entropy as a feedback signal.
        eps = 1e-6
        quantum_entropy = -tf.reduce_sum(quantum_out1 * tf.math.log(quantum_out1 + eps), axis=-1, keepdims=True)
        
        combined1 = tf.concat([latent, memory_read, hidden_act, quantum_out], axis=-1)
        combined2 = self.feature_combiner(combined1)
        
        # --- Apply the celestial branch to modulate the fused features.
        celestial_modulated = self.celestial_alignment(combined2)

         # --- Main outputs ---        
        uncertainty = self.uncertainty_head(celestial_modulated)
        class_out = tf.concat([celestial_modulated, dens], axis=-1)
        class_out = self.classification_head(class_out)

        # --- Critic outputs ---
        gating_signal, alignment_score = self.critic(latent, class_out, hidden_act, quantum_entropy)
        
        # Most of these return values are for feedback purposes.
        return class_out, combined_input, hidden_act, reconstruction, uncertainty, latent, quantum_entropy, gate_probs, projected_memory, gating_signal, alignment_score 

# =================
# Helper Functions
# =================
# Integrated training step: one forward/backward pass for both model and critic.
@tf.function
def train_step(model, loss_fn, recon_loss_fn, optimizer, ae_optimizer,
               critic_optimizer, images, labels, plasticity_weight_val):
    with tf.GradientTape(persistent=True) as tape:
        # Forward pass through the base model.
        (predictions, combined_input, hidden, reconstruction, uncertainty,
         latent, quantum_entropy, gate_probs, episodic_memory,
         gating_signal, alignment_score) = model(images, training=True)
        
        # Compute main losses.
        loss_value = loss_fn(labels, predictions)
        recon_loss = recon_loss_fn(images, reconstruction)
        mean_uncertainty = tf.reduce_mean(uncertainty)
        uncertainty_loss = tf.reduce_mean(tf.square(uncertainty - mean_uncertainty))

    # Update the base model.
    model_grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(model_grads, model.trainable_variables))
    # Update autoencoder.
    ae_grads = tape.gradient(recon_loss, model.unsupervised_extractor.trainable_variables)
    ae_optimizer.apply_gradients(zip(ae_grads, model.unsupervised_extractor.trainable_variables))  
    # # Update the critic network.
    critic_grads = tape.gradient(alignment_score, model.critic.trainable_variables)
    critic_optimizer.apply_gradients(zip(critic_grads, model.critic.trainable_variables))
    del tape
    return (loss_value, recon_loss, predictions, combined_input,
            hidden, latent, quantum_entropy, gate_probs, episodic_memory, gating_signal, alignment_score)

@tf.function
def val_step(model, loss_fn, recon_loss_fn, val_loss_metric, val_accuracy_metric,
             val_recon_loss_metric, val_iter, val_steps):
    for step in range(val_steps):
        val_images, val_labels = next(val_iter)
        val_predictions, _, _, val_reconstruction, _, _, _, _, _, _, _ = model(val_images, training=False)
        val_loss = loss_fn(val_labels, val_predictions)
        val_recon_loss = recon_loss_fn(val_images, val_reconstruction)
        val_loss_metric.update_state(val_loss)
        val_accuracy_metric.update_state(val_labels, val_predictions)
        val_recon_loss_metric.update_state(val_recon_loss)        

@tf.function
def test_step(model, loss_fn, recon_loss_fn, test_loss_metric, test_accuracy_metric,
              test_recon_loss_metric, test_iter, test_steps):
    for step in range(test_steps):
        test_images, test_labels = next(test_iter)
        test_predictions, _, _, test_reconstruction, _, _, _, _, _, _, _ = model(test_images, training=False)
        test_loss = loss_fn(test_labels, test_predictions)
        test_recon_loss = recon_loss_fn(test_images, test_reconstruction)
        test_loss_metric.update_state(test_loss)
        test_accuracy_metric.update_state(test_labels, test_predictions)
        test_recon_loss_metric.update_state(test_recon_loss)

@tf.function
def compute_reinforcement_signals(model, combined_input, hidden, quantum_entropy, loss_value,
                                  recon_loss, global_reward_scaling_factor):
    pre_activity = tf.reduce_mean(combined_input, axis=0)
    post_activity = tf.reduce_mean(hidden, axis=0)
    novelty = tf.math.reduce_std(hidden)
    avg_uncertainty = tf.reduce_mean(quantum_entropy)
    combined_reward = model.reinforcement_layer(loss_value, recon_loss, novelty, avg_uncertainty)
    reward = combined_reward * global_reward_scaling_factor
    return reward, pre_activity, post_activity

@tf.function
def scheduled_plasticity_updates(model, global_plasticity_weight_multiplier, global_reward_scaling_factor,
                                 pre_activity, post_activity, gating_signal, reward, neuromod_signal):
    chaotic_factor = model.chaotic_modulator(global_plasticity_weight_multiplier.numpy())
    global_plasticity_weight_multiplier.assign(chaotic_factor)
    global_reward_scaling_factor.assign(chaotic_factor)
    # Create a plasticity feature vector for meta plasticity controller.
    all_ws = tf.stack([tf.convert_to_tensor(expert.w) for expert in model.hidden.experts], axis=0) # Shape: (num_experts, ...)
    avg_weight = tf.reduce_mean(tf.abs(all_ws))
    plasticity_features = tf.stack([tf.reduce_mean(pre_activity),
                                    tf.reduce_mean(post_activity),
                                    avg_weight], axis=0)
    plasticity_features = tf.expand_dims(plasticity_features, axis=0)  # shape: (1,3)
    scale_factor, bias_adjustment, update_gate = model.meta_plasticity_controller(plasticity_features)
    gate_modulation = tf.reduce_mean(gating_signal)
    # Apply the batched plasticity update across all experts.
    delta_ws, delta_delays = model.hidden.batch_plasticity_update(pre_activity, post_activity, reward,
                                                                   neuromod_signal,
                                                                   scale_factor, bias_adjustment, update_gate, gate_modulation)
    return delta_ws, delta_delays

@tf.function
def compute_neuromodulatory_signal(predictions, external_reward=0.0):
    # Force computations to float32.
    predictions = tf.cast(predictions, tf.float32)
    eps = tf.constant(1e-6, dtype=tf.float32)
    entropy = -tf.reduce_sum(predictions * tf.math.log(predictions + eps), axis=-1)
    modulation = tf.sigmoid(tf.cast(external_reward, tf.float32) + tf.reduce_mean(entropy))
    return modulation

# @tf.function    
def compute_latent_mod(inputs):
    latent_modulator = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', name='latmod_dens1'),
        tf.keras.layers.Dense(1, activation='sigmoid', name='latmod_dens2')
    ])(inputs)
    return latent_modulator

# Define the Self-Supervised Alignment Loss
def critic_alignment_loss(episodic_memory, current_predictions):
    """
    Encourage the episodic memory readouts to align with current predictions.
    Here we use cosine similarity (1 - cosine similarity) as a loss.
    """
    # Normalize along the feature axis.
    norm_memory = tf.nn.l2_normalize(episodic_memory, axis=-1)
    norm_pred = tf.nn.l2_normalize(current_predictions, axis=-1)
    cosine_sim = tf.reduce_sum(norm_memory * norm_pred, axis=-1)
    # Loss is 1 minus the cosine similarity (averaged over the batch).
    loss = tf.reduce_mean(1.0 - cosine_sim)
    return loss

# ===================
# Main Training Loop
# ===================
def train_model(model, model_name, ds_train, ds_val, ds_test, train_steps, val_steps, test_steps,
                num_epochs=1000, homeostasis_interval=10, architecture_update_interval=2,
                plasticity_update_interval=10, plasticity_start_epoch=1,
                early_stop_patience=10, early_stop_min_delta=1e-4, learning_rate=1e-3,
                critic_loss_weight=0.1):
    # Set up optimizers.
    optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate)
    ae_optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate)
    critic_optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate)

    # Set up loss functions
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    recon_loss_fn = tf.keras.losses.Huber()

    # Define standard metrics.
    train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    val_loss_metric = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
    test_loss_metric = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    train_recon_loss_metric = tf.keras.metrics.Mean(name='train_recon_loss')
    val_recon_loss_metric = tf.keras.metrics.Mean(name='val_recon_loss')
    test_recon_loss_metric = tf.keras.metrics.Mean(name='test_recon_loss')
    mean_alignment_metric = tf.keras.metrics.Mean(name="mean_alignment_score")
    mean_gating_metric = tf.keras.metrics.Mean(name="mean_gating_signal")

    # Global plasticity parameters.
    global_plasticity_weight_multiplier = tf.Variable(0.5, trainable=True, dtype=tf.float32)
    global_reward_scaling_factor = tf.Variable(0.1, trainable=True, dtype=tf.float32)
    global_step = tf.Variable(0, dtype=tf.int64)
    best_val_loss = np.inf
    best_val_recon_loss = np.inf
    patience_counter = 0
    recon_patience_counter = 10
    best_weights = None
    recon_best_weights = None

    for epoch in range(num_epochs):
        start_time = time.time()
        train_iter = iter(ds_train)
        val_iter = iter(ds_val)
        test_iter = iter(ds_test)
        plasticity_weight_val = 0.0
        if epoch >= plasticity_start_epoch:
            plasticity_weight_val = ((epoch - plasticity_start_epoch + 1) /
                                     (num_epochs - plasticity_start_epoch + 1)
                                    ) * global_plasticity_weight_multiplier.numpy()

        for step in range(train_steps):
            images, labels = next(train_iter)
            (loss_value, recon_loss, predictions, combined_input,
             hidden, latent, quantum_entropy, gate_probs, episodic_memory,
             gating_signal, alignment_score) = train_step(model, loss_fn, recon_loss_fn, optimizer, ae_optimizer,
                                                          critic_optimizer, images, labels, plasticity_weight_val)

            # Update metrics.
            train_loss_metric.update_state(loss_value)
            train_accuracy_metric.update_state(labels, predictions)
            train_recon_loss_metric.update_state(recon_loss)
            mean_alignment_metric.update_state(alignment_score)
            mean_gating_metric.update_state(gating_signal)

            # Log every 100 steps.
            if step % 10 == 0:
                print(f"Training loss at step {step}/{train_steps}: {float(loss_value):.4f}")

            # Compute reinforcement signals and neuromodulatory factors.
            reward, pre_activity, post_activity = compute_reinforcement_signals(model, combined_input, hidden, quantum_entropy,
                                                   loss_value, recon_loss, global_reward_scaling_factor)
            base_neuromod_signal = compute_neuromodulatory_signal(predictions, reward)
            latent_signal = tf.reduce_mean(compute_latent_mod(latent))
            neuromod_signal = base_neuromod_signal * latent_signal            
            # Apply plasticity updates if scheduled.
            if plasticity_weight_val > 0 and tf.equal(global_step % plasticity_update_interval, 0):
                delta_ws, delta_delays = scheduled_plasticity_updates(model, global_plasticity_weight_multiplier,
                                                                      global_reward_scaling_factor, pre_activity,
                                                                      post_activity, gating_signal, reward, neuromod_signal)

            if tf.equal(global_step % homeostasis_interval, 0):
                for expert in model.hidden.experts:
                    expert.apply_homeostatic_scaling()
                    expert.apply_structural_plasticity()

            if tf.equal(global_step % architecture_update_interval, 0):
                for expert in model.hidden.experts:
                    expert.apply_architecture_modification()

            global_step.assign_add(1)
        
        # Epoch logging.
        print(f"\nEpoch {epoch+1}/{num_epochs}\n"
              f"Train Loss : {train_loss_metric.result():.4f}, "
              f"Train Accuracy : {train_accuracy_metric.result():.4f}, "
              f"Train Recon Loss : {train_recon_loss_metric.result():.4f}, "
              f"Critic Alignment Score: {mean_alignment_metric.result():.6f}, "
              f"Critic Gating Signal  : {mean_gating_metric.result():.6f}")            
        train_loss_metric.reset_state()
        train_accuracy_metric.reset_state()
        train_recon_loss_metric.reset_state()

        # Validation Evaluation.
        val_step(model, loss_fn, recon_loss_fn, val_loss_metric, val_accuracy_metric,
                 val_recon_loss_metric, val_iter, val_steps)
        current_val_loss = val_loss_metric.result().numpy()
        current_val_recon_loss = val_recon_loss_metric.result().numpy()
        print(f"Val Loss   : {current_val_loss:.4f}, "
              f"Val Accuracy   : {val_accuracy_metric.result():.4f}, "
              f"Val Recon Loss : {val_recon_loss_metric.result():.4f}")
        val_loss_metric.reset_state()
        val_accuracy_metric.reset_state()
        val_recon_loss_metric.reset_state()

        # Test Evaluation.
        test_step(model, loss_fn, recon_loss_fn, test_loss_metric, test_accuracy_metric,
                  test_recon_loss_metric, test_iter, test_steps)
        print(f"Test Loss  : {test_loss_metric.result():.4f}, "
              f"Test Accuracy  : {test_accuracy_metric.result():.4f}, "
              f"Test Recon Loss : {test_recon_loss_metric.result():.4f}")
        test_loss_metric.reset_state()
        test_accuracy_metric.reset_state()
        test_recon_loss_metric.reset_state()

        # Early stopping AutoEncoder
        if current_val_recon_loss < best_val_recon_loss - early_stop_min_delta:
            best_val_recon_loss = current_val_recon_loss
            recon_counter = 0
            recon_best_weights = model.unsupervised_extractor.get_weights()
            print("\nRecon validation loss improved; resetting recon counter.")
            model.unsupervised_extractor.save(f"checkpoints/extractors/{model_name}.keras")
        else:
            recon_counter += 1
            print(f"\nNo improvement in recon validation loss for {recon_counter} epoch(s).")
            if recon_counter >= recon_patience_counter:
                if recon_best_weights is not None:
                    print("Restoring recon best weights and saving model.\n")
                    model.unsupervised_extractor.set_weights(recon_best_weights)
                    model.unsupervised_extractor.save(f"models/extractors/{model_name}.keras")        

        # Early Stopping.
        if current_val_loss < best_val_loss - early_stop_min_delta:
            best_val_loss = current_val_loss
            patience_counter = 0
            best_weights = model.get_weights()
            print("\nValidation loss improved; resetting patience counter.")
            model.save(f"checkpoints/{model_name}.keras")
        else:
            patience_counter += 1
            print(f"\nNo improvement in validation loss for {patience_counter} epoch(s).")
            if patience_counter >= early_stop_patience:
                if best_weights is not None:
                    print("Restoring best weights and saving model.\n")
                    model.set_weights(best_weights)
                    model.save(f"models/{model_name}.keras")
                break
        print(f"\nTime taken: {time.time() - start_time:.2f}s\n")

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
    batch_size = 4096
    max_units = 64
    initial_units = 16
    epochs = 1000
    experts = 8
    num_layers = 8
    h = 16
    w = 16
    cnn_units = h*w
    window_size = h * w
    learning_rate = 1e-2
    
    # Load data.
    sequence = get_base_data(num_samples=1_000_000)
    input_data, labels = create_windows(sequence, window_size=window_size+1)
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
    
    # Warm-ups
    dummy_input1 = tf.zeros((1, h, w))
    dummy_input2 = tf.zeros((h, w))    
    # Instantiate the Model.
    model = PlasticityModelMoE(h, w, units=cnn_units, 
                           num_experts=experts, num_layers=num_layers, max_units=max_units, initial_units=initial_units, num_classes=10)
    _ = model(dummy_input1, training=False)
    _ = model.critic(tf.zeros([1,256]),tf.zeros([1,10]),tf.zeros([1,64]),tf.zeros([1,1]),training=False)
    # _ = plasticity_controller(tf.zeros([1,1,4]), training=False)
    _ = model.meta_plasticity_controller(tf.zeros([1,3]), training=False)
    _ = model.neural_ode_plasticity(tf.zeros([1]),tf.zeros([1]), training=False)
    _ = model.reinforcement_layer(tf.zeros([2]),tf.zeros([2]),tf.zeros([2]),tf.zeros([2]),training=False)
    _ = model.chaotic_modulator(tf.zeros([2]), training=False)
    _ = model.hidden.experts[0](tf.random.normal((1, 384)), training=False)
    model.summary()
    # model.load_weights("model_weights/MetaSynapse_NextGen_RL_v2/model.weights.h5", skip_mismatch=True)
    model_name = "MetaSynapse_NextGen_RL_v2c"
    train_model(model, model_name, ds_train, ds_val, ds_test, train_steps, val_steps, test_steps,
                num_epochs=epochs, homeostasis_interval=13, architecture_update_interval=34,
                plasticity_update_interval=5, plasticity_start_epoch=3, learning_rate=learning_rate)
    
if __name__ == '__main__':
    main()

