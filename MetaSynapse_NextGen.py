# Silencing annoying warnings
import shutup
shutup.please()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # '2' hides INFO and WARNING messages

import tensorflow as tf
import pandas as pd
import numpy as np
import math
import random

# Temporary Disable all GPUs
# tf.config.set_visible_devices([], 'GPU')

# --- Enable Mixed Precision (if GPU is available) ---
# if tf.config.list_physical_devices('GPU'):
#     tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Set seeds for reproducibility.
np.random.seed(42)
tf.random.set_seed(42)


import tensorflow as tf

class NeuralODEPlasticity(tf.keras.layers.Layer):
    def __init__(self, dt=0.1, n_steps=10, **kwargs):
        """
        Args:
            dt: Total integration time.
            n_steps: Number of integration steps for RK4.
        """
        super(NeuralODEPlasticity, self).__init__(**kwargs)
        self.dt = dt
        self.n_steps = n_steps
        # A simple neural network to parameterize the derivative.
        # Adjust the architecture as needed.
        self.deriv_net = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='tanh'),
            # Output dimension should match the flattened state size (here, assumed 1 per element).
            tf.keras.layers.Dense(1, activation='linear')
        ])

    def call(self, initial_state, context=None):
        """
        Args:
            initial_state: Tensor representing the state to be evolved (e.g., weight matrix).
            context: Optional additional context (e.g., neuromodulatory signal).
        Returns:
            Evolved state with the same shape as initial_state.
        """
        orig_shape = tf.shape(initial_state)
        state_flat = tf.reshape(initial_state, [-1])
        t0 = tf.constant(0.0, dtype=state_flat.dtype)
        dt = self.dt
        n_steps = self.n_steps
        h = dt / n_steps
        t = t0
        y = state_flat

        def ode_fn(t, y):
            # Optionally incorporate context.
            if context is not None:
                context_flat = tf.reshape(context, [-1])
                y_input = tf.concat([y, context_flat], axis=0)
            else:
                y_input = y
            # Expand dims for compatibility with dense layers.
            y_input = tf.expand_dims(y_input, axis=0)
            dy = self.deriv_net(y_input)
            return tf.squeeze(dy, axis=0)

        # Fixed-step RK4 integration.
        for _ in range(n_steps):
            k1 = ode_fn(t, y)
            k2 = ode_fn(t + h/2, y + h/2 * k1)
            k3 = ode_fn(t + h/2, y + h/2 * k2)
            k4 = ode_fn(t + h, y + h * k3)
            y = y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            t += h

        new_state = tf.reshape(y, orig_shape)
        return new_state



class MetaPlasticityController(tf.keras.layers.Layer):
    def __init__(self, units=32, **kwargs):
        super(MetaPlasticityController, self).__init__(**kwargs)
        self.dense1 = tf.keras.layers.Dense(units, activation='relu')
        # Let’s say we output three scalars:
        #   - scale_factor: how much to scale the plasticity delta.
        #   - bias_adjustment: a small bias to add to the update.
        #   - update_prob: a value (via sigmoid) that gates whether to update.
        self.dense2 = tf.keras.layers.Dense(3, activation='linear')  # raw outputs

    def call(self, plasticity_features):
        x = self.dense1(plasticity_features)
        adjustments = self.dense2(x)
        # Optionally, apply a sigmoid to one of the outputs (e.g., update probability)
        update_prob = tf.sigmoid(adjustments[:, 2:3])
        # Return scale, bias, and update probability
        return adjustments[:, 0:1], adjustments[:, 1:2], update_prob


class CelestialAlignmentLayer(tf.keras.layers.Layer):
    def __init__(self, units=256, period=12.0, **kwargs):
        """
        Args:
            units: The dimensionality of the modulation vector (e.g. matching feature size).
            period: The period of the cyclic signal (e.g. 12 to mimic zodiac cycles).
        """
        super(CelestialAlignmentLayer, self).__init__(**kwargs)
        self.units = units
        self.period = period  # The cycle period (e.g., 12 zodiac signs)

    def build(self, input_shape):
        # Trainable phase shift and amplitude for each unit.
        self.phase_shift = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name="phase_shift"
        )
        self.amplitude = self.add_weight(
            shape=(self.units,),
            initializer='ones',
            trainable=True,
            name="amplitude"
        )
        super(CelestialAlignmentLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs: shape (batch, features)
        # Generate a cyclic modulation signal. Instead of relying on tf.timestamp(),
        # which can be unstable in training, we use a continuously updated constant.
        # Here we simulate a time variable using the training step (or a fixed frequency signal).
        # For demonstration, we use a non-trainable sine wave based on a fixed "time" parameter.
        time_scalar = tf.cast(tf.timestamp(), tf.float32)  # Real time in seconds
        # Create a cyclic signal: sin(2*pi*time/period + phase)
        cyclic_signal = tf.sin(2 * math.pi * time_scalar / self.period + self.phase_shift)
        # Scale it with the amplitude.
        modulation = cyclic_signal * self.amplitude
        # Expand dims to match inputs and modulate features element-wise.
        modulation = tf.reshape(modulation, (1, self.units))
        modulated = inputs * modulation  # Elementwise modulation
        return modulated


class RefinedQuantumEntanglementLayer(tf.keras.layers.Layer):
    def __init__(self, units=256, **kwargs):
        """
        Quantum-inspired entanglement layer that operates purely in the real domain.
        
        Args:
            max_num_qubits: Maximum number of units.
                The unit size MUST be 2**max_num_qubits.
                (eg, 2,4,8,16,32,64...)                
        """
        super().__init__(**kwargs)
        self.max_num_qubits = int(np.log2(units))
        self.max_state_size = 2 ** self.max_num_qubits

    def build(self, input_shape):
        if input_shape[-1] != self.max_state_size:
            raise ValueError(f"Input dimension {input_shape[-1]} must equal 2**max_num_qubits = {self.max_state_size}\n "
            f"Either match units to input shape of {input_shape[-1]} or adjust previous layers output dim at axis[-1]")
        # Real-valued representation of a complex matrix (split into real and imaginary parts)
        self.M_real = self.add_weight(
            name="M_real",
            shape=(self.max_state_size, self.max_state_size),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
            dtype=tf.float32,
            trainable=True
        )
        self.M_imag = self.add_weight(
            name="M_imag",
            shape=(self.max_state_size, self.max_state_size),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
            dtype=tf.float32,
            trainable=True
        )
        # Bias term (real-valued)
        self.bias = self.add_weight(
            name="bias",
            shape=(self.max_state_size,),
            initializer=tf.keras.initializers.Zeros(),
            dtype=tf.float32,
            trainable=True
        )
        # Learnable gating vector for adaptive qubit usage
        self.qubit_gate = self.add_weight(
            name="qubit_gate",
            shape=(self.max_state_size,),
            initializer=tf.keras.initializers.Constant(1.0),
            dtype=tf.float32,
            trainable=True
        )
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


class ComplexRandomNormal(tf.keras.initializers.Initializer):
    def __init__(self, stddev=0.1):
        self.stddev = stddev

    def __call__(self, shape, dtype=None, **kwargs):
        # Generate real and imaginary parts with float32.
        real = tf.random.normal(shape, stddev=self.stddev, dtype=tf.float32)
        imag = tf.random.normal(shape, stddev=self.stddev, dtype=tf.float32)
        return tf.complex(real, imag)

    def get_config(self):
        return {"stddev": self.stddev}


# =============================================================================
# 1. Learned Activations combined with Quantum-Inspired Stochastic Activation
#    Dynamically adjust weights based on input context
# =============================================================================
class LearnedQuantumActivation(tf.keras.layers.Layer):
    def __init__(self, temperature=0.5, **kwargs):
        super(LearnedQuantumActivation, self).__init__(**kwargs)
        self.temperature = temperature  # Controls the noise intensity
        
    def build(self, input_shape):
        self.w = self.add_weight(name='activation_weights', shape=(9,),
                                 initializer='ones', trainable=True)
        super(LearnedQuantumActivation, self).build(input_shape)
    
    def call(self, inputs, training=True):
        # Compute multiple activations
        sig = tf.keras.activations.sigmoid(inputs)
        elu = tf.keras.activations.elu(inputs)
        tanh = tf.keras.activations.tanh(inputs)
        relu = tf.keras.activations.relu(inputs)
        silu = tf.keras.activations.silu(inputs)
        gelu = tf.keras.activations.gelu(inputs)
        selu = tf.keras.activations.selu(inputs)
        mish = tf.keras.activations.mish(inputs)
        linear = 0.0  # Acts as a "do-nothing" activation
        # Compute softmax-weighted activation mixture
        weights = tf.nn.softmax(self.w)
        base_activation = (weights[0]*sig + weights[1]*elu + weights[2]*tanh +
                           weights[3]*relu + weights[4]*silu + weights[5]*gelu +
                           weights[6]*selu + weights[7]*mish + weights[8]*linear)
        # Generate stochastic noise (quantum-inspired phase noise)
        phase_noise = tf.random.normal(tf.shape(inputs)) * self.temperature

        # Apply noise based on the sign of base activation
        if training:  # Apply stochasticity only during training
            noisy_activation = tf.where(base_activation > 0, 
                                        base_activation + phase_noise, 
                                        base_activation - tf.abs(phase_noise))
        else:
            noisy_activation = base_activation  # No noise during inference
        return noisy_activation


# =============================================================================
# 2. Recurrent Plasticity Controller with Self-Attention
# =============================================================================
class RecurrentPlasticityController(tf.keras.layers.Layer):
    def __init__(self, units=256, sequence_length=10, **kwargs):
        super(RecurrentPlasticityController, self).__init__(**kwargs)
        self.sequence_length = sequence_length
        self.units = units
        self.act = LearnedQuantumActivation()
        
    def build(self, input_shape):
        self.gru = tf.keras.layers.GRU(self.units, return_sequences=False)
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=self.units)
        self.dense = tf.keras.layers.Dense(2, activation=self.act, name='rpc_dens')
        super(RecurrentPlasticityController, self).build(input_shape)
        
    # @tf.function
    def call(self, input_sequence):
        attn_output = self.attention(input_sequence, input_sequence)
        x = self.gru(attn_output)
        adjustments = self.dense(x)
        return adjustments

# =============================================================================
# 3. Unsupervised Feature Extractor (Autoencoder with U-Net style architecture)
# =============================================================================
class UnsupervisedFeatureExtractor(tf.keras.Model):
    def __init__(self, units=256, **kwargs):
        super().__init__(**kwargs)
        # Encoder layers
        self.enc_cnn1 = tf.keras.layers.Conv2D(units//2, (3, 3), activation="relu", padding="same", name="enc_cnn1")
        self.enc_pool1 = tf.keras.layers.MaxPooling2D((2, 2), padding="same", name="enc_pool1")
        self.enc_cnn2 = tf.keras.layers.Conv2D(units, (3, 3), activation="relu", padding="same", name="enc_cnn2")
        self.enc_pool2 = tf.keras.layers.MaxPooling2D((2, 2), padding="same", name="enc_pool2")
        self.enc_latent = tf.keras.layers.Conv2D(units * 2, (3, 3), activation="relu", padding="same", name="enc_latent")
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")
        # Decoder layers
        self.dec_upconv1 = tf.keras.layers.Conv2DTranspose(units, (3, 3), strides=2, activation="relu", padding="same", name="dec_upconv1")
        self.dec_concat1 = tf.keras.layers.Concatenate(name="dec_concat1")
        self.dec_upconv2 = tf.keras.layers.Conv2DTranspose(units//2, (3, 3), strides=2, activation="relu", padding="same", name="dec_upconv2")
        self.dec_output = tf.keras.layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same", name="dec_output")

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

# =============================================================================
# 4. Dynamic Connectivity: Using learned activation and unsupervised latent cues.
# =============================================================================
class DynamicConnectivity(tf.keras.layers.Layer):
    def __init__(self, max_units, **kwargs):
        super(DynamicConnectivity, self).__init__(**kwargs)
        self.max_units = max_units
        self.act = LearnedQuantumActivation()
        
    def build(self, input_shape):
        self.attention_net = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation=self.act, name='dc_dens1'),
            tf.keras.layers.Dense(self.max_units, activation='sigmoid', name='dc_dens2')
        ])
        super(DynamicConnectivity, self).build(input_shape)
    
    def call(self, neuron_features):
        connectivity = self.attention_net(neuron_features)
        return connectivity

# =============================================================================
# 5. Dynamic Plastic Dense Layer with Dendritic Branches
# =============================================================================
class DynamicPlasticDenseDendritic(tf.keras.layers.Layer):
    def __init__(self, max_units, initial_units, plasticity_controller, num_branches=4,
                 decay_factor=0.9, prune_activation_threshold=0.01, growth_activation_threshold=0.8, **kwargs):
        super(DynamicPlasticDenseDendritic, self).__init__(**kwargs)
        self.max_units = max_units
        self.initial_units = initial_units
        self.plasticity_controller = plasticity_controller
        self.num_branches = num_branches
        self.decay_factor = decay_factor
        self.prune_activation_threshold = prune_activation_threshold
        self.growth_activation_threshold = growth_activation_threshold
        self.act = LearnedQuantumActivation()

        self.dynamic_connectivity = DynamicConnectivity(max_units=self.max_units)
        
        # Plasticity & homeostasis parameters.
        self.prune_threshold = tf.Variable(0.01, trainable=False, dtype=tf.float32)
        self.add_prob = tf.Variable(0.01, trainable=False, dtype=tf.float32)
        self.sparsity = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.avg_weight_magnitude = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.target_avg = tf.Variable(0.2, trainable=False, dtype=tf.float32)
        self.avg_delay_magnitude = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.target_delay = tf.Variable(0.2, trainable=False, dtype=tf.float32)
        
        # Neuron mask and running average (stored as float32 for stability).
        init_mask = [1] * initial_units + [0] * (max_units - initial_units)
        self.neuron_mask = tf.Variable(init_mask, trainable=False, dtype=tf.float32)
        self.neuron_activation_avg = tf.Variable(tf.zeros([max_units], dtype=tf.float32), trainable=False)
        
        # In-graph circular buffer for meta-plasticity features.
        seq_len = self.plasticity_controller.sequence_length
        self.feature_history = self.add_weight(
            shape=(seq_len, 4), initializer='zeros', trainable=False, name='feature_history'
        )
        self.feature_history_index = tf.Variable(0, trainable=False, dtype=tf.int32)

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        # Weight: shape (input_dim, max_units, num_branches)
        self.w = self.add_weight(shape=(input_dim, self.max_units, self.num_branches),
                                 initializer='glorot_uniform',
                                 trainable=True, name='dpd_kernel')
        # Bias: shape (max_units, num_branches)
        self.b = self.add_weight(shape=(self.max_units, self.num_branches),
                                 initializer='zeros',
                                 trainable=True, name='dpd_bias')
        # Delay for each branch.
        self.delay = self.add_weight(shape=(input_dim, self.max_units, self.num_branches),
                                     initializer='zeros',
                                     trainable=True, name='delay')
        # Gating mechanism to combine dendritic branch outputs.
        self.branch_gating = tf.keras.layers.Dense(self.num_branches, activation='softmax', name="branch_gating")
        super(DynamicPlasticDenseDendritic, self).build(input_shape)

    def call(self, inputs):
        # Vectorized branch computation.
        w_mod = self.w * tf.math.sigmoid(self.delay)  # (input_dim, max_units, num_branches)
        branch_outputs = tf.tensordot(inputs, w_mod, axes=[[1], [0]])  # (batch, max_units, num_branches)
        branch_outputs += self.b  # Bias broadcasted
        
        gating_weights = self.branch_gating(inputs)  # (batch, num_branches)
        gating_weights = tf.expand_dims(gating_weights, axis=1)  # (batch, 1, num_branches)
        z_combined = tf.reduce_sum(branch_outputs * gating_weights, axis=-1)  # (batch, max_units)
        
        # Apply dynamic connectivity and neuron mask (cast to match z_combined dtype).
        connectivity = self.dynamic_connectivity(tf.expand_dims(self.neuron_activation_avg, axis=0))
        connectivity = tf.cast(connectivity, z_combined.dtype)
        mask = tf.cast(tf.expand_dims(self.neuron_mask, axis=0), z_combined.dtype)
        z_modulated = z_combined * connectivity * mask
        
        a = self.act(z_modulated)
        
        # Update running average of activations.
        batch_mean = tf.reduce_mean(a, axis=0)
        alpha = tf.cast(0.9, self.neuron_activation_avg.dtype)
        new_avg = alpha * self.neuron_activation_avg + (1 - alpha) * tf.cast(batch_mean, self.neuron_activation_avg.dtype)
        self.neuron_activation_avg.assign(new_avg)
        return a

    def plasticity_update(self, pre_activity, post_activity, reward):
        pre_mean = tf.reduce_mean(pre_activity)
        post_mean = tf.reduce_mean(post_activity)
        weight_mean = tf.reduce_mean(self.w)
        new_feature = tf.stack([pre_mean, post_mean, weight_mean, tf.cast(reward, tf.float32)])
        
        seq_len = self.plasticity_controller.sequence_length
        idx = self.feature_history_index % seq_len
        updated_history = tf.tensor_scatter_nd_update(self.feature_history, [[idx]], [new_feature])
        self.feature_history.assign(updated_history)
        self.feature_history_index.assign_add(1)
        
        # Only update if sufficient history is accumulated.
        def no_update():
            return (tf.zeros_like(self.w), tf.zeros_like(self.delay))
        
        def do_update():
            adjustments = self.plasticity_controller(tf.expand_dims(self.feature_history, axis=0))  # (1, 2)
            delta_add, delta_mult = adjustments[0, 0], adjustments[0, 1]
            plasticity_delta_w = tf.cast(reward, tf.float32) * (delta_add + delta_mult * self.w)
            plasticity_delta_delay = tf.cast(reward, tf.float32) * (delta_add + delta_mult * self.delay)
            return plasticity_delta_w, plasticity_delta_delay
        
        return tf.cond(tf.greater_equal(self.feature_history_index, seq_len),
                       do_update, no_update)
    
    def apply_plasticity(self, plasticity_delta_w, plasticity_delta_delay):
        self.w.assign_add(plasticity_delta_w)
        self.delay.assign_add(plasticity_delta_delay)
    
    def apply_homeostatic_scaling(self):
        avg_w = tf.reduce_mean(tf.abs(self.w))
        new_target = self.decay_factor * self.target_avg + (1 - self.decay_factor) * avg_w
        scaling_factor = new_target / (avg_w + tf.cast(1e-6, avg_w.dtype))
        self.w.assign(self.w * scaling_factor)
        self.avg_weight_magnitude.assign(avg_w)
        
        avg_delay = tf.reduce_mean(tf.abs(self.delay))
        new_target_delay = self.decay_factor * self.target_delay + (1 - self.decay_factor) * avg_delay
        scaling_factor_delay = new_target_delay / (avg_delay + tf.cast(1e-6, avg_delay.dtype))
        self.delay.assign(self.delay * scaling_factor_delay)
        self.avg_delay_magnitude.assign(avg_delay)
    
    def apply_structural_plasticity(self):
        pruned_w = tf.where(tf.abs(self.w) < self.prune_threshold, tf.zeros_like(self.w), self.w)
        self.w.assign(pruned_w)
        random_matrix = tf.random.uniform(tf.shape(self.w))
        add_mask = tf.cast(random_matrix < self.add_prob, tf.float32)
        new_connections = tf.where(tf.logical_and(tf.equal(self.w, 0.0), tf.equal(add_mask, 1.0)),
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
        new_delays = tf.where(tf.logical_and(tf.equal(self.delay, 0.0), tf.equal(add_mask_delay, 1.0)),
                               tf.random.uniform(tf.shape(self.delay), minval=0.01, maxval=0.05),
                               self.delay)
        self.delay.assign(new_delays)
    
    def adjust_prune_threshold(self):
        self.prune_threshold.assign(tf.cond(tf.greater(self.sparsity, 0.8),
                                              lambda: self.prune_threshold * 1.05,
                                              lambda: tf.cond(tf.less(self.sparsity, 0.3),
                                                              lambda: self.prune_threshold * 0.95,
                                                              lambda: self.prune_threshold)))
    
    def adjust_add_prob(self):
        self.add_prob.assign(tf.cond(tf.less(self.avg_weight_magnitude, 0.01),
                                       lambda: self.add_prob * 1.05,
                                       lambda: tf.cond(tf.greater(self.avg_weight_magnitude, 0.1),
                                                       lambda: self.add_prob * 0.95,
                                                       lambda: self.add_prob)))

    def apply_architecture_modification(self):
        mask_after_prune = tf.where(
            tf.logical_and(tf.equal(self.neuron_mask, 1.0),
                           tf.less(self.neuron_activation_avg, self.prune_activation_threshold)),
            tf.zeros_like(self.neuron_mask),
            self.neuron_mask
        )
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

# =============================================================================
# 6. MoE Dynamic Plastic Layer
# =============================================================================
class MoE_DynamicPlasticLayer(tf.keras.layers.Layer):
    def __init__(self, num_experts, max_units, initial_units, plasticity_controller, **kwargs):
        super(MoE_DynamicPlasticLayer, self).__init__(**kwargs)
        self.num_experts = num_experts
        self.max_units = max_units
        self.initial_units = initial_units
        self.plasticity_controller = plasticity_controller
        
    def build(self, input_shape):
        self.experts = [DynamicPlasticDenseDendritic(self.max_units, self.initial_units, self.plasticity_controller)
                        for _ in range(self.num_experts)]
        self.gating_network = tf.keras.Sequential([
            tf.keras.layers.Dense(self.num_experts*4, activation='relu', name='moe_dens1'),
            tf.keras.layers.Dense(self.num_experts, activation='softmax', name='moe_dens2')
        ])
        super(MoE_DynamicPlasticLayer, self).build(input_shape)
    
    def call(self, inputs, latent=None):
        expert_outputs = [expert(inputs) for expert in self.experts]
        #for i, expert_out in enumerate(expert_outputs):
        #    print(f"Expert {i} Output:", expert_out.numpy())  # Print expert activations
        expert_stack = tf.stack(expert_outputs, axis=-1)  # (batch, features, num_experts)
        gating_input = tf.concat([inputs, latent], axis=-1) if latent is not None else inputs
        gate_weights = self.gating_network(gating_input)  # (batch, num_experts)
        gate_weights = tf.expand_dims(gate_weights, axis=1)  # (batch, 1, num_experts)
        #print("Gate Weights:", gate_weights.numpy())  # Check if they are mostly zero

        output = tf.reduce_sum(expert_stack * gate_weights, axis=-1)
        return output

# =============================================================================
# 7. Episodic Memory for Continual Learning
# =============================================================================        
class EpisodicMemory(tf.keras.layers.Layer):
    def __init__(self, memory_size, memory_dim, **kwargs):
        super(EpisodicMemory, self).__init__(**kwargs)
        self.memory_size = memory_size
        self.memory_dim = memory_dim

    def build(self, input_shape):
        self.memory = self.add_weight(
            shape=(self.memory_size, self.memory_dim),
            initializer='glorot_uniform',
            trainable=False,
            name='episodic_memory'
        )
        self.write_layer = tf.keras.layers.Dense(self.memory_dim, activation='tanh', name='em_dens1')
        self.read_layer = tf.keras.layers.Dense(self.memory_size, activation='softmax', name='em_dens2')
        super(EpisodicMemory, self).build(input_shape)
    
    def call(self, inputs):
        attention = self.read_layer(inputs)  # (batch, memory_size)
        read_vector = tf.matmul(tf.expand_dims(attention, 1), self.memory)
        read_vector = tf.squeeze(read_vector, axis=1)
        write_vector = self.write_layer(inputs)
        aggregated_write_vector = tf.reduce_mean(write_vector, axis=0)
        idx = tf.random.uniform(shape=[], maxval=self.memory_size, dtype=tf.int32)
        memory_update = tf.tensor_scatter_nd_update(self.memory, [[idx]], [aggregated_write_vector])
        self.memory.assign(memory_update)
        return read_vector

# =============================================================================
# 8. Full Model with Integrated Unsupervised Branch and Dynamic Plastic Dense Layer.
# =============================================================================
class PlasticityModelMoE(tf.keras.Model):
    def __init__(self, h, w, plasticity_controller, units=128, num_experts=2, max_units=128, 
                 initial_units=32, num_classes=10, memory_size=512, memory_dim=128, **kwargs):
        super(PlasticityModelMoE, self).__init__(**kwargs)
        # Existing layers.
        self.act = LearnedQuantumActivation()
        self.cnn1 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')
        self.cnn2 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same')
        self.cnn3 = tf.keras.layers.Conv2D(512, (3,3), activation='relu', padding='same')
        self.drop = tf.keras.layers.Dropout(0.9)
        self.flatten = tf.keras.layers.Flatten()
        self.dens = tf.keras.layers.Dense(512, activation='relu', name='pm_dens1')
        self.unsupervised_extractor = UnsupervisedFeatureExtractor(units)
        self.episodic_memory = EpisodicMemory(memory_size, units)
        self.hidden = MoE_DynamicPlasticLayer(num_experts, max_units, initial_units, plasticity_controller)
        # Project hidden activations to dimension 16 (for 4 qubits: 2**4 = 16)
        self.quantum_dense = tf.keras.layers.Dense(units, activation=self.act, name='quantum_dense')
        # Instantiate the refined quantum entanglement layer.
        self.quantum_branch = RefinedQuantumEntanglementLayer(units=units, name="QuantumEntanglement")
        # Use the same dimension as the combined features.
        self.celestial_alignment = CelestialAlignmentLayer(units=units, period=12.0, name="CelestialAlignment")
        self.feature_combiner = tf.keras.layers.Dense(units, activation='relu', name='pm_dens2')
        self.classification_head = tf.keras.layers.Dense(num_classes, activation='softmax', name='pm_dens2')
        self.uncertainty_head = tf.keras.layers.Dense(1, activation='sigmoid', name='pm_dens3')
        self.meta_plasticity_controller = MetaPlasticityController(units=units)
        self.neural_ode_plasticity = NeuralODEPlasticity(dt=0.1, n_steps=10)

    def call(self, x, training=False):
        latent, reconstruction = self.unsupervised_extractor(x)
        memory_read = self.episodic_memory(latent)
        cnn1 = self.cnn1(x)
        cnn2 = self.cnn2(cnn1)
        cnn3 = self.cnn3(cnn2)
        drop = self.drop(cnn3, training=training)
        x_flat = self.flatten(drop)
        dens = self.dens(x_flat)
        
        combined_input = tf.concat([dens, latent, memory_read], axis=-1)
        hidden_act = self.hidden(combined_input)
        
        # --- Quantum Branch Processing ---
        quantum_input = self.quantum_dense(hidden_act)
        quantum_out, gate_probs = self.quantum_branch(quantum_input)

        # Compute quantum entropy as a feedback signal.
        eps = 1e-6
        quantum_entropy = -tf.reduce_sum(quantum_out * tf.math.log(quantum_out + eps), axis=-1, keepdims=True)

        # Fuse classical and quantum representations.
        combined = tf.concat([latent, memory_read, hidden_act, quantum_out], axis=-1)
        combined = self.feature_combiner(combined)
        
        # Apply the celestial branch to modulate the fused features.
        celestial_modulated = self.celestial_alignment(combined)
        
        class_out = self.classification_head(celestial_modulated)
        uncertainty = self.uncertainty_head(celestial_modulated)
        # Return quantum entropy and gate_probs for feedback purposes.
        return class_out, hidden_act, reconstruction, uncertainty, latent, quantum_entropy, gate_probs

# =============================================================================
# 9. Helper Functions for Signals and Rewards
# =============================================================================
# IMPORTANT: To avoid dtype mismatches when mixing float16 and float32,
# we force all computations in reinforcement signals to float32.
def chaotic_modulation(c, r=3.8):
    return r * c * (1 - c)
    
def compute_neuromodulatory_signal(predictions, external_reward=0.0):
    # Force computations to float32.
    predictions = tf.cast(predictions, tf.float32)
    eps = tf.constant(1e-6, dtype=tf.float32)
    entropy = -tf.reduce_sum(predictions * tf.math.log(predictions + eps), axis=-1)
    modulation = tf.sigmoid(tf.cast(external_reward, tf.float32) + tf.reduce_mean(entropy))
    return modulation

def compute_reinforcement_signals(loss, recon_loss, predictions, hidden, external_reward=0.0):
    # Force all inputs to float32.
    loss = tf.cast(loss, tf.float32)
    recon_loss = tf.cast(recon_loss, tf.float32)
    predictions = tf.cast(predictions, tf.float32)
    hidden = tf.cast(hidden, tf.float32)
    external_reward = tf.cast(external_reward, tf.float32)
    
    half_val = tf.constant(1.0, dtype=tf.float32) # Change to 0.5 when working with float16
    eps = tf.constant(1e-6, dtype=tf.float32)
    novelty = tf.math.reduce_std(hidden)
    entropy = -tf.reduce_sum(predictions * tf.math.log(predictions + eps), axis=-1)
    avg_uncertainty = tf.reduce_mean(entropy)
    loss_signal = tf.sigmoid(-loss)
    recon_signal = half_val * tf.sigmoid(-recon_loss)
    novelty_signal = half_val * tf.sigmoid(novelty)
    uncertainty_signal = half_val * tf.sigmoid(avg_uncertainty)
    combined_reward = loss_signal + recon_signal + novelty_signal + uncertainty_signal + external_reward
    return combined_reward
    
def compute_latent_mod(inputs):
    latent_modulator = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', name='latmod_dens1'),
        tf.keras.layers.Dense(1, activation='sigmoid', name='latmod_dens2')
    ])(inputs)
    return latent_modulator

# =============================================================================
# 10. Data Loading and Preprocessing Functions
# =============================================================================
def get_real_data(num_samples):
    dataset = 'Take5'
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
    inputs = inputs.reshape((-1, h, w, 1))
    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size).cache().repeat().prefetch(tf.data.AUTOTUNE)
    return dataset

# =============================================================================
# 11. Training Loop
# =============================================================================
def train_model(model, model_name, ds_train, ds_val, ds_test, train_steps, val_steps, test_steps,
                num_epochs=1000, homeostasis_interval=13, architecture_update_interval=21,
                plasticity_update_interval=8, plasticity_start_epoch=3,
                early_stop_patience=100, early_stop_min_delta=1e-4, learning_rate=1e-3):
    
    optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    recon_loss_fn = tf.keras.losses.Huber()
    uncertainty_loss_fn = tf.keras.losses.MeanSquaredError()
    
    global_plasticity_weight_multiplier = tf.Variable(0.5, trainable=False, dtype=tf.float32)
    global_reward_scaling_factor = tf.Variable(0.1, trainable=False, dtype=tf.float32)
    
    train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    val_loss_metric = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
    test_loss_metric = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    train_recon_loss_metric = tf.keras.metrics.Mean(name='train_recon_loss')
    val_recon_loss_metric = tf.keras.metrics.Mean(name='val_recon_loss')
    test_recon_loss_metric = tf.keras.metrics.Mean(name='test_recon_loss')    
    
    reward = 0.0
    global_step = 0
    best_val_loss = np.inf
    patience_counter = 0
    best_weights = None

    # @tf.function
    def train_step(images, labels, plasticity_weight, global_step):
        with tf.GradientTape() as tape:
            predictions, hidden, reconstruction, uncertainty, latent, quantum_entropy, gate_probs = model(images, training=True)
            loss = loss_fn(labels, predictions)
            recon_loss = recon_loss_fn(images, reconstruction)
            target_uncertainty = tf.reduce_mean(uncertainty)
            uncertainty_loss = uncertainty_loss_fn(target_uncertainty, uncertainty)
            total_loss = loss + tf.cast(0.1, loss.dtype) * recon_loss + tf.cast(0.05, loss.dtype) * uncertainty_loss
            
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, recon_loss, predictions, hidden, latent, total_loss, reconstruction, quantum_entropy, gate_probs

    for epoch in range(num_epochs):
        train_iter = iter(ds_train)
        val_iter = iter(ds_val)
        test_iter = iter(ds_test)
        batch_errors = []
        plasticity_weight_val = 0.0
        if epoch >= plasticity_start_epoch:
            plasticity_weight_val = ((epoch - plasticity_start_epoch + 1) /
                                     (num_epochs - plasticity_start_epoch + 1)) * global_plasticity_weight_multiplier.numpy()
        
        for step in range(train_steps):
            images, labels = next(train_iter)
            loss, recon_loss, predictions, hidden, latent, total_loss, reconstruction, quantum_entropy, gate_probs = train_step(images, labels, plasticity_weight_val, global_step)
            batch_errors.append(loss.numpy())
            train_loss_metric(loss)
            train_accuracy_metric(labels, predictions)
            train_recon_loss_metric(recon_loss)
            
            # Compute activities for plasticity update.
            x_flat = model.flatten(images)
            pre_activity = tf.reduce_mean(x_flat, axis=0)
            post_activity = tf.reduce_mean(hidden, axis=0)
            reward = tf.cast(global_reward_scaling_factor, loss.dtype) * compute_reinforcement_signals(loss, recon_loss, predictions, hidden, reward)
            base_neuromod_signal = compute_neuromodulatory_signal(predictions, reward)
            latent_signal = tf.reduce_mean(compute_latent_mod(latent))
            neuromod_signal = base_neuromod_signal * latent_signal
            # Modulate further with quantum feedback using sigmoid of quantum entropy.
            
            if plasticity_weight_val > 0 and global_step % plasticity_update_interval == 0:
                chaotic_factor = chaotic_modulation(global_plasticity_weight_multiplier.numpy())
                global_plasticity_weight_multiplier.assign(chaotic_factor)
                global_reward_scaling_factor.assign(chaotic_factor)
                context = tf.expand_dims(neuromod_signal, axis=0)

                # Compile a feature vector from the current plasticity state.
                # Use average pre_activity, post_activity, and average weight magnitude.
                pre_mean = tf.reduce_mean(pre_activity)
                post_mean = tf.reduce_mean(post_activity)
                # Assuming each expert has a weight matrix "w"
                avg_weight = tf.reduce_mean(tf.abs(expert.w))
                
                # Create a feature vector (shape: [1, feature_dim])
                plasticity_features = tf.expand_dims(tf.stack([pre_mean, post_mean, avg_weight]), axis=0)
                
                # Get adjustments: scale_factor, bias_adjustment, and update probability.
                scale_factor, bias_adjustment, update_prob = model.meta_plasticity_controller(plasticity_features)
                
                # Use the update probability to decide whether to update.
                if tf.reduce_mean(update_prob) > 0.5:
                    # For each expert in your MoE dynamic layer:
                    for expert in model.hidden.experts:
                        delta_w, delta_delay = expert.plasticity_update(pre_activity, post_activity, reward)
                        # Modulate delta_w with the scale factor and add a bias adjustment
                        plasticity_weight_cast = tf.cast(plasticity_weight_val, delta_w.dtype)
                        delta_w = (delta_w * scale_factor + bias_adjustment) * plasticity_weight_cast * neuromod_signal * chaotic_factor
                        delta_w = tf.clip_by_norm(delta_w, clip_norm=0.05)
                        delta_delay = tf.clip_by_norm(delta_delay, clip_norm=0.05)
                        
                        # Evolve the weight matrix continuously further using the Neural ODE.
                        updated_w = neural_ode_plasticity(expert.w, context=context) * delta_w
                        # Optionally, also evolve delay parameters.
                        updated_delay = neural_ode_plasticity(expert.delay, context=context) * delta_delay
                        
                        expert.apply_plasticity(updated_w, updated_delay)
                
            if global_step % homeostasis_interval == 0:
                for expert in model.hidden.experts:
                    expert.apply_homeostatic_scaling()
                    expert.apply_structural_plasticity()
                    
            if global_step % architecture_update_interval == 0:
                for expert in model.hidden.experts:
                    expert.apply_architecture_modification()
                    
            global_step += 1

        print(f"\nEpoch {epoch+1}/{num_epochs}\n"
              f"Train Loss : {train_loss_metric.result():.4f}, "
              f"Train Accuracy : {train_accuracy_metric.result():.4f}, "
              f"Train Recon Loss : {train_recon_loss_metric.result():.4f}")
        train_loss_metric.reset_state()
        train_accuracy_metric.reset_state()
        train_recon_loss_metric.reset_state()
        
        # Validation Evaluation.
        for step in range(val_steps):
            val_images, val_labels = next(val_iter)
            val_predictions, _, val_reconstruction, _, _, _, _ = model(val_images, training=False)
            val_loss = loss_fn(val_labels, val_predictions)
            val_recon_loss = recon_loss_fn(val_images, val_reconstruction)
            val_loss_metric(val_loss)
            val_accuracy_metric(val_labels, val_predictions)
            val_recon_loss_metric(val_recon_loss)
        current_val_loss = val_loss_metric.result().numpy()
        print(f"Val Loss   : {current_val_loss:.4f}, "
              f"Val Accuracy   : {val_accuracy_metric.result():.4f}, "
              f"Val Recon Loss : {val_recon_loss_metric.result():.4f}")
        val_loss_metric.reset_state()
        val_accuracy_metric.reset_state()
        val_recon_loss_metric.reset_state()
        
        # Test Evaluation.
        for step in range(test_steps):
            test_images, test_labels = next(test_iter)
            test_predictions, _, test_reconstruction, _, _, _, _ = model(test_images, training=False)
            test_loss = loss_fn(test_labels, test_predictions)
            test_recon_loss = recon_loss_fn(test_images, test_reconstruction)
            test_loss_metric(test_loss)
            test_accuracy_metric(test_labels, test_predictions)
            test_recon_loss_metric(test_recon_loss)
        print(f"Test Loss  : {test_loss_metric.result():.4f}, "
              f"Test Accuracy  : {test_accuracy_metric.result():.4f}, "
              f"Test Recon Loss : {test_recon_loss_metric.result():.4f}")
        test_loss_metric.reset_state()
        test_accuracy_metric.reset_state()
        test_recon_loss_metric.reset_state()
        
        # Early Stopping.
        if current_val_loss < best_val_loss - early_stop_min_delta:
            best_val_loss = current_val_loss
            patience_counter = 0
            best_weights = model.get_weights()
            print("\nValidation loss improved; resetting patience counter.")
            model.save(f"checkpoints/{model_name}.keras")
            model.unsupervised_extractor.save(f"checkpoints/extractors/{model_name}.keras")
        else:
            patience_counter += 1
            print(f"\nNo improvement in validation loss for {patience_counter} epoch(s).")
            if patience_counter >= early_stop_patience:
                if best_weights is not None:
                    print("Restoring best weights and saving model.\n")
                    model.set_weights(best_weights)
                    model.save(f"models/{model_name}.keras")
                    model.unsupervised_extractor.save(f"models/extractors/{model_name}.keras")
                break

# =============================================================================
# 12. Main Function: Build dataset, instantiate controllers and model, then train.
# =============================================================================
def main():
    batch_size = 64
    max_units = 128
    initial_units = 64
    epochs = 1000
    experts = 2
    h = 8
    w = 8
    cnn_units = h*w
    window_size = h * w
    learning_rate = 1e-3
    
    # Load data.
    sequence = get_real_data(num_samples=10_000)
    inputs, labels = create_windows(sequence, window_size=window_size+1)
    (inp_train, lbl_train), (inp_val, lbl_val), (inp_test, lbl_test) = split_dataset(inputs, labels)
    
    train_len = inp_train.shape[0]
    val_len = inp_val.shape[0]
    test_len = inp_test.shape[0]
    
    train_steps = math.ceil(train_len / batch_size)
    val_steps = math.ceil(val_len / batch_size)
    test_steps = math.ceil(test_len / batch_size)
    
    ds_train = create_tf_dataset(inp_train, lbl_train, h, w, batch_size, shuffle=False)
    ds_val = create_tf_dataset(inp_val, lbl_val, h, w, batch_size)
    ds_test = create_tf_dataset(inp_test, lbl_test, h, w, batch_size)
    
    # model.load_weights("model_weights/MetaSynapse_NextGen_v6c/model.weights.h5", skip_mismatch=True)
    
    dummy_input1 = tf.zeros((1, h, w, 1))
    dummy_input2 = tf.zeros((h, w))
    
    # Warm-ups
    plasticity_controller = RecurrentPlasticityController(units=initial_units)
    _ = plasticity_controller(tf.zeros((1,1,4)), training=False)
    _ = EpisodicMemory(h, w)(dummy_input2, training=False)
    _ = MoE_DynamicPlasticLayer(1, h, w, plasticity_controller)(dummy_input2, training=False)
    _ = RefinedQuantumEntanglementLayer(h)(dummy_input2, training=False)
    _ = DynamicConnectivity(h)(dummy_input2, training=False)
    _ = DynamicPlasticDenseDendritic(h, w, plasticity_controller)(dummy_input2, training=False)
    _ = CelestialAlignmentLayer(h)(dummy_input2, training=False)
    _ = MetaPlasticityController(h)(dummy_input2, training=False)
    _ = NeuralODEPlasticity(dt=0.1, n_steps=10)(dummy_input2, training=False)
    
    model = PlasticityModelMoE(h, w, plasticity_controller, units=cnn_units, 
                           num_experts=experts, max_units=max_units, initial_units=initial_units, num_classes=10)
    ae_model = model.unsupervised_extractor
   
    _ = model(dummy_input1, training=False)
    _ = ae_model(dummy_input1, training=False)
    
    model.summary()
    
    model_name = "MetaSynapse_NextGen_v7"
        
    train_model(model, model_name, ds_train, ds_val, ds_test, train_steps, val_steps, test_steps,
                num_epochs=epochs, homeostasis_interval=13, architecture_update_interval=55,
                plasticity_update_interval=8, plasticity_start_epoch=5, learning_rate=learning_rate)
    
if __name__ == '__main__':
    main()
