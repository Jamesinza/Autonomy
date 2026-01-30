import tensorflow as tf
import pandas as pd
import numpy as np
import math
import random

# --- Enable Mixed Precision (if GPU is available) ---
# if tf.config.list_physical_devices('GPU'):
#     tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Set seeds for reproducibility.
np.random.seed(42)
tf.random.set_seed(42)

# =============================================================================
# 1. Learned Activation Function: Dynamically adjust weights based on input context
# =============================================================================
class LearnedActivation(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LearnedActivation, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.w = self.add_weight(name='activation_weights', shape=(9,),
                                 initializer='ones', trainable=True)
        super(LearnedActivation, self).build(input_shape)
    
    # @tf.function
    def call(self, inputs):
        sig = tf.keras.activations.sigmoid(inputs)
        elu = tf.keras.activations.elu(inputs)
        tanh = tf.keras.activations.tanh(inputs)
        relu = tf.keras.activations.relu(inputs)
        silu = tf.keras.activations.silu(inputs)
        gelu = tf.keras.activations.gelu(inputs)
        selu = tf.keras.activations.selu(inputs)
        mish = tf.keras.activations.mish(inputs)
        linear = 0.0
        weights = tf.nn.softmax(self.w)
        results = (weights[0]*sig + weights[1]*elu + weights[2]*tanh +
                   weights[3]*relu + weights[4]*silu + weights[5]*gelu +
                   weights[6]*selu + weights[7]*mish + weights[8]*linear)
        return results

# =============================================================================
# 2. Recurrent Plasticity Controller with Self-Attention
# =============================================================================
class RecurrentPlasticityController(tf.keras.layers.Layer):
    def __init__(self, units=16, sequence_length=10, **kwargs):
        super(RecurrentPlasticityController, self).__init__(**kwargs)
        self.sequence_length = sequence_length
        self.units = units
        
    def build(self, input_shape):
        self.gru = tf.keras.layers.GRU(self.units, return_sequences=False)
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=self.units)
        self.dense = tf.keras.layers.Dense(2, activation=None)
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

    # @tf.function
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
        
    def build(self, input_shape):
        self.attention_net = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(self.max_units, activation='sigmoid')
        ])
        super(DynamicConnectivity, self).build(input_shape)
        
    # @tf.function
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
                                 trainable=True, name='kernel')
        # Bias: shape (max_units, num_branches)
        self.b = self.add_weight(shape=(self.max_units, self.num_branches),
                                 initializer='zeros',
                                 trainable=True, name='bias')
        # Delay for each branch.
        self.delay = self.add_weight(shape=(input_dim, self.max_units, self.num_branches),
                                     initializer='zeros',
                                     trainable=True, name='delay')
        # Gating mechanism to combine dendritic branch outputs.
        self.branch_gating = tf.keras.layers.Dense(self.num_branches, activation='softmax', name="branch_gating")
        super(DynamicPlasticDenseDendritic, self).build(input_shape)

    # @tf.function
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
        
        a = tf.keras.activations.relu(z_modulated)
        
        # Update running average of activations.
        batch_mean = tf.reduce_mean(a, axis=0)
        alpha = tf.cast(0.9, self.neuron_activation_avg.dtype)
        new_avg = alpha * self.neuron_activation_avg + (1 - alpha) * tf.cast(batch_mean, self.neuron_activation_avg.dtype)
        self.neuron_activation_avg.assign(new_avg)
        return a

    # @tf.function
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

    # @tf.function
    def apply_plasticity(self, plasticity_delta_w, plasticity_delta_delay):
        self.w.assign_add(plasticity_delta_w)
        self.delay.assign_add(plasticity_delta_delay)

    # @tf.function
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

    # @tf.function
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

    # @tf.function
    def adjust_prune_threshold(self):
        self.prune_threshold.assign(tf.cond(tf.greater(self.sparsity, 0.8),
                                              lambda: self.prune_threshold * 1.05,
                                              lambda: tf.cond(tf.less(self.sparsity, 0.3),
                                                              lambda: self.prune_threshold * 0.95,
                                                              lambda: self.prune_threshold)))

    # @tf.function
    def adjust_add_prob(self):
        self.add_prob.assign(tf.cond(tf.less(self.avg_weight_magnitude, 0.01),
                                       lambda: self.add_prob * 1.05,
                                       lambda: tf.cond(tf.greater(self.avg_weight_magnitude, 0.1),
                                                       lambda: self.add_prob * 0.95,
                                                       lambda: self.add_prob)))

    # @tf.function
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
            tf.keras.layers.Dense(self.num_experts*2, activation='relu'),
            tf.keras.layers.Dense(self.num_experts, activation='softmax')
        ])
        super(MoE_DynamicPlasticLayer, self).build(input_shape)
        
    # @tf.function
    def call(self, inputs, latent=None):
        expert_outputs = [expert(inputs) for expert in self.experts]
        expert_stack = tf.stack(expert_outputs, axis=-1)  # (batch, features, num_experts)
        gating_input = tf.concat([inputs, latent], axis=-1) if latent is not None else inputs
        gate_weights = self.gating_network(gating_input)  # (batch, num_experts)
        gate_weights = tf.expand_dims(gate_weights, axis=1)  # (batch, 1, num_experts)
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
        self.write_layer = tf.keras.layers.Dense(self.memory_dim, activation='tanh')
        self.read_layer = tf.keras.layers.Dense(self.memory_size, activation='softmax')
        super(EpisodicMemory, self).build(input_shape)

    # @tf.function
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
        self.cnn1 = tf.keras.layers.Conv2D(units//2, (3,3), activation='relu', padding='same')
        self.cnn2 = tf.keras.layers.Conv2D(units, (3,3), activation='relu', padding='same')
        self.cnn3 = tf.keras.layers.Conv2D(units*2, (3,3), activation='relu', padding='same')
        self.drop = tf.keras.layers.Dropout(0.9)
        self.flatten = tf.keras.layers.Flatten()
        self.dens = tf.keras.layers.Dense(units, activation=LearnedActivation())
        self.unsupervised_extractor = UnsupervisedFeatureExtractor(units)
        self.episodic_memory = EpisodicMemory(memory_size, units)
        self.hidden = MoE_DynamicPlasticLayer(num_experts, max_units, initial_units, plasticity_controller)
        self.feature_combiner = tf.keras.layers.Dense(units, activation='relu')
        self.classification_head = tf.keras.layers.Dense(num_classes, activation='softmax')
        self.uncertainty_head = tf.keras.layers.Dense(1, activation='sigmoid')
    
    # Do not decorate call to avoid retracing issues.
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
        combined = tf.concat([hidden_act, latent, memory_read], axis=-1)
        combined = self.feature_combiner(combined)
        class_out = self.classification_head(combined)
        uncertainty = self.uncertainty_head(combined)
        return class_out, hidden_act, reconstruction, uncertainty, latent

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
    
    half_val = tf.constant(0.5, dtype=tf.float32)
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
    
def compute_expert_reward(expert_hidden, expert_reconstruction, labels, predictions):
    # Compute in float32.
    expert_hidden = tf.cast(expert_hidden, tf.float32)
    expert_reconstruction = tf.cast(expert_reconstruction, tf.float32)
    predictions = tf.cast(predictions, tf.float32)
    recon_loss = tf.keras.losses.MeanSquaredError()(expert_hidden, expert_reconstruction)
    half_val = tf.constant(0.5, dtype=tf.float32)
    eps = tf.constant(1e-6, dtype=tf.float32)
    novelty = tf.math.reduce_std(expert_hidden)
    entropy = -tf.reduce_sum(predictions * tf.math.log(predictions + eps), axis=-1)
    uncertainty = tf.reduce_mean(entropy)
    expert_reward = tf.sigmoid(-recon_loss) + half_val * tf.sigmoid(novelty) + half_val * tf.sigmoid(-uncertainty)
    return expert_reward
    
def compute_latent_mod(inputs):
    latent_modulator = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
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
                early_stop_patience=10, early_stop_min_delta=1e-4, learning_rate=1e-3):
    
    optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate)
    # ae_optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate)
    
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

    # @tf.function(experimental_compile=False)
    def train_step(images, labels, plasticity_weight, global_step):
        with tf.GradientTape() as tape:
            predictions, hidden, reconstruction, uncertainty, latent = model(images, training=True)
            loss = loss_fn(labels, predictions)
            recon_loss = recon_loss_fn(images, reconstruction)
            target_uncertainty = tf.reduce_mean(uncertainty)
            uncertainty_loss = uncertainty_loss_fn(target_uncertainty, uncertainty)
            total_loss = loss + tf.cast(0.1, loss.dtype) * recon_loss + tf.cast(0.05, loss.dtype) * uncertainty_loss
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, recon_loss, predictions, hidden, latent, total_loss, reconstruction

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
            loss, recon_loss, predictions, hidden, latent, total_loss, reconstruction = train_step(images, labels, plasticity_weight_val, global_step)
            batch_errors.append(loss.numpy())
            train_loss_metric(loss)
            train_accuracy_metric(labels, predictions)
            train_recon_loss_metric(recon_loss)
            
            # Compute activities for plasticity update.
            x_flat = model.flatten(images)
            pre_activity = tf.reduce_mean(x_flat, axis=0)
            post_activity = tf.reduce_mean(hidden, axis=0)
            reward = tf.cast(global_reward_scaling_factor, loss.dtype) * compute_reinforcement_signals(loss, recon_loss, predictions, hidden, reward)
            
            if plasticity_weight_val > 0 and global_step % plasticity_update_interval == 0:
                chaotic_factor = chaotic_modulation(global_plasticity_weight_multiplier.numpy())
                global_plasticity_weight_multiplier.assign(chaotic_factor)
                for expert in model.hidden.experts:
                    expert_reward = compute_expert_reward(images, reconstruction, labels, predictions)
                    base_neuromod_signal = compute_neuromodulatory_signal(predictions, expert_reward)
                    latent_signal = tf.reduce_mean(compute_latent_mod(latent))
                    neuromod_signal = base_neuromod_signal * latent_signal
                    delta_w, delta_delay = expert.plasticity_update(pre_activity, post_activity, expert_reward)
                    if tf.reduce_sum(tf.abs(delta_w)) > 0:
                        plasticity_weight_cast = tf.cast(plasticity_weight_val, delta_w.dtype)
                        delta_w *= plasticity_weight_cast * neuromod_signal
                        delta_w = tf.clip_by_norm(delta_w, clip_norm=0.05)
                        delta_delay = tf.clip_by_norm(delta_delay, clip_norm=0.05)
                        expert.apply_plasticity(delta_w, delta_delay)
            
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
            val_predictions, _, val_reconstruction, _, _ = model(val_images, training=False)
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
            test_predictions, _, test_reconstruction, _, _ = model(test_images, training=False)
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
    batch_size = 512
    max_units = 256
    initial_units = 64
    epochs = 1000
    experts = 64
    h = 16
    w = 16
    cnn_units = h*w
    window_size = h * w
    learning_rate = 1e-3
    
    # Load data.
    sequence = get_base_data(num_samples=10_000)
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
    
    plasticity_controller = RecurrentPlasticityController(units=initial_units)
    _ = plasticity_controller(tf.zeros((1, 1, 4)))  # Warm up
    
    model = PlasticityModelMoE(h, w, plasticity_controller, units=cnn_units, 
                               num_experts=experts, max_units=max_units, initial_units=initial_units, num_classes=10)
    ae_model = model.unsupervised_extractor
    dummy_input = tf.zeros((1, h, w, 1))
    _ = model(dummy_input, training=False)
    _ = ae_model(dummy_input, training=False)
    model.summary()
    
    model_name = "MetaSynapse_NextGen_v6b"
        
    train_model(model, model_name, ds_train, ds_val, ds_test, train_steps, val_steps, test_steps,
                num_epochs=epochs, homeostasis_interval=13, architecture_update_interval=55,
                plasticity_update_interval=8, plasticity_start_epoch=3, learning_rate=learning_rate)
    
if __name__ == '__main__':
    main()
