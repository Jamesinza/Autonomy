import tensorflow as tf
import pandas as pd
import numpy as np
import random
import math

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
        self.w = self.add_weight(name='activation_weights', shape=(8,),
                                 initializer='ones', trainable=True)
        super(LearnedActivation, self).build(input_shape)
    @tf.function
    def call(self, inputs):
        sig = tf.keras.activations.sigmoid(inputs)
        elu = tf.keras.activations.elu(inputs)
        tanh = tf.keras.activations.tanh(inputs)
        relu = tf.keras.activations.relu(inputs)
        silu = tf.keras.activations.silu(inputs)
        gelu = tf.keras.activations.gelu(inputs)
        selu = tf.keras.activations.selu(inputs)
        mish = tf.keras.activations.mish(inputs)
        weights = tf.nn.softmax(self.w)
        results = (weights[0]*sig + weights[1]*elu + weights[2]*tanh +
                   weights[3]*relu + weights[4]*silu + weights[5]*gelu +
                   weights[6]*selu + weights[7]*mish
                   )
        return results
        
# =============================================================================
# 2. Recurrent Plasticity Controller with Self-Att: Meta-learning the plasticity rules.
# =============================================================================
class RecurrentPlasticityController(tf.keras.layers.Layer):
    def __init__(self, units=16, sequence_length=10, **kwargs):
        """
        Args:
            units: Number of GRU units.
            sequence_length: Number of time steps to consider.
        """
        super(RecurrentPlasticityController, self).__init__(**kwargs)
        self.sequence_length = sequence_length
        self.units = units
        
    def build(self, input_shape):
        self.gru = tf.keras.layers.GRU(self.units, return_sequences=False)
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=self.units)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(2, activation=None)
        super(RecurrentPlasticityController, self).build(input_shape)
        
    @tf.function
    def call(self, input_sequence):
        """
        Args:
            input_sequence: Tensor of shape (batch, sequence_length, feature_dim)
        Returns:
            adjustments: Tensor of shape (batch, 2) representing [delta_add, delta_mult]
        """
        attn_output = self.attention(input_sequence,input_sequence)
        x = self.gru(attn_output)
        adjustments = self.dense(x)
        return adjustments

# =============================================================================
# 3. Unsupervised Feature Extractor: A simple auto-encoder for latent representations.
# =============================================================================
import tensorflow as tf

class UnsupervisedFeatureExtractor(tf.keras.Model):
    def __init__(self, units=128, **kwargs):
        super().__init__(**kwargs)
        """
        Autoencoder based on the U-Net architecture.
        """
        self.units = units

        # Encoder layers
        self.enc_cnn1 = tf.keras.layers.Conv2D(units, (3, 3), activation="relu", padding="same", name="enc_cnn1")
        self.enc_pool1 = tf.keras.layers.MaxPooling2D((2, 2), padding="same", name="enc_pool1")
        self.enc_cnn2 = tf.keras.layers.Conv2D(units * 2, (3, 3), activation="relu", padding="same", name="enc_cnn2")
        self.enc_pool2 = tf.keras.layers.MaxPooling2D((2, 2), padding="same", name="enc_pool2")
        self.enc_latent = tf.keras.layers.Conv2D(units * 4, (3, 3), activation="relu", padding="same", name="enc_latent")
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")

        # Decoder layers
        self.dec_upconv1 = tf.keras.layers.Conv2DTranspose(units * 2, (3, 3), strides=2, activation="relu", padding="same", name="dec_upconv1")
        self.dec_concat1 = tf.keras.layers.Concatenate(name="dec_concat1")
        self.dec_upconv2 = tf.keras.layers.Conv2DTranspose(units, (3, 3), strides=2, activation="relu", padding="same", name="dec_upconv2")
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

    @tf.function
    def call(self, inputs):
        x1, latent, flat_latent = self.encode(inputs)
        reconstruction = self.decode(x1, latent)
        return flat_latent, reconstruction

    def compute_output_shape(self, input_shape):
        return input_shape

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
            tf.keras.layers.Dense(self.max_units, activation='sigmoid')  # Outputs in [0,1]
        ])
        super(DynamicConnectivity, self).build(input_shape)
        
    @tf.function    
    def call(self, neuron_features):
        # neuron_features shape: (batch, max_units)
        connectivity = self.attention_net(neuron_features)
        return connectivity

# =============================================================================
# 5. Dynamic Plastic Dense Layer (Advanced Version 2.0)
#    Features:
#      - Uses a mask (of length max_units) that can be updated to prune/grow neurons.
#      - Uses a recurrent plasticity controller to compute plasticity update rules.
#      - Incorporates dynamic connectivity modulation.
# =============================================================================
import tensorflow as tf

class DynamicPlasticDenseAdvancedHyperV2(tf.keras.layers.Layer):
    def __init__(self, max_units, initial_units, plasticity_controller,
                 decay_factor=0.9, prune_activation_threshold=0.01, 
                 growth_activation_threshold=0.8, **kwargs):
        """
        Dynamic plastic dense layer with evolving architecture.
        """
        super().__init__(**kwargs)
        self.max_units = max_units
        self.initial_units = initial_units
        self.plasticity_controller = plasticity_controller
        self.decay_factor = decay_factor
        self.prune_activation_threshold = prune_activation_threshold
        self.growth_activation_threshold = growth_activation_threshold
        self.dynamic_connectivity = DynamicConnectivity(max_units=self.max_units)

    def build(self, input_shape):
        input_dim = int(input_shape[-1])

        # Trainable weights
        self.w = self.add_weight(shape=(input_dim, self.max_units),
                                 initializer='glorot_uniform',
                                 trainable=True, name='kernel')
        self.b = self.add_weight(shape=(self.max_units,),
                                 initializer='zeros',
                                 trainable=True, name='bias')

        # Non-trainable variables
        self.neuron_mask = self.add_weight(shape=(self.max_units,), initializer=tf.constant_initializer(
            [1] * self.initial_units + [0] * (self.max_units - self.initial_units)), trainable=False)
        self.neuron_activation_avg = self.add_weight(shape=(self.max_units,),
                                                     initializer='zeros', trainable=False)
        self.prune_threshold = self.add_weight(shape=(), initializer=tf.constant_initializer(0.01), trainable=False)
        self.add_prob = self.add_weight(shape=(), initializer=tf.constant_initializer(0.01), trainable=False)
        self.avg_weight_magnitude = self.add_weight(shape=(), initializer='zeros', trainable=False)

        super().build(input_shape)

    @tf.function
    def call(self, inputs):
        z = tf.matmul(inputs, self.w) + self.b  # (batch, max_units)
        connectivity = self.dynamic_connectivity(tf.expand_dims(self.neuron_activation_avg, axis=0))
        mask = tf.expand_dims(self.neuron_mask, axis=0)
        z_mod = z * connectivity * mask
        a = tf.keras.activations.relu(z_mod)

        # Update running average activation
        self.neuron_activation_avg.assign(0.9 * self.neuron_activation_avg + 0.1 * tf.reduce_mean(a, axis=0))
        return a

    @tf.function
    def apply_homeostatic_scaling(self):
        avg_w = tf.reduce_mean(tf.abs(self.w))
        scaling_factor = (self.decay_factor * 0.2 + (1 - self.decay_factor) * avg_w) / (avg_w + 1e-6)
        self.w.assign(self.w * scaling_factor)
        self.avg_weight_magnitude.assign(avg_w)

    @tf.function
    def apply_structural_plasticity(self):
        """Applies structural changes by pruning weak connections and adding new ones."""
        # Prune weak weights
        self.w.assign(tf.where(tf.abs(self.w) < self.prune_threshold, tf.zeros_like(self.w), self.w))
        
        # Add new connections with some probability
        random_matrix = tf.random.uniform(tf.shape(self.w))
        add_mask = tf.cast(random_matrix < self.add_prob, tf.float32)
        new_connections = tf.where((self.w == 0.0) & (add_mask == 1.0),
                                   tf.random.uniform(tf.shape(self.w), minval=0.01, maxval=0.05),
                                   self.w)
        self.w.assign(new_connections)

        # Update sparsity
        self.sparsity.assign(tf.reduce_mean(tf.cast(self.w == 0.0, tf.float32)))
        self.adjust_prune_threshold()
        self.adjust_add_prob()

    @tf.function
    def adjust_prune_threshold(self):
        self.prune_threshold.assign(self.prune_threshold * tf.cond(self.sparsity > 0.8, lambda: 1.05, lambda: 0.95))

    @tf.function
    def adjust_add_prob(self):
        self.add_prob.assign(self.add_prob * tf.cond(self.avg_weight_magnitude < 0.01, lambda: 1.05, lambda: 0.95))

    @tf.function
    def apply_architecture_modification(self):
        """Dynamically prunes and grows neurons."""
        mask = self.neuron_mask

        # Prune neurons with low activation
        mask = tf.where(self.neuron_activation_avg < self.prune_activation_threshold, 0.0, mask)

        # Extract active neurons
        active_values = tf.boolean_mask(self.neuron_activation_avg, mask == 1.0)

        # Condition: at least one active neuron and mean activation exceeds threshold
        growth_condition = tf.logical_and(tf.size(active_values) > 0,
                                          tf.reduce_mean(active_values) > self.growth_activation_threshold)

        def grow_neuron():
            """Activate a random inactive neuron."""
            inactive_indices = tf.where(mask == 0.0)
            if tf.size(inactive_indices) == 0:
                return mask  # No inactive neurons left

            chosen_index = inactive_indices[tf.random.uniform(shape=(), maxval=tf.size(inactive_indices),
                                                              dtype=tf.int32)]
            return tf.tensor_scatter_nd_update(mask, [[chosen_index]], [1.0])

        # Apply growth if condition is met
        self.neuron_mask.assign(tf.cond(growth_condition, grow_neuron, lambda: mask))
                
# --- Mixture-of-Experts Dynamic Plastic Layer ---
class MoE_DynamicPlasticLayer(tf.keras.layers.Layer):
    def __init__(self, num_experts, max_units, initial_units, plasticity_controller, use_latent=False, **kwargs):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.max_units = max_units
        self.initial_units = initial_units
        self.plasticity_controller = plasticity_controller
        self.use_latent = use_latent

        # Initialize experts
        self.experts = [
            DynamicPlasticDenseAdvancedHyperV2(self.max_units, self.initial_units, self.plasticity_controller)
            for _ in range(self.num_experts)
        ]

        # Define gating network
        self.gating_network = tf.keras.Sequential([
            tf.keras.layers.Dense(self.num_experts * 2, activation='relu'),
            tf.keras.layers.Dense(self.num_experts, activation='softmax')
        ])

    def compute_gating_weights(self, inputs, latent=None):
        """Compute the gating weights based on inputs (and optionally latent)."""
        if self.use_latent and latent is not None:
            inputs = tf.concat([inputs, latent], axis=-1)
        return self.gating_network(inputs)  # Shape: (batch, num_experts)

    @tf.function
    def call(self, inputs, latent=None):
        # Process input through experts (map_fn for better efficiency)
        expert_outputs = tf.map_fn(lambda expert: expert(inputs), self.experts, dtype=tf.float32)

        # Get gating weights
        gate_weights = self.compute_gating_weights(inputs, latent)  # Shape: (batch, num_experts)

        # Reshape for broadcasting (batch, 1, num_experts)
        gate_weights = tf.expand_dims(gate_weights, axis=1)

        # Weighted sum across experts
        output = tf.reduce_sum(expert_outputs * gate_weights, axis=-1)
        return output

        
# =============================================================================
# 6. Integrate Episodic Memory for Continual Learning
# =============================================================================        
class EpisodicMemory(tf.keras.layers.Layer):
    def __init__(self, memory_size, memory_dim, update_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.update_rate = update_rate  # Weight for memory update blending

    def build(self, input_shape):
        self.memory = self.add_weight(
            shape=(self.memory_size, self.memory_dim),
            initializer='glorot_uniform',
            trainable=False,
            name='episodic_memory'
        )
        self.write_layer = tf.keras.layers.Dense(self.memory_dim, activation='tanh', trainable=True)
        self.read_layer = tf.keras.layers.Dense(self.memory_size, activation='softmax', trainable=True)
        super().build(input_shape)

    def write_to_memory(self, write_vector):
        """Update memory using a weighted moving average approach."""
        idx = tf.random.uniform(shape=[1], maxval=self.memory_size, dtype=tf.int32)

        # Extract current memory slot
        current_memory = tf.gather(self.memory, idx)  # Shape: (1, memory_dim)

        # Blend the new information with old memory
        updated_memory = (1 - self.update_rate) * current_memory + self.update_rate * write_vector

        # Apply the update
        self.memory.assign(tf.tensor_scatter_nd_update(self.memory, idx, updated_memory))

    @tf.function
    def call(self, inputs):
        # Read memory using attention
        attention = self.read_layer(inputs)  # Shape: (batch, memory_size)
        read_vector = tf.einsum('bm,md->bd', attention, self.memory)  # Shape: (batch, memory_dim)

        # Compute write vectors and aggregate them
        write_vector = self.write_layer(inputs)  # Shape: (batch, memory_dim)
        aggregated_write_vector = tf.reduce_mean(write_vector, axis=0, keepdims=True)  # Shape: (1, memory_dim)

        # Update memory
        self.write_to_memory(aggregated_write_vector)
        
        return read_vector

# =============================================================================
# 7. Full Model with Integrated Unsupervised Branch and Dynamic Plastic Dense Layer.
# =============================================================================
class PlasticityModelMoE(tf.keras.Model):
    def __init__(self, h, w, plasticity_controller, num_experts=2, max_units=128, 
                 initial_units=32, num_classes=10, memory_size=512, memory_dim=32, **kwargs):
        super().__init__(**kwargs)
        self.unsupervised_extractor = UnsupervisedFeatureExtractor()
        self.episodic_memory = EpisodicMemory(memory_size, memory_dim)
        self.hidden = MoE_DynamicPlasticLayer(num_experts, max_units, initial_units, plasticity_controller)

        # Feature transformation layers
        self.feature_combiner = tf.keras.layers.Dense(128, activation='relu')
        self.classification_head = tf.keras.layers.Dense(num_classes, activation='softmax')
        self.uncertainty_head = tf.keras.layers.Dense(1, activation='sigmoid')

    def _process_features(self, x, latent, memory_read):
        """Combine extracted features before sending them through MoE layer."""
        x_flat = tf.reshape(x, (tf.shape(x)[0], -1))  # Instead of Flatten()
        combined_features = tf.concat([x_flat, latent, memory_read], axis=-1)
        return combined_features

    @tf.function
    def call(self, x, training=False):
        latent, reconstruction = self.unsupervised_extractor(x)
        memory_read = self.episodic_memory(latent)
        
        combined_input = self._process_features(x, latent, memory_read)
        hidden_act = self.hidden(combined_input)

        # Single feature combination step
        combined_features = tf.concat([hidden_act, latent, memory_read], axis=-1)
        combined = self.feature_combiner(combined_features)

        return {
            "class": self.classification_head(combined),
            "hidden": hidden_act,
            "reconstruction": reconstruction,
            "uncertainty": self.uncertainty_head(combined),
            "latent": latent
        }


# =============================================================================
# 8. Training Loop
# =============================================================================
def train_model(
    model, ds_train, ds_val, ds_test, train_steps, val_steps, test_steps, num_epochs=1000,
    homeostasis_interval=13, architecture_update_interval=21,
    plasticity_update_interval=8, plasticity_start_epoch=3,
    early_stop_patience=15, early_stop_min_delta=1e-4, learning_rate=1e-3, **kwargs):
    
    optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate)
    ae_optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    recon_loss_fn = tf.keras.losses.MeanSquaredError()
    uncertainty_loss_fn = tf.keras.losses.MeanSquaredError()

    # Global plasticity parameters.
    global_plasticity_weight_multiplier = tf.Variable(0.5, trainable=False, dtype=tf.float32)
    global_reward_scaling_factor = tf.Variable(0.1, trainable=False, dtype=tf.float32)

    # Metrics
    metrics = {
        "train_loss": tf.keras.metrics.Mean(),
        "train_accuracy": tf.keras.metrics.SparseCategoricalAccuracy(),
        "val_loss": tf.keras.metrics.Mean(),
        "val_accuracy": tf.keras.metrics.SparseCategoricalAccuracy(),
        "test_loss": tf.keras.metrics.Mean(),
        "test_accuracy": tf.keras.metrics.SparseCategoricalAccuracy(),
        "train_recon_loss": tf.keras.metrics.Mean(),
        "val_recon_loss": tf.keras.metrics.Mean(),
        "test_recon_loss": tf.keras.metrics.Mean(),
    }

    reward = 0.0
    global_step = 0
    best_val_loss = np.inf
    patience_counter = 0
    best_weights = None

    # --- Helper functions ---
    def run_step(images, labels, training):
        """Perform a single training step and return losses."""
        with tf.GradientTape(persistent=True) as tape:
            predictions, hidden, reconstruction, uncertainty, latent = model(images, training=training)
            loss = loss_fn(labels, predictions)
            recon_loss = recon_loss_fn(images, reconstruction)
            uncertainty_loss = uncertainty_loss_fn(tf.reduce_mean(uncertainty), uncertainty)
            total_loss = loss + 0.1 * recon_loss + 0.05 * uncertainty_loss

        return total_loss, loss, recon_loss, hidden, latent, predictions, tape

    def update_metrics(mode, loss, labels, predictions, recon_loss):
        """Update the relevant metrics for train/val/test mode."""
        metrics[f"{mode}_loss"](loss)
        metrics[f"{mode}_accuracy"](labels, predictions)
        metrics[f"{mode}_recon_loss"](recon_loss)

    def reset_metrics():
        """Reset all tracking metrics."""
        for metric in metrics.values():
            metric.reset_state()

    def apply_plasticity_updates(hidden, images, latent, predictions, labels, reconstruction):
        """Apply plasticity-based updates at specified intervals."""
        nonlocal reward
        x_flat = model.flatten(images)
        pre_activity = tf.reduce_mean(x_flat, axis=0)
        post_activity = tf.reduce_mean(hidden, axis=0)

        reward = global_reward_scaling_factor.numpy() * compute_reinforcement_signals(
            loss, recon_loss, predictions, hidden, reward
        )

        if plasticity_weight > 0 and global_step % plasticity_update_interval == 0:
            for expert in model.hidden.experts:
                expert_reward = compute_expert_reward(images, reconstruction, labels, predictions)
                neuromod_signal = compute_neuromodulatory_signal(predictions, expert_reward) * tf.reduce_mean(compute_latent_mod(latent))
                plasticity_delta = expert.plasticity_update(pre_activity, post_activity, expert_reward)

                if tf.reduce_sum(tf.abs(plasticity_delta)) > 0:
                    expert.apply_plasticity(tf.clip_by_norm(plasticity_delta * plasticity_weight * neuromod_signal, clip_norm=0.05))

        if global_step % homeostasis_interval == 0 or global_step % architecture_update_interval == 0:
            for expert in model.hidden.experts:
                if global_step % homeostasis_interval == 0:
                    expert.apply_homeostatic_scaling()
                    expert.apply_structural_plasticity()
                if global_step % architecture_update_interval == 0:
                    expert.apply_architecture_modification()

    def evaluate_model(ds, steps, mode):
        """Run evaluation for validation or test datasets."""
        for images, labels in ds.take(steps):
            loss, loss_value, recon_loss, _, _, predictions, _ = run_step(images, labels, training=False)
            update_metrics(mode, loss_value, labels, predictions, recon_loss)

        print(f"{mode.capitalize()} Loss: {metrics[f'{mode}_loss'].result():.4f}, "
              f"{mode.capitalize()} Accuracy: {metrics[f'{mode}_accuracy'].result():.4f}, "
              f"{mode.capitalize()} Recon Loss: {metrics[f'{mode}_recon_loss'].result():.4f}")

    # --- Training Loop ---
    for epoch in range(num_epochs):
        plasticity_weight = max(0, (epoch - plasticity_start_epoch + 1) /
                                (num_epochs - plasticity_start_epoch + 1) * global_plasticity_weight_multiplier.numpy())

        # Training
        for step, (images, labels) in enumerate(ds_train.take(train_steps)):
            total_loss, loss, recon_loss, hidden, latent, predictions, tape = run_step(images, labels, training=True)

            # Apply gradients
            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            ae_grads = tape.gradient(recon_loss, model.unsupervised_extractor.trainable_variables)
            ae_optimizer.apply_gradients(zip(ae_grads, model.unsupervised_extractor.trainable_variables))

            update_metrics("train", loss, labels, predictions, recon_loss)
            apply_plasticity_updates(hidden, images, latent, predictions, labels, reconstruction)

            global_step += 1

        # Print Training Metrics
        print(f"\nEpoch {epoch + 1}/{num_epochs}\n"
              f"Train Loss: {metrics['train_loss'].result():.4f}, "
              f"Train Accuracy: {metrics['train_accuracy'].result():.4f}, "
              f"Train Recon Loss: {metrics['train_recon_loss'].result():.4f}")

        reset_metrics()

        # Validation
        evaluate_model(ds_val, val_steps, "val")
        current_val_loss = metrics["val_loss"].result().numpy()
        reset_metrics()

        # Testing
        evaluate_model(ds_test, test_steps, "test")
        reset_metrics()

        # Early Stopping
        if current_val_loss < best_val_loss - early_stop_min_delta:
            best_val_loss = current_val_loss
            patience_counter = 0
            best_weights = model.get_weights()
            print("\nValidation loss improved; resetting patience counter.")
        else:
            patience_counter += 1
            print(f"\nNo improvement in validation loss for {patience_counter} epoch(s).")
            if patience_counter >= early_stop_patience:
                print("Restoring best weights and saving model.\n")
                if best_weights is not None:
                    model.set_weights(best_weights)
                    model.save("models/MetaSynapse_NextGen_v5b.keras")
                    model.unsupervised_extractor.save("models/unsupervised_extractor_v5b.keras")
                break  # Stop training


# =============================================================================
# 9. Helper functions for various signals.
# =============================================================================
def chaotic_modulation(c, r=3.8):
    # Simple logistic map.
    return r * c * (1 - c)
    
def compute_neuromodulatory_signal(predictions, external_reward=0.0):
    # Use entropy as a proxy for uncertainty.
    entropy = -tf.reduce_sum(predictions * tf.math.log(predictions + 1e-6), axis=-1)
    modulation = tf.sigmoid(external_reward + tf.reduce_mean(entropy))
    return modulation

def compute_reinforcement_signals(loss, recon_loss, predictions, hidden, external_reward=0.0):
    """
    Combines multiple signals into a reinforcement reward:
      - Loss-based signal: tf.sigmoid(-loss)
      - Reconstruction signal: 0.5 * tf.sigmoid(-recon_loss)
      - Novelty signal: 0.5 * tf.sigmoid(std(hidden))
      - Uncertainty signal: 0.5 * tf.sigmoid(mean(entropy(predictions)))
      - External reward can be added.
    """
    novelty = tf.math.reduce_std(hidden)
    entropy = -tf.reduce_sum(predictions * tf.math.log(predictions + 1e-6), axis=-1)
    avg_uncertainty = tf.reduce_mean(entropy)
    loss_signal = tf.sigmoid(-loss)
    recon_signal = 0.5 * tf.sigmoid(-recon_loss)
    novelty_signal = 0.5 * tf.sigmoid(novelty)
    uncertainty_signal = 0.5 * tf.sigmoid(avg_uncertainty)
    combined_reward = loss_signal + recon_signal + novelty_signal + uncertainty_signal + external_reward
    return combined_reward
    
def compute_expert_reward(expert_hidden, expert_reconstruction, labels, predictions):
    # Calculate expert-specific reconstruction loss.
    recon_loss = tf.keras.losses.MeanSquaredError()(expert_hidden, expert_reconstruction)
    # Measure novelty through activation standard deviation.
    novelty = tf.math.reduce_std(expert_hidden)
    # Use entropy as a proxy for uncertainty.
    entropy = -tf.reduce_sum(predictions * tf.math.log(predictions + 1e-6), axis=-1)
    uncertainty = tf.reduce_mean(entropy)
    # Combine the metrics into a specialized reward signal.
    expert_reward = tf.sigmoid(-recon_loss) + 0.5 * tf.sigmoid(novelty) + 0.5 * tf.sigmoid(-uncertainty)
    return expert_reward
    
def compute_latent_mod(inputs):
    latent_modulator = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Outputs a value between 0 and 1.
    ])(inputs)
    return latent_modulator
    
# =============================================================================
# 9. Data loading functions.
# =============================================================================
def get_real_data(num_samples):
    dataset = 'Take5'
    df = pd.read_csv(f'datasets/{dataset}_Full.csv')
    target_df = df.head(num_samples // 10).copy()
    cols = ['A', 'B', 'C', 'D', 'E']
    target_df = target_df[cols].dropna().astype(np.int8)
    target_df = target_df.map(lambda x: f'{x:02d}')
    flattened = target_df.values.flatten()
    full_data = np.array([int(d) for num in flattened for d in str(num)], dtype=np.int8)
    return full_data

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
    dataset = dataset.batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)
    return dataset

# =============================================================================
# 10. Main function: Build dataset, instantiate controllers and model, then train.
# =============================================================================
def main():
    batch_size = 128
    max_units = 256
    initial_units = 64
    epochs = 1000
    experts = 16
    h = 16
    w = 16
    window_size = h*w
    learning_rate=1e-4
    
    # Load data.
    sequence = get_real_data(num_samples=100_000)
    inputs, labels = create_windows(sequence, window_size=window_size+1)
    (inp_train, lbl_train), (inp_val, lbl_val), (inp_test, lbl_test) = split_dataset(inputs, labels)
    
    train_len = inp_train.shape[0]+lbl_train.shape[0]
    val_len = inp_val.shape[0]+lbl_val.shape[0]
    test_len = inp_test.shape[0]+lbl_test.shape[0]
    
    train_steps = math.ceil((train_len-window_size-1)/batch_size)
    val_steps = math.ceil((train_len-window_size-1)/batch_size)
    test_steps = math.ceil((train_len-window_size-1)/batch_size)
    
    ds_train = create_tf_dataset(inp_train, lbl_train, h, w, batch_size)
    ds_val = create_tf_dataset(inp_val, lbl_val, h, w, batch_size)
    ds_test = create_tf_dataset(inp_test, lbl_test, h, w, batch_size)
    
    # Instantiate the recurrent plasticity controller.
    plasticity_controller = RecurrentPlasticityController(units=initial_units)
    _ = plasticity_controller(tf.zeros((1,1,4)))  # Warm up
    
    # Instantiate the new MoE-based model.
    model = PlasticityModelMoE(h, w, plasticity_controller, num_experts=experts, max_units=max_units, initial_units=initial_units, num_classes=10)
    ae_model = model.unsupervised_extractor
    dummy_input = tf.zeros((1, h, w, 1))
    _ = model(dummy_input, training=False)
    _ = ae_model(dummy_input, training=False)
    model.summary()
        
    # Train the model.
    train_model(model, ds_train, ds_val, ds_test, train_steps, val_steps, test_steps, num_epochs=epochs,
                homeostasis_interval=8, architecture_update_interval=21,
                plasticity_update_interval=5, plasticity_start_epoch=3, learning_rate=learning_rate)
    
if __name__ == '__main__':
    main()

