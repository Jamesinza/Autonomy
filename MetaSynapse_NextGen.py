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
        self.w = self.add_weight(name='activation_weights', shape=(9,),
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
        linear = 0.0
        weights = tf.nn.softmax(self.w)
        results = (weights[0]*sig + weights[1]*elu + weights[2]*tanh +
                   weights[3]*relu + weights[4]*silu + weights[5]*gelu +
                   weights[6]*selu + weights[7]*mish + weights[8]*linear
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
        #combined = x + attn_output
        #flattened = self.flatten(combined)
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
class DynamicPlasticDenseAdvancedHyperV2(tf.keras.layers.Layer):
    def __init__(self, max_units, initial_units, plasticity_controller, 
                 decay_factor=0.9, prune_activation_threshold=0.01, growth_activation_threshold=0.8, **kwargs):
        """
        max_units: total candidate neurons.
        initial_units: neurons active at initialization.
        plasticity_controller: instance of RecurrentPlasticityController.
        decay_factor: for homeostatic scaling.
        prune_activation_threshold: activation below which an active neuron is pruned.
        growth_activation_threshold: if the mean activation of active neurons exceeds this,
                                     then an inactive neuron is activated (growth).
        """
        super(DynamicPlasticDenseAdvancedHyperV2, self).__init__(**kwargs)
        self.max_units = max_units
        self.initial_units = initial_units
        self.plasticity_controller = plasticity_controller
        self.decay_factor = decay_factor
        self.prune_activation_threshold = prune_activation_threshold
        self.growth_activation_threshold = growth_activation_threshold
        self.dynamic_connectivity = DynamicConnectivity(max_units=self.max_units)
        # Plasticity & Homeostasis parameters.
        self.prune_threshold = tf.Variable(0.01, trainable=False, dtype=tf.float32)
        self.add_prob = tf.Variable(0.01, trainable=False, dtype=tf.float32)
        self.sparsity = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.avg_weight_magnitude = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.target_avg = tf.Variable(0.2, trainable=False, dtype=tf.float32)        
        # Create a mask for active neurons.
        init_mask = [1] * self.initial_units + [0] * (self.max_units - self.initial_units)
        self.neuron_mask = tf.Variable(self.init_mask, trainable=False, dtype=tf.float32)
        # Running average activation (per neuron)
        self.neuron_activation_avg = tf.Variable(tf.zeros([self.max_units], dtype=tf.float32),
                                                 trainable=False)
        # Buffer for accumulating feature vectors for meta-updates.
        self.feature_history = []

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        self.w = self.add_weight(shape=(input_dim, self.max_units),
                                 initializer='glorot_uniform',
                                 trainable=True, name='kernel')
        self.b = self.add_weight(shape=(self.max_units,),
                                 initializer='zeros',
                                 trainable=True, name='bias')
        super(DynamicPlasticDenseAdvancedHyperV2, self).build(input_shape)
    @tf.function
    def call(self, inputs):
        z = tf.matmul(inputs, self.w) + self.b  # (batch, max_units)
        connectivity = self.dynamic_connectivity(tf.expand_dims(self.neuron_activation_avg, axis=0))
        mask = tf.expand_dims(self.neuron_mask, axis=0)
        z_mod = z * connectivity * mask
        a = tf.keras.activations.get('relu')(z_mod)
        # Update running average activation.
        batch_mean = tf.reduce_mean(a, axis=0)
        alpha = 0.9
        new_avg = alpha * self.neuron_activation_avg + (1 - alpha) * batch_mean
        self.neuron_activation_avg.assign(new_avg)
        return a

    def plasticity_update(self, pre_activity, post_activity, reward):
        """
        Compute plasticity update for self.w using the recurrent controller with temporal context.
        Accumulates a history of feature vectors and uses a GRU once the buffer is full.
        """
        pre_mean = tf.reduce_mean(pre_activity)
        post_mean = tf.reduce_mean(post_activity)
        weight_mean = tf.reduce_mean(self.w)
        reward_feature = tf.cast(reward, tf.float32)
        # Form the current feature vector.
        current_feature = tf.stack([pre_mean, post_mean, weight_mean, reward_feature])
        self.feature_history.append(current_feature)
        # Only compute an update if the history is long enough.
        if len(self.feature_history) < self.plasticity_controller.sequence_length:
            return tf.zeros_like(self.w)
        # Build the sequence tensor: shape (1, sequence_length, feature_dim)
        recent_features = self.feature_history[-self.plasticity_controller.sequence_length:]
        feature_sequence = tf.stack(recent_features, axis=0)
        feature_sequence = tf.expand_dims(feature_sequence, axis=0)
        adjustments = self.plasticity_controller(feature_sequence)  # (1, 2)
        delta_add, delta_mult = adjustments[0, 0], adjustments[0, 1]
        plasticity_delta = tf.cast(reward, tf.float32) * (delta_add + delta_mult * self.w)
        return plasticity_delta

    def apply_plasticity(self, plasticity_delta):
        self.w.assign_add(plasticity_delta)
    @tf.function
    def apply_homeostatic_scaling(self):
        avg_w = tf.reduce_mean(tf.abs(self.w))
        new_target = self.decay_factor * self.target_avg + (1 - self.decay_factor) * avg_w
        scaling_factor = new_target / (avg_w + 1e-6)
        self.w.assign(self.w * scaling_factor)
        self.avg_weight_magnitude.assign(avg_w)
    @tf.function
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
    @tf.function
    def adjust_prune_threshold(self):
        if self.sparsity > 0.8:
            self.prune_threshold.assign(self.prune_threshold * 1.05)
        elif self.sparsity < 0.3:
            self.prune_threshold.assign(self.prune_threshold * 0.95)
    @tf.function
    def adjust_add_prob(self):
        if self.avg_weight_magnitude < 0.01:
            self.add_prob.assign(self.add_prob * 1.05)
        elif self.avg_weight_magnitude > 0.1:
            self.add_prob.assign(self.add_prob * 0.95)
    @tf.function
    def apply_architecture_modification(self):
        # Prune: For active neurons, set mask to 0 if their running average activation is below threshold.
        mask_after_prune = tf.where(
            tf.logical_and(
                tf.equal(self.neuron_mask, 1.0),
                tf.less(self.neuron_activation_avg, self.prune_activation_threshold)
            ),
            tf.zeros_like(self.neuron_mask),
            self.neuron_mask
        )
        # Extract activations of the currently active neurons.
        active_values = tf.boolean_mask(self.neuron_activation_avg, tf.equal(mask_after_prune, 1.0))
        # Determine if the growth condition is met:
        # There must be at least one active neuron and their mean activation exceeds the growth threshold.
        growth_cond = tf.cond(
            tf.greater(tf.size(active_values), 0),
            lambda: tf.greater(tf.reduce_mean(active_values), self.growth_activation_threshold),
            lambda: tf.constant(False)
        )
        def grow_neuron():
            # Find indices of inactive neurons.
            inactive_indices = tf.squeeze(tf.where(tf.equal(mask_after_prune, 0.0)), axis=1)
            # If there is at least one inactive neuron, activate one at random.
            def activate_random():
                # Shuffle inactive indices and pick the first.
                random_idx = tf.random.shuffle(inactive_indices)[0]
                # Update the mask: set the chosen index to 1.
                new_mask = tf.tensor_scatter_nd_update(
                    mask_after_prune,
                    indices=tf.reshape(random_idx, (-1, 1)),
                    updates=[1.0]
                )
                return new_mask
            # If no inactive neuron is available, return the mask unchanged.
            new_mask = tf.cond(
                tf.greater(tf.shape(inactive_indices)[0], 0),
                activate_random,
                lambda: mask_after_prune
            )
            return new_mask
        # If growth condition is met, update the mask by growing one neuron.
        # Otherwise, keep the pruned mask.
        new_mask = tf.cond(growth_cond, grow_neuron, lambda: mask_after_prune)
        # Assign the updated mask back to the variable.
        self.neuron_mask.assign(new_mask)
                
# --- Mixture-of-Experts Dynamic Plastic Layer ---
class MoE_DynamicPlasticLayer(tf.keras.layers.Layer):
    def __init__(self, num_experts, max_units, initial_units, plasticity_controller, **kwargs):
        super(MoE_DynamicPlasticLayer, self).__init__(**kwargs)
        self.num_experts = num_experts
        self.max_units = max_units
        self.initial_units = initial_units
        self.plasticity_controller = plasticity_controller
        
    def build(self, input_shape):
        # Instantiate several experts (each an instance of your existing dynamic plastic dense layer).
        self.experts = [
            DynamicPlasticDenseAdvancedHyperV2(self.max_units, self.initial_units, self.plasticity_controller)
            for _ in range(self.num_experts)
        ]
        # A gating network that assigns a weight to each expert based on the input.
        self.gating_network = tf.keras.Sequential([
            tf.keras.layers.Dense(self.num_experts*2, activation='relu'),
            tf.keras.layers.Dense(self.num_experts, activation='softmax')
        ])
        super(MoE_DynamicPlasticLayer, self).build(input_shape)
        
    @tf.function
    def call(self, inputs, latent=None):
        # Process the input through each expert.
        expert_outputs = [expert(inputs) for expert in self.experts]
        # Stack expert outputs: shape (batch, features, num_experts)
        expert_stack = tf.stack(expert_outputs, axis=-1)
        # If latent is provided, concatenate it with inputs.
        if latent is not None:
            gating_input = tf.concat([inputs, latent], axis=-1)
        else:
            gating_input = inputs        
        # Get gating weights: shape (batch, num_experts)
        gate_weights = self.gating_network(gating_input)
        # Expand dims for broadcasting: (batch, 1, num_experts)
        gate_weights = tf.expand_dims(gate_weights, axis=1)
        # Compute a weighted sum over experts.
        output = tf.reduce_sum(expert_stack * gate_weights, axis=-1)
        return output
        
# =============================================================================
# 6. Integrate Episodic Memory for Continual Learning
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
        # Output a weighting over memory slots.
        self.read_layer = tf.keras.layers.Dense(self.memory_size, activation='softmax')
        super(EpisodicMemory, self).build(input_shape)

    @tf.function
    def call(self, inputs):
        # Read: Compute attention weights over memory slots.
        attention = self.read_layer(inputs)  # Expected shape: (batch, memory_size)
        read_vector = tf.matmul(tf.expand_dims(attention, 1), self.memory)
        read_vector = tf.squeeze(read_vector, axis=1)  # Now shape: (batch, memory_dim)
        # Write: Compute write vectors from the inputs.
        write_vector = self.write_layer(inputs)  # Shape: (batch, memory_dim)
        # Aggregate the batch into one vector.
        aggregated_write_vector = tf.reduce_mean(write_vector, axis=0)  # Shape: (memory_dim,)
        # Update a random memory slot with the aggregated write vector.
        idx = tf.random.uniform(shape=[], maxval=self.memory_size, dtype=tf.int32)
        memory_update = tf.tensor_scatter_nd_update(self.memory, [[idx]], [aggregated_write_vector])
        self.memory.assign(memory_update)
        return read_vector

# =============================================================================
# 7. Full Model with Integrated Unsupervised Branch and Dynamic Plastic Dense Layer.
# =============================================================================
class PlasticityModelMoE(tf.keras.Model):
    def __init__(self, h, w, plasticity_controller, units=128, num_experts=2, max_units=128, 
                 initial_units=32, num_classes=10, memory_size=512, memory_dim=32, **kwargs):
        super(PlasticityModelMoE, self).__init__(**kwargs)
        self.cnn1 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')
        self.cnn2 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same')
        self.cnn3 = tf.keras.layers.Conv2D(512, (3,3), activation='relu', padding='same')
        self.drop = tf.keras.layers.Dropout(0.9)
        self.flatten = tf.keras.layers.Flatten()
        self.dens = tf.keras.layers.Dense(128, activation=LearnedActivation())
        self.unsupervised_extractor = UnsupervisedFeatureExtractor()
        self.episodic_memory = EpisodicMemory(memory_size, memory_dim)
        self.hidden = MoE_DynamicPlasticLayer(num_experts, max_units, initial_units, plasticity_controller)
        self.feature_combiner = tf.keras.layers.Dense(128, activation='relu')
        self.classification_head = tf.keras.layers.Dense(num_classes, activation='softmax')
        self.uncertainty_head = tf.keras.layers.Dense(1, activation='sigmoid')
    
    @tf.function
    def call(self, x, training=False):
        latent, reconstruction = self.unsupervised_extractor(x)
        memory_read = self.episodic_memory(latent)
        cnn1 = self.cnn1(x)
        cnn2 = self.cnn2(cnn1)
        cnn3 = self.cnn3(cnn2)
        drop = self.drop(cnn3)
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
# 8. Training Loop: Incorporates classification loss, unsupervised (reconstruction) loss,
#    recurrent meta-plasticity updates (with unsupervised influence), chaotic modulation,
#    and architecture modifications.
# =============================================================================
def train_model(model, ds_train, ds_val, ds_test, train_steps, val_steps, test_steps, num_epochs=1000,
                homeostasis_interval=13, architecture_update_interval=21,
                plasticity_update_interval=8, plasticity_start_epoch=3,
                early_stop_patience=10, early_stop_min_delta=1e-4, learning_rate=1e-3, **kwargs):
                
    optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate)
    ae_optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate)
    
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    recon_loss_fn = tf.keras.losses.Huber()
    uncertainty_loss_fn = tf.keras.losses.MeanSquaredError()
    
    # Global variables controlled by the recurrent plasticity controller.
    global_plasticity_weight_multiplier = tf.Variable(0.5, trainable=False, dtype=tf.float32)
    global_reward_scaling_factor = tf.Variable(0.1, trainable=False, dtype=tf.float32)
    
    train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    
    # Metrics for validation, test and autoencoder.
    val_loss_metric = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
    
    test_loss_metric = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    
    train_recon_loss_metric = tf.keras.metrics.Mean(name='train_recon_loss')
    val_recon_loss_metric = tf.keras.metrics.Mean(name='val_recon_loss')
    test_recon_loss_metric = tf.keras.metrics.Mean(name='test_recon_loss')    
    
    reward = 0.0
    global_step = 0

    # Early stopping variables.
    best_val_loss = np.inf
    patience_counter = 0
    best_weights = None
    
    for epoch in range(num_epochs):
        train_iter = iter(ds_train)
        val_iter = iter(ds_val)
        test_iter = iter(ds_test)
        batch_errors = []
        plasticity_weight = 0.0
        if epoch >= plasticity_start_epoch:
            plasticity_weight = ((epoch - plasticity_start_epoch + 1) /
                                 (num_epochs - plasticity_start_epoch + 1))
            plasticity_weight *= global_plasticity_weight_multiplier.numpy()
            
        # -----------------------
        # Training Loop.
        # -----------------------
        for step in range(train_steps):
            images, labels = next(train_iter)        
            with tf.GradientTape(persistent=False) as tape:
                predictions, hidden, reconstruction, uncertainty, latent = model(images, training=True)
                # hidden, reconstruction = model.unsupervised_extractor(images, training=True)
                
                loss = loss_fn(labels, predictions)
                # Unsupervised reconstruction loss.
                recon_loss = recon_loss_fn(images, reconstruction)
                
                # Learn uncertainty dynamically.
                target_uncertainty = tf.reduce_mean(uncertainty)
                uncertainty_loss = uncertainty_loss_fn(target_uncertainty, uncertainty)
                
                # Combine losses as needed (weights can be tuned).
                total_loss = loss + 0.1 * recon_loss + 0.05 * uncertainty_loss                            
                
            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
            # ae_grads = tape.gradient(recon_loss, model.unsupervised_extractor.trainable_variables)
            # ae_optimizer.apply_gradients(zip(ae_grads, model.unsupervised_extractor.trainable_variables))
            
            batch_errors.append(loss.numpy())
            train_loss_metric(loss)
            train_accuracy_metric(labels, predictions)
            train_recon_loss_metric(recon_loss)
            
            # Compute activities for plasticity update.
            x_flat = model.flatten(images)
            pre_activity = tf.reduce_mean(x_flat, axis=0)
            post_activity = tf.reduce_mean(hidden, axis=0)
            
            # Update reward: including unsupervised reconstruction loss.
            reward = global_reward_scaling_factor.numpy() * \
                     compute_reinforcement_signals(loss, recon_loss, predictions, hidden, reward)
            
            if plasticity_weight > 0 and global_step % plasticity_update_interval == 0:
                for expert in model.hidden.experts:
                    expert_reward = compute_expert_reward(images, reconstruction, labels, predictions)
                    base_neuromod_signal = compute_neuromodulatory_signal(predictions, expert_reward)
                    latent_signal = tf.reduce_mean(compute_latent_mod(latent))
                    neuromod_signal = base_neuromod_signal * latent_signal
                    plasticity_delta = expert.plasticity_update(pre_activity, post_activity, expert_reward)
                    if tf.reduce_sum(tf.abs(plasticity_delta)) > 0:
                        plasticity_delta *= plasticity_weight * neuromod_signal
                        plasticity_delta = tf.clip_by_norm(plasticity_delta, clip_norm=0.05)
                        expert.apply_plasticity(plasticity_delta)
            
            if global_step % homeostasis_interval == 0:
                for expert in model.hidden.experts:
                    expert.apply_homeostatic_scaling()
                    expert.apply_structural_plasticity()
                
            if global_step % architecture_update_interval == 0:
                for expert in model.hidden.experts:
                    expert.apply_architecture_modification()
                
            global_step += 1
        
        # Logging training metrics.
        mean_error = np.mean(batch_errors)
        std_error = np.std(batch_errors)
        print(f"\nEpoch {epoch+1}/{num_epochs}\n"
              f"Train Loss : {train_loss_metric.result():.4f}, "
              f"Train Accuracy : {train_accuracy_metric.result():.4f}, "
              f"Train Recon Loss : {train_recon_loss_metric.result():.4f}")

        # Reset training metrics.
        train_loss_metric.reset_state()
        train_accuracy_metric.reset_state()
        train_recon_loss_metric.reset_state()
        
        # -----------------------
        # Validation Evaluation.
        # -----------------------
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

        # Reset validation metrics.
        val_loss_metric.reset_state()
        val_accuracy_metric.reset_state()
        val_recon_loss_metric.reset_state()
        
        # -----------------------
        # Test Evaluation.
        # -----------------------
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
        
        # ---------------
        # Early Stopping Check.
        # ---------------
        if current_val_loss < best_val_loss - early_stop_min_delta:
            best_val_loss = current_val_loss
            patience_counter = 0
            best_weights = model.get_weights()
            print("\nValidation loss improved; resetting patience counter.")
        else:
            patience_counter += 1
            print(f"\nNo improvement in validation loss for {patience_counter} epoch(s).")
            if patience_counter >= early_stop_patience:
                #print("Early stopping triggered. Restoring best weights and ending training.\n")
                if best_weights is not None:
                    print("Restoring best weights and saving model.\n")
                    model.set_weights(best_weights)
                    model.save("models/MetaSynapse_NextGen_v5d.keras")
                    model.unsupervised_extractor.save("models/unsupervised_extractor_v5d.keras")
                break  # Exit the epoch loop.

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
    batch_size = 32
    max_units = 256
    initial_units = 64
    cnn_units = 128
    epochs = 1000
    experts = 8
    h = 16
    w = 16
    window_size = h*w
    learning_rate=1e-3
    
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
    model = PlasticityModelMoE(h, w, plasticity_controller, units=cnn_units, 
                               num_experts=experts, max_units=max_units, initial_units=initial_units, num_classes=10)
    ae_model = model.unsupervised_extractor
    dummy_input = tf.zeros((1, h, w, 1))
    _ = model(dummy_input, training=False)
    _ = ae_model(dummy_input, training=False)
    model.summary()
        
    # Train the model.
    train_model(model, ds_train, ds_val, ds_test, train_steps, val_steps, test_steps, num_epochs=epochs,
                homeostasis_interval=13, architecture_update_interval=55,
                plasticity_update_interval=8, plasticity_start_epoch=3, learning_rate=learning_rate)
    
if __name__ == '__main__':
    main()
