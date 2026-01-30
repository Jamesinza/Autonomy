import tensorflow as tf
import numpy as np
import pandas as pd
import random

# Set seeds for reproducibility.
np.random.seed(42)
tf.random.set_seed(42)

# =============================================================================
# 1. Learned Activation Function (as before)
# =============================================================================
class LearnedActivation(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LearnedActivation, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.w = self.add_weight(name='activation_weights', shape=(6,),
                                 initializer='ones', trainable=True)
        super(LearnedActivation, self).build(input_shape)

    def call(self, inputs):
        sig = tf.keras.activations.sigmoid(inputs)
        elu = tf.keras.activations.elu(inputs)
        tanh = tf.keras.activations.tanh(inputs)
        relu = tf.keras.activations.relu(inputs)
        silu = tf.keras.activations.silu(inputs)
        gelu = tf.keras.activations.gelu(inputs)
        weights = tf.nn.softmax(self.w)
        results = (weights[0]*sig + weights[1]*elu + weights[2]*tanh +
                   weights[3]*relu + weights[4]*silu + weights[5]*gelu)
        return results

# =============================================================================
# 2. Recurrent Plasticity Controller: Meta-learning the plasticity rules.
# =============================================================================
class RecurrentPlasticityController(tf.keras.layers.Layer):
    def __init__(self, units=16, **kwargs):
        super(RecurrentPlasticityController, self).__init__(**kwargs)
        # Use a full GRU layer (which expects a time dimension) instead of GRUCell.
        self.gru = tf.keras.layers.GRU(units, return_sequences=False)
        self.dense = tf.keras.layers.Dense(2, activation=None)

    def call(self, inputs):
        # inputs shape: (batch, feature_dim); we add a time dimension.
        inputs_expanded = tf.expand_dims(inputs, axis=1)  # now (batch, 1, feature_dim)
        output = self.gru(inputs_expanded)
        adjustments = self.dense(output)  # produces two values: [delta_add, delta_mult]
        return adjustments

# =============================================================================
# 3. Unsupervised Feature Extractor: A simple autoencoder for latent representations.
# =============================================================================
class UnsupervisedFeatureExtractor(tf.keras.Model):
    def __init__(self, latent_dim=32, **kwargs):
        super(UnsupervisedFeatureExtractor, self).__init__(**kwargs)
        # Encoder: a few Dense layers.
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation=LearnedActivation()),
            tf.keras.layers.Dense(latent_dim, activation=LearnedActivation())
        ])
        # Decoder: mirror of encoder.
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation=LearnedActivation()),
            tf.keras.layers.Dense(8 * 8, activation='sigmoid'),
            tf.keras.layers.Reshape((8, 8, 1))
        ])
        
    def call(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return latent, reconstruction

# =============================================================================
# 4. Dynamic Connectivity: Using learned activation and unsupervised latent cues.
# =============================================================================
class DynamicConnectivity(tf.keras.layers.Layer):
    def __init__(self, max_units, **kwargs):
        super(DynamicConnectivity, self).__init__(**kwargs)
        self.max_units = max_units
        self.attention_net = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation=LearnedActivation()),
            tf.keras.layers.Dense(max_units, activation='sigmoid')  # Outputs in [0,1]
        ])
        
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
        
        # For homeostasis.
        self.target_avg = tf.Variable(0.2, trainable=False, dtype=tf.float32)        
        
        # Create a mask for active neurons. 1 means active, 0 inactive.
        init_mask = [1] * initial_units + [0] * (max_units - initial_units)
        self.neuron_mask = tf.Variable(init_mask, trainable=False, dtype=tf.float32)
        
        # Running average activation (per neuron)
        self.neuron_activation_avg = tf.Variable(tf.zeros([max_units], dtype=tf.float32),
                                                 trainable=False)
        
    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        self.w = self.add_weight(shape=(input_dim, self.max_units),
                                 initializer='glorot_uniform',
                                 trainable=True, name='kernel')
        self.b = self.add_weight(shape=(self.max_units,),
                                 initializer='zeros',
                                 trainable=True, name='bias')
        super(DynamicPlasticDenseAdvancedHyperV2, self).build(input_shape)
    
    def call(self, inputs):
        # Linear combination.
        z = tf.matmul(inputs, self.w) + self.b  # shape: (batch, max_units)
        # Compute connectivity modulation based on running activation.
        connectivity = self.dynamic_connectivity(tf.expand_dims(self.neuron_activation_avg, axis=0))
        # Multiply with the current neuron mask.
        mask = tf.expand_dims(self.neuron_mask, axis=0)
        z_mod = z * connectivity * mask
        a = tf.keras.activations.relu(z_mod)
        
        # Update running average activation.
        batch_mean = tf.reduce_mean(a, axis=0)
        alpha = 0.9
        new_avg = alpha * self.neuron_activation_avg + (1 - alpha) * batch_mean
        self.neuron_activation_avg.assign(new_avg)
        
        return a

    def plasticity_update(self, pre_activity, post_activity, reward):
        """
        Compute plasticity update for self.w using the recurrent controller.
        Instead of a per-connection update, here we compute a global adjustment factor.
        """
        # Compute summary statistics.
        pre_mean = tf.reduce_mean(pre_activity)
        post_mean = tf.reduce_mean(post_activity)
        weight_mean = tf.reduce_mean(self.w)
        # Create a feature vector (batch=1, feature_dim=4).
        features = tf.reshape(tf.stack([pre_mean, post_mean, weight_mean, tf.cast(reward, tf.float32)]), (1, 4))
        adjustments = self.plasticity_controller(features)  # shape (1,2)
        delta_add, delta_mult = adjustments[0, 0], adjustments[0, 1]
        plasticity_delta = tf.cast(reward, tf.float32) * (delta_add + delta_mult * self.w)
        return plasticity_delta
    
    def apply_plasticity(self, plasticity_delta):
        self.w.assign_add(plasticity_delta)
    
    def apply_homeostatic_scaling(self):
        avg_w = tf.reduce_mean(tf.abs(self.w))
        new_target = self.decay_factor * self.target_avg + (1 - self.decay_factor) * avg_w
        scaling_factor = new_target / (avg_w + 1e-6)
        self.w.assign(self.w * scaling_factor)
        self.avg_weight_magnitude.assign(avg_w)
    
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

    def adjust_prune_threshold(self):
        if self.sparsity > 0.8:
            self.prune_threshold.assign(self.prune_threshold * 1.05)
        elif self.sparsity < 0.3:
            self.prune_threshold.assign(self.prune_threshold * 0.95)

    def adjust_add_prob(self):
        if self.avg_weight_magnitude < 0.01:
            self.add_prob.assign(self.add_prob * 1.05)
        elif self.avg_weight_magnitude > 0.1:
            self.add_prob.assign(self.add_prob * 0.95)
    
    def apply_architecture_modification(self):
        """
        Update the neuron mask:
         - Prune neurons (set mask=0) if running average is below threshold.
         - Grow (activate) neurons if the mean activation of active neurons exceeds a growth threshold.
        """
        current_mask = self.neuron_mask.numpy()
        current_avg = self.neuron_activation_avg.numpy()
        # Prune: deactivate neurons with low activation.
        for i in range(self.max_units):
            if current_mask[i] == 1 and current_avg[i] < self.prune_activation_threshold:
                current_mask[i] = 0
        # Growth: if active neurons are very “busy”, try to activate one more.
        active_idx = np.where(np.array(current_mask)==1)[0]
        if len(active_idx) > 0 and np.mean(current_avg[active_idx]) > self.growth_activation_threshold:
            inactive_idx = np.where(np.array(current_mask)==0)[0]
            if len(inactive_idx) > 0:
                # Activate one random inactive neuron.
                idx = np.random.choice(inactive_idx)
                current_mask[idx] = 1
        self.neuron_mask.assign(np.array(current_mask, dtype=np.float32))

# =============================================================================
# 6. Full Model with Integrated Unsupervised Branch and Dynamic Plastic Dense Layer.
# =============================================================================
class PlasticityModelHyperV2(tf.keras.Model):
    def __init__(self, plasticity_controller, max_units=256, initial_units=128, num_classes=10, **kwargs):
        super(PlasticityModelHyperV2, self).__init__(**kwargs)
        self.unsupervised_extractor = UnsupervisedFeatureExtractor(latent_dim=32)
        self.flatten = tf.keras.layers.Flatten()
        self.hidden = DynamicPlasticDenseAdvancedHyperV2(max_units, initial_units, plasticity_controller)
        self.out = tf.keras.layers.Dense(num_classes, activation='softmax')
    
    def call(self, x, training=False):
        # Get latent representation and reconstruction (for unsupervised loss).
        latent, reconstruction = self.unsupervised_extractor(x)
        x_flat = self.flatten(x)
        hidden_act = self.hidden(x_flat)
        output = self.out(hidden_act)
        return output, hidden_act, reconstruction

# =============================================================================
# 7. Training Loop: Incorporates classification loss, unsupervised (reconstruction) loss,
#    recurrent meta-plasticity updates (with unsupervised influence), chaotic modulation,
#    and architecture modifications.
# =============================================================================
def train_model(model, ds_train, ds_val, ds_test, num_epochs=1000,
                homeostasis_interval=100, architecture_update_interval=100,
                plasticity_update_interval=10, plasticity_start_epoch=3):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    recon_loss_fn = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001)
    
    # Global variables controlled by the recurrent plasticity controller.
    global_plasticity_weight_multiplier = tf.Variable(0.5, trainable=False, dtype=tf.float32)
    global_reward_scaling_factor = tf.Variable(0.1, trainable=False, dtype=tf.float32)
    
    train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    
    # Metrics for validation and test
    val_loss_metric = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
    test_loss_metric = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    
    reward = 0.0
    global_step = 0
    for epoch in range(num_epochs):
        batch_errors = []
        plasticity_weight = 0.0
        if epoch >= plasticity_start_epoch:
            plasticity_weight = ((epoch - plasticity_start_epoch + 1) /
                                 (num_epochs - plasticity_start_epoch + 1))
            plasticity_weight *= global_plasticity_weight_multiplier.numpy()
            
        # Training loop.
        for images, labels in ds_train:
            with tf.GradientTape() as tape:
                predictions, hidden, reconstruction = model(images, training=True)
                loss = loss_fn(labels, predictions)
                # Unsupervised reconstruction loss.
                recon_loss = recon_loss_fn(images, reconstruction)
                total_loss = loss + 0.1 * recon_loss  # weighted unsupervised loss
                
            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
            batch_errors.append(loss.numpy())
            train_loss_metric(loss)
            train_accuracy_metric(labels, predictions)
            
            # Compute activities for plasticity update.
            x_flat = model.flatten(images)
            pre_activity = tf.reduce_mean(x_flat, axis=0)
            post_activity = tf.reduce_mean(hidden, axis=0)
            
            # Update reward: now including unsupervised reconstruction loss.
            reward = global_reward_scaling_factor.numpy() * \
                     compute_reinforcement_signals(loss, recon_loss, predictions, hidden, reward)
            
            if plasticity_weight > 0 and global_step % plasticity_update_interval == 0:
                chaos_state = chaotic_modulation(0.5)  # starting chaos state (can be updated)
                neuromod_signal = compute_neuromodulatory_signal(predictions, reward)
                plasticity_delta = model.hidden.plasticity_update(pre_activity, post_activity, reward)
                # Modulate the plasticity update.
                plasticity_delta *= plasticity_weight * chaos_state * neuromod_signal
                model.hidden.apply_plasticity(plasticity_delta)
            
            if global_step % homeostasis_interval == 0:
                model.hidden.apply_homeostatic_scaling()
                model.hidden.apply_structural_plasticity()
                
            if global_step % architecture_update_interval == 0:
                model.hidden.apply_architecture_modification()
                
            global_step += 1
        
        # Logging training metrics.
        mean_error = np.mean(batch_errors)
        std_error = np.std(batch_errors)
        print(f"Epoch {epoch+1}/{num_epochs}\n"
              f"Train Loss : {train_loss_metric.result():.4f}, "
              f"Train Accuracy : {train_accuracy_metric.result():.4f}, "
              )
        
        # Reset training metrics.
        train_loss_metric.reset_state()
        train_accuracy_metric.reset_state()
        
        # -----------------------
        # Validation Evaluation.
        # -----------------------
        for val_images, val_labels in ds_val:
            val_predictions, _, _ = model(val_images, training=False)
            val_loss = loss_fn(val_labels, val_predictions)
            val_loss_metric(val_loss)
            val_accuracy_metric(val_labels, val_predictions)
        
        print(f"Val Loss   : {val_loss_metric.result():.4f}, "
              f"Val Accuracy   : {val_accuracy_metric.result():.4f}")
        val_loss_metric.reset_state()
        val_accuracy_metric.reset_state()
        
        # -----------------------
        # Test Evaluation.
        # -----------------------
        for test_images, test_labels in ds_test:
            test_predictions, _, _ = model(test_images, training=False)
            test_loss = loss_fn(test_labels, test_predictions)
            test_loss_metric(test_loss)
            test_accuracy_metric(test_labels, test_predictions)
            
        print(f"Test Loss  : {test_loss_metric.result():.4f}, "
              f"Test Accuracy  : {test_accuracy_metric.result():.4f}\n")
        test_loss_metric.reset_state()
        test_accuracy_metric.reset_state()
    
    model.save("models/MetaSynapse_NextGen_v2.keras")

    
# =============================================================================
# 8. Helper functions for chaotic modulation, neuromodulatory signal, and reinforcement signals.
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

def create_tf_dataset(inputs, labels, batch_size=128, shuffle=False, repeat_epochs=1):
    inputs = inputs.astype(np.float32) / 9.0
    inputs = inputs.reshape((-1, 8, 8, 1))
    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.repeat(repeat_epochs).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# =============================================================================
# 10. Main function: Build dataset, instantiate controllers and model, then train.
# =============================================================================
def main():
    batch_size = 128
    max_units = 256
    initial_units = 128
    window_size = 64
    # Load data.
    sequence = get_real_data(num_samples=10_000)
    inputs, labels = create_windows(sequence, window_size=window_size+1)
    (inp_train, lbl_train), (inp_val, lbl_val), (inp_test, lbl_test) = split_dataset(inputs, labels)
    
    ds_train = create_tf_dataset(inp_train, lbl_train, batch_size=batch_size, shuffle=False)
    ds_val = create_tf_dataset(inp_val, lbl_val, batch_size=batch_size, shuffle=False)
    ds_test = create_tf_dataset(inp_test, lbl_test, batch_size=batch_size, shuffle=False)
    
    # Instantiate the recurrent plasticity controller.
    plasticity_controller = RecurrentPlasticityController(units=16)
    _ = plasticity_controller(tf.zeros((1,4)))  # Warm up
    
    # Instantiate the model.
    model = PlasticityModelHyperV2(plasticity_controller, max_units=max_units,
                                   initial_units=initial_units, num_classes=10)
    dummy_input = tf.zeros((1, 8, 8, 1))
    _ = model(dummy_input, training=False)
    model.summary()
    
    # Train the model.
    train_model(model, ds_train, ds_val, ds_test, num_epochs=1000,
                homeostasis_interval=100, architecture_update_interval=100,
                plasticity_update_interval=5, plasticity_start_epoch=5)
    
if __name__ == '__main__':
    main()
