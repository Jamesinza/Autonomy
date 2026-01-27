import tensorflow as tf
import numpy as np
import pandas as pd
import collections

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# -----------------------------------------------------------------------------
# 1. LearnedActivation remains as before.
# -----------------------------------------------------------------------------
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
        
    def compute_output_shape(self, input_shape):
        return input_shape

# -----------------------------------------------------------------------------
# 2. Hypernetwork for Meta-Plasticity.
# -----------------------------------------------------------------------------
class DynamicPlasticityHypernetwork(tf.keras.layers.Layer):
    def __init__(self, hidden_units=32, **kwargs):
        super(DynamicPlasticityHypernetwork, self).__init__(**kwargs)
        self.dense1 = tf.keras.layers.Dense(hidden_units, activation=LearnedActivation())
        self.dense2 = tf.keras.layers.Dense(hidden_units, activation=LearnedActivation())
        # Output two parameters per connection: additive and multiplicative factors.
        self.out = tf.keras.layers.Dense(2, activation=None)
        
    def call(self, features):
        x = self.dense1(features)
        x = self.dense2(x)
        return self.out(x)
        
    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = 2
        return tuple(output_shape)

# -----------------------------------------------------------------------------
# 3. DynamicPlasticDenseAdvancedHyper: A dense layer with self-modifying architecture.
# -----------------------------------------------------------------------------
class DynamicPlasticDenseAdvancedHyper(tf.keras.layers.Layer):
    def __init__(self, max_units, initial_units, hypernetwork, 
                 initial_target=0.2, decay_factor=0.9, **kwargs):
        """
        max_units: maximum number of candidate neurons.
        initial_units: number of neurons active at initialization.
        """
        super(DynamicPlasticDenseAdvancedHyper, self).__init__(**kwargs)
        self.max_units = max_units
        self.initial_units = initial_units
        self.hypernetwork = hypernetwork
        self.decay_factor = decay_factor
        self.initial_target = initial_target
        
        # Create a binary mask for active neurons: 1 = active, 0 = inactive.
        init_mask = [1] * initial_units + [0] * (max_units - initial_units)
        self.neuron_mask = tf.Variable(init_mask, trainable=False, dtype=tf.float32)
        
        # Running average activation for each neuron (used by the meta-controller).
        self.neuron_activation_avg = tf.Variable(tf.zeros([max_units], dtype=tf.float32),
                                                 trainable=False)
        # Threshold for deactivating neurons.
        self.prune_activation_threshold = 0.01
        # Minimum desired number of active neurons.
        self.desired_min_active = initial_units

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        self.w = self.add_weight(shape=(input_dim, self.max_units),
                                 initializer='glorot_uniform',
                                 trainable=True, name='kernel')
        self.b = self.add_weight(shape=(self.max_units,),
                                 initializer='zeros',
                                 trainable=True, name='bias')
        super(DynamicPlasticDenseAdvancedHyper, self).build(input_shape)
    
    def call(self, inputs):
        # Compute linear combination for all candidate neurons.
        z = tf.matmul(inputs, self.w) + self.b  # shape: (batch, max_units)
        # Mask out inactive neurons.
        z_masked = z * self.neuron_mask
        # Apply activation (ReLU in this example).
        a = tf.keras.activations.relu(z_masked)
        
        # Update running average activation per neuron (simple EMA).
        batch_mean = tf.reduce_mean(a, axis=0)  # shape: (max_units,)
        alpha = 0.9
        new_avg = alpha * self.neuron_activation_avg + (1 - alpha) * batch_mean
        self.neuron_activation_avg.assign(new_avg)
        
        return a

    def plasticity_update(self, pre_activity, post_activity, reward):
        # Meta-plasticity update: compute features for each connection.
        in_features = tf.shape(self.w)[0]
        total_neurons = self.max_units
        pre_broadcast = tf.reshape(pre_activity, (-1, 1))
        pre_tile = tf.tile(pre_broadcast, [1, total_neurons])
        post_broadcast = tf.reshape(post_activity, (1, -1))
        post_tile = tf.tile(post_broadcast, [in_features, 1])
        w_val = self.w
        random_delta = tf.random.uniform(tf.shape(self.w), minval=-20.0, maxval=20.0)
        features = tf.stack([pre_tile, post_tile, w_val, random_delta], axis=-1)  # shape: (in_features, total_neurons, 4)
        features_flat = tf.reshape(features, [-1, 4])
        hyper_params_flat = self.hypernetwork(features_flat)  # shape: (in_features * total_neurons, 2)
        # Fix: Append the new dimension correctly.
        new_shape = tf.concat([tf.shape(self.w), [2]], axis=0)
        hyper_params = tf.reshape(hyper_params_flat, new_shape)
        delta_add = hyper_params[..., 0]
        delta_mult = hyper_params[..., 1]
        plasticity_delta = tf.cast(reward, tf.float32) * (delta_add + delta_mult * self.w)
        return plasticity_delta
    
    def apply_plasticity(self, plasticity_delta):
        self.w.assign_add(plasticity_delta)
    
    def apply_homeostatic_scaling(self):
        avg_w = tf.reduce_mean(tf.abs(self.w))
        new_target = self.decay_factor * self.initial_target + (1 - self.decay_factor) * avg_w
        scaling_factor = new_target / (avg_w + 1e-6)
        self.w.assign(self.w * scaling_factor)
    
    def apply_structural_plasticity(self):
        # (Optional) Prune individual connections that are too small.
        pruned_w = tf.where(tf.abs(self.w) < 0.01, tf.zeros_like(self.w), self.w)
        self.w.assign(pruned_w)
    
    def apply_architecture_modification(self):
        """
        Meta-controller: adjust the neuron_mask based on running average activations.
          - Deactivate neurons with average activation below a threshold.
          - Reactivate neurons if the count of active neurons falls below a desired minimum.
        """
        current_mask = self.neuron_mask.numpy()
        current_avg = self.neuron_activation_avg.numpy()
        
        # Deactivate (prune) neurons with low activation.
        for i in range(self.max_units):
            if current_mask[i] == 1 and current_avg[i] < self.prune_activation_threshold:
                current_mask[i] = 0  # deactivate neuron
                
        # Count active neurons.
        active_count = int(np.sum(current_mask))
        
        # If active count is below the desired minimum, activate additional neurons.
        if active_count < self.desired_min_active:
            for i in range(self.max_units):
                if current_mask[i] == 0:
                    current_mask[i] = 1
                    active_count += 1
                    if active_count >= self.desired_min_active:
                        break
        
        self.neuron_mask.assign(np.array(current_mask, dtype=np.float32))

# -----------------------------------------------------------------------------
# 4. Model definition using the dynamic plastic dense layer.
# -----------------------------------------------------------------------------
class PlasticityModelHyper(tf.keras.Model):
    def __init__(self, hypernetwork, max_units=256, initial_units=128, num_classes=10):
        super(PlasticityModelHyper, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        # Use the dynamic plastic dense layer with self-modifying architecture.
        self.hidden = DynamicPlasticDenseAdvancedHyper(max_units, initial_units, hypernetwork)
        self.out = tf.keras.layers.Dense(num_classes, activation='softmax')
    
    def call(self, x, training=False):
        x = self.flatten(x)
        hidden_act = self.hidden(x)
        output = self.out(hidden_act)
        return output, hidden_act

# -----------------------------------------------------------------------------
# 5. Data loading functions.
# -----------------------------------------------------------------------------
def get_real_data(num_samples):
    # Example: load a dataset (ensure the CSV exists at the specified path)
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
    # Example reshape: adjust according to your specific input dimensions.
    inputs = inputs.reshape((-1, 8, 8, 1))
    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.repeat(repeat_epochs).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# -----------------------------------------------------------------------------
# 6. Reinforcement Signals Beyond Loss.
# -----------------------------------------------------------------------------
def compute_reinforcement_signals(loss, predictions, hidden, external_reward=0.0):
    """
    Combines multiple signals into a reinforcement reward.
      - Loss-based signal: tf.sigmoid(-loss)
      - Novelty signal: standard deviation of hidden activations.
      - Uncertainty signal: average entropy of predictions.
      - External reward can be added.
    """
    novelty = tf.math.reduce_std(hidden)
    entropy = -tf.reduce_sum(predictions * tf.math.log(predictions + 1e-6), axis=-1)
    avg_uncertainty = tf.reduce_mean(entropy)
    loss_signal = tf.sigmoid(-loss)
    novelty_signal = 0.5 * tf.sigmoid(novelty)
    uncertainty_signal = 0.5 * tf.sigmoid(avg_uncertainty)
    combined_reward = loss_signal + novelty_signal + uncertainty_signal + external_reward
    return combined_reward

# -----------------------------------------------------------------------------
# 7. MetaPlasticityController: Nested meta-controller for hyperparameter tuning.
# -----------------------------------------------------------------------------
class MetaPlasticityController(tf.keras.layers.Layer):
    def __init__(self, hidden_units=16, **kwargs):
        super(MetaPlasticityController, self).__init__(**kwargs)
        self.dense1 = tf.keras.layers.Dense(hidden_units, activation='relu')
        self.dense2 = tf.keras.layers.Dense(hidden_units, activation='relu')
        # Output adjustments for [plasticity_multiplier_delta, reward_scaling_delta]
        self.out = tf.keras.layers.Dense(2, activation='tanh')  # tanh to bound adjustments

    def call(self, inputs):
        # inputs: a vector with [mean_error, std_error, current_multiplier, current_reward_scaling]
        x = self.dense1(inputs)
        x = self.dense2(x)
        adjustments = self.out(x)
        return adjustments

# -----------------------------------------------------------------------------
# 8. Training loop with nested meta-plasticity.
# -----------------------------------------------------------------------------
def train_model(model, meta_controller, ds_train, ds_val, ds_test, num_epochs=1000,
                homeostasis_interval=100, architecture_update_interval=100,
                plasticity_start_epoch=3, plasticity_update_interval=10):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001)
    
    # Initialize plasticity weight and reward scaling (now controlled by meta_controller).
    global_plasticity_weight_multiplier = tf.Variable(0.5, trainable=False, dtype=tf.float32)
    global_reward_scaling_factor = tf.Variable(0.1, trainable=False, dtype=tf.float32)

    # Define optimal target values for mean and std of plasticity delta.
    MEAN_MIN, MEAN_MAX = 1e-4, 1e-3
    STD_MIN, STD_MAX = 5e-4, 1e-2
    OPTIMAL_MEAN = (MEAN_MIN + MEAN_MAX) / 2
    OPTIMAL_STD = (STD_MIN + STD_MAX) / 2

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    
    global_step = 0
    for epoch in range(num_epochs):
        plasticity_weight = ((epoch - plasticity_start_epoch + 1) / 
                             (num_epochs - plasticity_start_epoch + 1)) if epoch >= plasticity_start_epoch else 0.0
        plasticity_weight *= global_plasticity_weight_multiplier.numpy()  # use current multiplier

        for images, labels in ds_train:
            with tf.GradientTape() as tape:
                predictions, hidden = model(images, training=True)
                loss = loss_fn(labels, predictions)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
            train_loss(loss)
            train_accuracy(labels, predictions)
            
            x_flat = model.flatten(images)
            pre_activity = tf.reduce_mean(x_flat, axis=0)
            post_activity = tf.reduce_mean(hidden, axis=0)
            
            # Compute combined reward.
            reward = global_reward_scaling_factor.numpy() * compute_reinforcement_signals(loss, predictions, hidden, external_reward=0.0)
            
            if plasticity_weight > 0 and global_step % plasticity_update_interval == 0:
                plasticity_delta = model.hidden.plasticity_update(pre_activity, post_activity, reward)
                plasticity_delta *= plasticity_weight
                model.hidden.apply_plasticity(plasticity_delta)
                
                # Compute mean and std of plasticity updates.
                delta_mean = tf.reduce_mean(tf.abs(plasticity_delta))
                delta_std = tf.math.reduce_std(plasticity_delta)

                # Calculate errors with respect to optimal values.
                mean_error = delta_mean - OPTIMAL_MEAN
                std_error = delta_std - OPTIMAL_STD

                # Use the meta controller to get adjustments.
                meta_input = tf.convert_to_tensor([[mean_error, std_error,
                                                     global_plasticity_weight_multiplier.numpy(),
                                                     global_reward_scaling_factor.numpy()]], dtype=tf.float32)
                adjustments = meta_controller(meta_input)
                # adjustments are in [-1,1]; we scale them to a small update
                adjust_factor = 0.05
                new_multiplier = global_plasticity_weight_multiplier * (1.0 + adjust_factor * adjustments[0, 0])
                new_reward_scaling = global_reward_scaling_factor * (1.0 + adjust_factor * adjustments[0, 1])
                
                # Clamp the values to reasonable ranges.
                new_multiplier = tf.clip_by_value(new_multiplier, 0.01, 5.0)
                new_reward_scaling = tf.clip_by_value(new_reward_scaling, 0.01, 5.0)
                
                global_plasticity_weight_multiplier.assign(new_multiplier)
                global_reward_scaling_factor.assign(new_reward_scaling)
            
            if global_step % homeostasis_interval == 0:
                model.hidden.apply_homeostatic_scaling()
                model.hidden.apply_structural_plasticity()
            
            if global_step % architecture_update_interval == 0:
                model.hidden.apply_architecture_modification()
            
            global_step += 1
        
        print(f"\nEpoch {epoch+1}/{num_epochs}\nTrain Loss: {train_loss.result():.4f}, "
              f"Train Accuracy: {train_accuracy.result():.4f}, Plasticity Weight: {plasticity_weight:.6f}")
        train_loss.reset_state()
        train_accuracy.reset_state()
        
        # Validation Evaluation.
        val_loss_metric = tf.keras.metrics.Mean(name='val_loss')
        val_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
        for images, labels in ds_val:
            predictions, _ = model(images, training=False)
            loss_val = loss_fn(labels, predictions)
            val_loss_metric(loss_val)
            val_accuracy_metric(labels, predictions)
        print(f"Val Loss: {val_loss_metric.result():.4f}, Val Accuracy: {val_accuracy_metric.result():.4f}")
        
        # Test Evaluation.
        test_loss_metric = tf.keras.metrics.Mean(name='test_loss')
        test_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        for images, labels in ds_test:
            predictions, _ = model(images, training=False)
            loss_test = loss_fn(labels, predictions)
            test_loss_metric(loss_test)
            test_accuracy_metric(labels, predictions)
        print(f"Test Loss: {test_loss_metric.result():.4f}, Test Accuracy: {test_accuracy_metric.result():.4f}")
    
    model.save("models/MetaSynapse_vNested.keras")

# -----------------------------------------------------------------------------
# 9. Main: Create dataset, instantiate models, and train.
# -----------------------------------------------------------------------------
def main():
    batch_size = 128
    max_units = 512
    initial_units = 16
    # Load sequence data.
    sequence = get_real_data(num_samples=100_000)
    inputs, labels = create_windows(sequence, window_size=65)
    (inp_train, lbl_train), (inp_val, lbl_val), (inp_test, lbl_test) = split_dataset(inputs, labels)
    
    ds_train = create_tf_dataset(inp_train, lbl_train, batch_size=batch_size, shuffle=True)
    ds_val = create_tf_dataset(inp_val, lbl_val, batch_size=batch_size, shuffle=False)
    ds_test = create_tf_dataset(inp_test, lbl_test, batch_size=batch_size, shuffle=False)
    
    # Instantiate the hypernetwork.
    dummy_input = tf.zeros((1, 4))
    dynamic_hypernet = DynamicPlasticityHypernetwork(hidden_units=16)
    _ = dynamic_hypernet(dummy_input)
    
    # Instantiate the meta plasticity controller.
    meta_controller = MetaPlasticityController(hidden_units=16)
    dummy_meta = tf.zeros((1, 4))
    _ = meta_controller(dummy_meta)
    
    # Instantiate the model with the hypernetwork integrated.
    model = PlasticityModelHyper(hypernetwork=dynamic_hypernet, max_units=max_units,
                                  initial_units=initial_units, num_classes=10)
    
    # Build the model by passing a dummy input.
    dummy2 = tf.zeros((1, 8, 8, 1))
    _ = model(dummy2, training=False)
    model.summary()
    
    # Train the model.
    train_model(model, meta_controller, ds_train, ds_val, ds_test, num_epochs=1000,
                homeostasis_interval=100, architecture_update_interval=100)

if __name__ == '__main__':
    main()

