import tensorflow as tf
import numpy as np
import pandas as pd

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# =============================================================================
# 1. LearnedActivation remains as before.
# =============================================================================
class LearnedActivation(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
            
    def build(self, input_shape):
        self.w = self.add_weight(name='activation_weights', shape=(6,),
                                 initializer='ones', trainable=True)
        super().build(input_shape)

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
# 2. PlasticDenseAdvanced with recurrent meta-plasticity support.
# =============================================================================
class PlasticDenseAdvanced(tf.keras.layers.Layer):
    def __init__(self, units, initial_target=0.2, decay_factor=0.9, **kwargs):
        super(PlasticDenseAdvanced, self).__init__(**kwargs)
        self.units = units
        self.activation = LearnedActivation()
        self.decay_factor = decay_factor
        
        # Structural plasticity parameters.
        self.prune_threshold = tf.Variable(0.01, trainable=False, dtype=tf.float32)
        self.add_prob = tf.Variable(0.01, trainable=False, dtype=tf.float32)        
        
        # Meta-plasticity and STDP parameters.
        self.meta_lr = tf.Variable(0.01, trainable=True, dtype=tf.float32)
        self.A_plus = tf.Variable(0.01, trainable=True, dtype=tf.float32)
        self.A_minus = tf.Variable(0.01, trainable=True, dtype=tf.float32)
        
        # For homeostasis.
        self.target_avg = tf.Variable(initial_target, trainable=False, dtype=tf.float32)
        
        # Dynamic tau values for STDP decay.
        self.tau_plus = tf.Variable(20.0, trainable=True, dtype=tf.float32)
        self.tau_minus = tf.Variable(20.0, trainable=True, dtype=tf.float32)
        
        # Tracking variables.
        self.avg_weight_magnitude = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.sparsity = tf.Variable(0.0, trainable=False, dtype=tf.float32)

    def build(self, input_shape):
        self.w = self.add_weight(shape=(int(input_shape[-1]), self.units),
                                 initializer='glorot_uniform',
                                 trainable=True, name='kernel')
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros',
                                 trainable=True, name='bias')
        super(PlasticDenseAdvanced, self).build(input_shape)

    def call(self, inputs):
        z = tf.matmul(inputs, self.w) + self.b
        return self.activation(z)

    def plasticity_update(self, pre_activity, post_activity, reward):
        # Compute a random delta_t per weight element.
        delta_t = tf.random.uniform(tf.shape(self.w), minval=-20.0, maxval=20.0)
        # Use the dynamic tau values.
        stdp_update = tf.where(delta_t > 0,
                               self.A_plus * tf.exp(-delta_t / self.tau_plus),
                               -self.A_minus * tf.exp(delta_t / self.tau_minus))
        pre = tf.reshape(pre_activity, [-1, 1])
        post = tf.reshape(post_activity, [1, -1])
        hebbian_term = pre * post
        plasticity_delta = stdp_update * hebbian_term
        plasticity_delta = tf.cast(reward, tf.float32) * plasticity_delta
        return plasticity_delta

    def apply_plasticity(self, plasticity_delta, meta_multiplier):
        # The meta_multiplier is now provided from the recurrent meta-controller.
        # We update using the product of meta_lr and the meta output.
        self.w.assign_add(self.meta_lr * meta_multiplier * plasticity_delta)

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

# =============================================================================
# 3. Define a Recurrent Meta-Controller for dynamic plasticity.
# =============================================================================
class RecurrentMetaController(tf.keras.Model):
    def __init__(self, hidden_units=32, **kwargs):
        super(RecurrentMetaController, self).__init__(**kwargs)
        # Use a stateful RNN so that state is managed internally.
        self.rnn = tf.keras.layers.RNN(
            tf.keras.layers.LSTMCell(hidden_units),
            stateful=True,  # Enables persistence of state across calls.
            return_sequences=False,
            return_state=False
        )
        self.out_dense = tf.keras.layers.Dense(1, activation='softplus')

    def call(self, inputs):
        # Ensure the inputs have shape (batch_size, time_steps, features).
        # If inputs are (batch_size, features), add a time dimension.
        if len(inputs.shape) == 2:
            inputs = tf.expand_dims(inputs, axis=1)
        x = self.rnn(inputs)
        multiplier = self.out_dense(x)
        return tf.squeeze(multiplier, axis=-1)


# =============================================================================
# 4. Define the end-to-end model for classification.
# =============================================================================
class PlasticityModel(tf.keras.Model):
    def __init__(self, num_hidden=128, num_classes=10):
        super(PlasticityModel, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.hidden = PlasticDenseAdvanced(num_hidden)
        self.drop = tf.keras.layers.Dropout(0.3)
        self.out = tf.keras.layers.Dense(num_classes, activation='softmax')
    
    def call(self, x, training=False):
        x = self.flatten(x)
        hidden_act = self.hidden(x)
        drop = self.drop(hidden_act, training=training)
        output = self.out(drop)
        return output, hidden_act

# =============================================================================
# 5. Data loading functions (as before).
# =============================================================================
def get_real_data(num_samples):
    dataset = 'Take5'
    df = pd.read_csv(f'datasets/{dataset}_Full.csv')
    target_df = df.head(num_samples//10).copy()
    cols = ['A','B','C','D','E']
    target_df = target_df[cols].dropna().astype(np.int8)
    target_df = target_df.map(lambda x: f'{x:02d}')
    flattened = target_df.values.flatten()
    full_data = np.array([int(d) for num in flattened for d in str(num)], dtype=np.int8)
    return full_data

def create_windows(sequence, window_size=33):
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
# 6. Training loop with recurrent meta-controller integration.
# =============================================================================
def train_model(model, meta_controller, ds_train, ds_val, ds_test, num_epochs=5, homeostasis_interval=100):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    
    global_step = 0
    # For tracking moving magnitude of plasticity deltas.
    moving_delta = tf.Variable(0.0, trainable=False, dtype=tf.float32)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        for images, labels in ds_train:
            with tf.GradientTape() as tape:
                predictions, hidden = model(images, training=True)
                loss = loss_fn(labels, predictions)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
            train_loss(loss)
            train_accuracy(labels, predictions)
            
            # Compute pre and post activations.
            x_flat = model.flatten(images)
            pre_activity = tf.reduce_mean(x_flat, axis=0)
            post_activity = tf.reduce_mean(hidden, axis=0)
            reward = tf.sigmoid(-loss)
            
            plasticity_delta = model.hidden.plasticity_update(pre_activity, post_activity, reward)
            # Update moving average of plasticity magnitude.
            moving_delta.assign(0.99 * moving_delta + 0.01 * tf.reduce_mean(tf.abs(plasticity_delta)))
            
            # Build meta-controller input vector: shape (batch_size, features)
            meta_input = tf.stack([loss, model.hidden.avg_weight_magnitude, moving_delta])
            meta_input = tf.expand_dims(meta_input, axis=0)  # shape: (1, 3)
            meta_input = tf.expand_dims(meta_input, axis=1)  # now shape: (1, 1, 3) as required
            multiplier = meta_controller(meta_input)
            
            # Apply plasticity update using the meta multiplier.
            model.hidden.apply_plasticity(plasticity_delta, multiplier)
            
            if global_step % homeostasis_interval == 0:
                model.hidden.apply_homeostatic_scaling()
                model.hidden.apply_structural_plasticity()
            
            global_step += 1
        
        print(f"Train Loss : {train_loss.result():.4f}, Train Accuracy : {train_accuracy.result():.4f}")
        train_loss.reset_state()
        train_accuracy.reset_state()
        
        # Validation evaluation.
        val_loss_metric = tf.keras.metrics.Mean(name='val_loss')
        val_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
        for images, labels in ds_val:
            predictions, _ = model(images, training=False)
            loss_val = loss_fn(labels, predictions)
            val_loss_metric(loss_val)
            val_accuracy_metric(labels, predictions)
        print(f"Val Loss   : {val_loss_metric.result():.4f}, Val Accuracy   : {val_accuracy_metric.result():.4f}")
        
        # Testing evaluation.
        test_loss_metric = tf.keras.metrics.Mean(name='test_loss')
        test_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        for images, labels in ds_test:
            predictions, _ = model(images, training=False)
            loss_test = loss_fn(labels, predictions)
            test_loss_metric(loss_test)
            test_accuracy_metric(labels, predictions)
        print(f"Test Loss  : {test_loss_metric.result():.4f}, Test Accuracy  : {test_accuracy_metric.result():.4f}")
    
    model.save("models/MetaSynapse_v5.keras")
    meta_controller.save("models/MetaSynapse_v5_controller.keras")

# =============================================================================
# 7. Main: Create dataset, instantiate models, and train.
# =============================================================================
def main():
    batch_size = 128
    num_hidden = 128
    hidden_units = 32
    # Load sequence data
    sequence = get_real_data(num_samples=100_000)
    inputs, labels = create_windows(sequence, window_size=65)
    (inp_train, lbl_train), (inp_val, lbl_val), (inp_test, lbl_test) = split_dataset(inputs, labels)
    
    ds_train = create_tf_dataset(inp_train, lbl_train, batch_size=batch_size, shuffle=False)
    ds_val = create_tf_dataset(inp_val, lbl_val, batch_size=batch_size, shuffle=False)
    ds_test = create_tf_dataset(inp_test, lbl_test, batch_size=batch_size, shuffle=False)
    
    model = PlasticityModel(num_hidden=num_hidden, num_classes=10)
    meta_controller = RecurrentMetaController(hidden_units=hidden_units)
    
    train_model(model, meta_controller, ds_train, ds_val, ds_test, num_epochs=1000, homeostasis_interval=100)

if __name__ == '__main__':
    main()
