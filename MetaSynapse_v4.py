import tensorflow as tf
import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# =============================================================================
# 1. Define LearnedActivation: a custom activation that mixes several activations.
# =============================================================================

class LearnedActivation(tf.keras.layers.Layer):
    def __init__(self, initial_r=3.8, **kwargs):
        super().__init__(**kwargs)
        self.initial_r = initial_r
            
    def build(self, input_shape):
        # 6 learnable weights for mixing six activation functions.
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
        # Softmax-normalize the mixing weights.
        weights = tf.nn.softmax(self.w)
        # Weighted sum of activations.
        results = (weights[0]*sig + weights[1]*elu + weights[2]*tanh +
                   weights[3]*relu + weights[4]*silu + weights[5]*gelu)
        return results


# =============================================================================
# 2. Define PlasticDenseAdvanced layer with integrated plasticity mechanisms.
# We'll now include trainable variables for A_plus and A_minus.
# =============================================================================

class PlasticDenseAdvanced(tf.keras.layers.Layer):
    def __init__(self, units, prune_threshold=0.01, add_prob=0.001, **kwargs):
        """
        units: number of neurons in the layer.
        prune_threshold: threshold below which synapses are pruned.
        add_prob: probability per weight element to add a new connection if weight==0.
        """
        super(PlasticDenseAdvanced, self).__init__(**kwargs)
        self.units = units
        # Use our custom LearnedActivation.
        self.activation = LearnedActivation()
        self.prune_threshold = prune_threshold
        self.add_prob = add_prob
        # Meta-plasticity: a multiplicative factor for plasticity updates.
        self.meta_lr = tf.Variable(0.01, trainable=False, dtype=tf.float32)
        # Trainable STDP amplitude parameters.
        self.A_plus = tf.Variable(0.01, trainable=True, dtype=tf.float32)
        self.A_minus = tf.Variable(0.01, trainable=True, dtype=tf.float32)

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(int(input_shape[-1]), self.units),
            initializer='glorot_uniform',
            trainable=True,
            name='kernel')
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='bias')
        super(PlasticDenseAdvanced, self).build(input_shape)

    def call(self, inputs):
        z = tf.matmul(inputs, self.w) + self.b
        return self.activation(z)

    def plasticity_update(self, pre_activity, post_activity, reward):
        """
        Compute an STDP-like update.
        - pre_activity: average input vector (shape: [input_dim])
        - post_activity: average output vector (shape: [units])
        - reward: scalar neuromodulatory signal in [0,1]
        """
        # Simulate a random spike timing difference delta_t (in ms) per weight element.
        delta_t = tf.random.uniform(tf.shape(self.w), minval=-100.0, maxval=100.0)
        tau_plus = 100.0
        tau_minus = 100.0

        # Use the trainable A_plus and A_minus variables.
        stdp_update = tf.where(delta_t > 0,
                               self.A_plus * tf.exp(-delta_t / tau_plus),
                               -self.A_minus * tf.exp(delta_t / tau_minus))
        # Compute Hebbian correlation (outer product).
        pre = tf.reshape(pre_activity, [-1, 1])
        post = tf.reshape(post_activity, [1, -1])
        hebbian_term = pre * post
        plasticity_delta = stdp_update * hebbian_term
        # Gate update by neuromodulatory signal (reward).
        plasticity_delta = tf.cast(reward, tf.float32) * plasticity_delta
        return plasticity_delta

    def apply_plasticity(self, plasticity_delta):
        self.w.assign_add(self.meta_lr * plasticity_delta)

    def apply_homeostatic_scaling(self):
        avg_w = tf.reduce_mean(tf.abs(self.w))
        target_avg = 0.05  # Target average weight.
        scaling_factor = target_avg / (avg_w + 1e-6)
        self.w.assign(self.w * scaling_factor)

    def apply_structural_plasticity(self):
        # Prune weights below the threshold.
        pruned_w = tf.where(tf.abs(self.w) < self.prune_threshold, tf.zeros_like(self.w), self.w)
        self.w.assign(pruned_w)
        # Randomly add new connections where weight == 0.
        random_matrix = tf.random.uniform(tf.shape(self.w))
        add_mask = tf.cast(random_matrix < self.add_prob, tf.float32)
        new_connections = tf.where(tf.logical_and(tf.equal(self.w, 0.0), tf.equal(add_mask, 1.0)),
                                    tf.random.uniform(tf.shape(self.w), minval=0.01, maxval=0.05),
                                    self.w)
        self.w.assign(new_connections)

    def apply_meta_plasticity(self, plasticity_delta):
        avg_update = tf.reduce_mean(tf.abs(plasticity_delta))
        new_meta_lr = tf.maximum(self.meta_lr * tf.exp(-0.01 * avg_update), 1e-5)
        self.meta_lr.assign(new_meta_lr)

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)
        
        
# ---------- PlasticDense Layer with Hebbian Updates ----------
class PlasticDenseBasic(tf.keras.layers.Layer):
    def __init__(self, units, hebbian_rate=1e-5, use_layer_norm=True, **kwargs):
        super(PlasticDenseBasic, self).__init__(**kwargs)
        self.units = units
        self.hebbian_rate = hebbian_rate
        self.use_bias = True
        self.use_layer_norm = use_layer_norm

    def build(self, input_shape):
        last_dim = int(input_shape[-1])
        self.activation_fn = LearnedActivation()
        self.layer_norm = tf.keras.layers.LayerNormalization() if self.use_layer_norm else None
        self.kernel = self.add_weight(
            shape=(last_dim, self.units),
            initializer='glorot_uniform',
            trainable=True,
            name='kernel',
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer='zeros',
                trainable=True,
                name='bias'
            )
        else:
            self.bias = None
        super(PlasticDenseBasic, self).build(input_shape)

    def call(self, inputs, modulation=None, training=None):
        output = tf.matmul(inputs, self.kernel)
        if self.use_bias:
            output = output + self.bias
        if self.activation_fn is not None:
            output = self.activation_fn(output)
        if self.layer_norm is not None:
            output = self.layer_norm(output)            
        if training:
            adapted_hebbian = self.adaptive_hebbian_rate(self.kernel, self.hebbian_rate)
            mod_hebbian = self.hebbian_rate * modulation if modulation is not None else adapted_hebbian
            delta = self.compute_hebbian_delta(inputs, output)
            self.kernel.assign_add(mod_hebbian * delta)
        return output
    
    def compute_hebbian_delta(self, inputs, output):
        if len(inputs.shape) == 3:
            inp_reduced = tf.reduce_mean(inputs, axis=1)
            out_reduced = tf.reduce_mean(output, axis=1)
        else:
            inp_reduced = inputs
            out_reduced = output
        batch_size = tf.cast(tf.shape(inp_reduced)[0], tf.float32)
        delta = tf.einsum('bi,bo->io', inp_reduced, out_reduced) / batch_size
        return delta
        
    def adaptive_hebbian_rate(self, kernel, hebbian_rate):
        norm = tf.norm(kernel) + 1e-8  # Prevent division by zero
        return hebbian_rate / norm  # Scale based on weight magnitude        
        
    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)        

        
# =============================================================================
# 3. Define a MetaController module that adjusts plasticity parameters.
# =============================================================================

class MetaController(tf.keras.Model):
    def __init__(self, hidden_units=16, **kwargs):
        """
        A simple MLP that takes a feature vector and outputs scaling multipliers for:
          - meta_lr, A_plus, and A_minus.
        """
        super(MetaController, self).__init__(**kwargs)
        self.dense1 = PlasticDenseBasic(hidden_units)
        # Output layer: 3 values. Use softplus to ensure multipliers are positive.
        self.out_layer = tf.keras.layers.Dense(3, activation='softplus')
    
    def call(self, features):
        """
        features: a tensor of shape (batch_size, feature_dim) or (feature_dim,)
        Returns a tensor of shape (3,) with scaling multipliers.
        """
        x = self.dense1(tf.expand_dims(features, axis=0)) if len(features.shape)==1 else self.dense1(features)
        multipliers = self.out_layer(x)
        return tf.squeeze(multipliers, axis=0)


# =============================================================================
# 4. Define the end-to-end PlasticityModel for classification.
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
# 5. Data loading: Build a sliding-window dataset from a single CSV.
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
    """
    Given a 1D numpy array 'sequence', create overlapping windows of length 'window_size'.
    The first window_size-1 values are input and the last value is the label.
    """
    num_windows = len(sequence) - window_size + 1
    windows = np.lib.stride_tricks.as_strided(
        sequence,
        shape=(num_windows, window_size),
        strides=(sequence.strides[0], sequence.strides[0])
    ).copy()
    inputs = windows[:, :-1]  # first window_size-1 values
    labels = windows[:, -1]   # last value
    return inputs, labels

def split_dataset(inputs, labels, train_frac=0.8, val_frac=0.1):
    total = inputs.shape[0]
    train_end = int(total * train_frac)
    val_end = int(total * (train_frac + val_frac))
    return (inputs[:train_end], labels[:train_end]), (inputs[train_end:val_end], labels[train_end:val_end]), (inputs[val_end:], labels[val_end:])

def create_tf_dataset(inputs, labels, batch_size=128, shuffle=False, repeat_epochs=1):
    # Normalize inputs to [0, 1] by dividing by 9.
    inputs = inputs.astype(np.float32) / 9.0
    # Reshape inputs: here, we assume window_size-1 = 32, reshaped to (4, 8, 1).
    inputs = inputs.reshape((-1, 4, 8, 1))
    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.repeat(repeat_epochs).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


# =============================================================================
# 6. Training loop with meta-controller integration.
# =============================================================================

def train_model(model, meta_controller, ds_train, ds_val, ds_test, num_epochs=5, homeostasis_interval=100):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adadelta(learning_rate=1.0)
    
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    
    val_loss_metric = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
    
    global_step = 0

    # We'll collect features for meta-controller at the end of each epoch.
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}")
        for images, labels in ds_train:
            with tf.GradientTape() as tape:
                predictions, hidden = model(images, training=True)
                loss = loss_fn(labels, predictions)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
            train_loss(loss)
            train_accuracy(labels, predictions)
            
            # Plasticity updates:
            x_flat = model.flatten(images)
            pre_activity = tf.reduce_mean(x_flat, axis=0)
            post_activity = tf.reduce_mean(hidden, axis=0)
            reward = tf.sigmoid(-loss)
            plasticity_delta = model.hidden.plasticity_update(pre_activity, post_activity, reward)
            model.hidden.apply_plasticity(plasticity_delta)
            model.hidden.apply_meta_plasticity(plasticity_delta)
            
            if global_step % homeostasis_interval == 0:
                model.hidden.apply_homeostatic_scaling()
                model.hidden.apply_structural_plasticity()
            
            global_step += 1
        
        print(f"Train Loss: {train_loss.result():.4f}, Train Accuracy: {train_accuracy.result():.4f}")
        # Reset training metrics.
        current_train_loss = train_loss.result()
        train_loss.reset_state()
        train_accuracy.reset_state()
        
        # Collect features for meta-controller.
        # For instance: [current_train_loss, average absolute weight, current meta_lr]
        avg_weight = tf.reduce_mean(tf.abs(model.hidden.w))
        current_meta_lr = model.hidden.meta_lr
        features = tf.stack([current_train_loss, avg_weight, current_meta_lr])
        
        # Get scaling multipliers from meta-controller.
        scaling_multipliers = meta_controller(features)  # shape (3,)
        # Update the plasticity parameters in the hidden layer:
        # Multiply meta_lr, A_plus, and A_minus by the respective scaling multipliers.
        model.hidden.meta_lr.assign(tf.maximum(model.hidden.meta_lr * scaling_multipliers[0], 1e-5))
        model.hidden.A_plus.assign(tf.maximum(model.hidden.A_plus * scaling_multipliers[1], 1e-5))
        model.hidden.A_minus.assign(tf.maximum(model.hidden.A_minus * scaling_multipliers[2], 1e-5))
        print(f"Meta updates: meta_lr -> {model.hidden.meta_lr.numpy():.5f}, A_plus -> {model.hidden.A_plus.numpy():.5f}, A_minus -> {model.hidden.A_minus.numpy():.5f}")
        
        # Validation step.
        for images, labels in ds_val:
            predictions, _ = model(images, training=False)
            loss_val = loss_fn(labels, predictions)
            val_loss_metric(loss_val)
            val_accuracy_metric(labels, predictions)
        print(f"Validation Loss: {val_loss_metric.result():.4f}, Validation Accuracy: {val_accuracy_metric.result():.4f}")
        val_loss_metric.reset_state()
        val_accuracy_metric.reset_state()
        
        # Test evaluation at end of epoch.
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        for images, labels in ds_test:
            predictions, _ = model(images, training=False)
            loss_test = loss_fn(labels, predictions)
            test_loss(loss_test)
            test_accuracy(labels, predictions)
        print(f"Test Loss: {test_loss.result():.4f}, Test Accuracy: {test_accuracy.result():.4f}")
    
    model.save("models/meta_synapse_model.keras")
    meta_controller.save("models/meta_controller.keras")

    
# =============================================================================
# 7. Main: Load data from CSV, create dataset splits, instantiate model and meta-controller, and train.
# =============================================================================

def main():
    batch_size = 128
    num_hidden = 256
    hidden_units = 64
    # Load entire sequence from CSV
    sequence = get_real_data(num_samples=100_000)
    # Create sliding windows; window_size=33 means 32 inputs and 1 label.
    inputs, labels = create_windows(sequence, window_size=33)
    # Split the windows: 80% train, 10% validation, 10% test.
    (inp_train, lbl_train), (inp_val, lbl_val), (inp_test, lbl_test) = split_dataset(inputs, labels, train_frac=0.8, val_frac=0.1)
    
    # Create tf.data.Datasets for each split.
    ds_train = create_tf_dataset(inp_train, lbl_train, batch_size=batch_size, shuffle=False, repeat_epochs=1)
    ds_val   = create_tf_dataset(inp_val, lbl_val, batch_size=batch_size, shuffle=False, repeat_epochs=1)
    ds_test  = create_tf_dataset(inp_test, lbl_test, batch_size=batch_size, shuffle=False, repeat_epochs=1)
    
    # Instantiate the main model and the meta-controller.
    model = PlasticityModel(num_hidden=num_hidden, num_classes=10)
    meta_controller = MetaController(hidden_units=hidden_units)
    
    # Train the model with meta-controller adjustments.
    train_model(model, meta_controller, ds_train, ds_val, ds_test, num_epochs=1000, homeostasis_interval=100)

if __name__ == '__main__':
    main()
