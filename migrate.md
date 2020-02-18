# Migrate from TF-1 to TF-2

First of all, it is still possible to run tf-1 code by the following method:

```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
```



## Automatic conversion script

skipped

## Top-level behavioral changes

Four API's still need to fix even used `tf.compat.v1.disable_v2_behavior`

- *Eager execution, [`v1.enable_eager_execution()`](https://www.tensorflow.org/api_docs/python/tf/compat/v1/enable_eager_execution)* 
- *Resource variables, [`v1.enable_resource_variables()`](https://www.tensorflow.org/api_docs/python/tf/compat/v1/enable_resource_variables)*
- *Tensor shapes, [`v1.enable_v2_tensorshape()`](https://www.tensorflow.org/api_docs/python/tf/compat/v1/enable_v2_tensorshape)*
- *Control flow, [`v1.enable_control_flow_v2()`](https://www.tensorflow.org/api_docs/python/tf/compat/v1/enable_control_flow_v2)*

## Make the code 2.0-native

### Replace `v1.Session.run` calls

1. Every [`v1.Session.run`](https://www.tensorflow.org/api_docs/python/tf/compat/v1/Session#run) call should be replaced by a Python function.

   1. The `feed_dict` and [`v1.placeholder`](https://www.tensorflow.org/api_docs/python/tf/compat/v1/placeholder)s become function arguments.
   2. The `fetches` become the function's return value.
   3. During conversion eager execution allows __easy debugging__ with standard Python tools like `pdb`.

2. After that add a [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) decorator to make it run efficiently in graph. Note that:

   - Unlike [`v1.Session.run`](https://www.tensorflow.org/api_docs/python/tf/compat/v1/Session#run) a [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) has a __fixed return signature, and always returns all outputs__. If this causes performance problems, create two separate functions.
   - There is no need for a [`tf.control_dependencies`](https://www.tensorflow.org/api_docs/python/tf/control_dependencies) or similar operations: A [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) behaves as if it were run in the order written. [`tf.Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable) assignments and `tf.assert`s, for example, are executed automatically.

3. Use Python **objects** to **track variables and losses**

   1. All name-based variable tracking is strongly discouraged in TF 2.0. Use Python objects to to track variables.
   2. Use [`tf.Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable) instead of [`v1.get_variable`](https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable).
   3. Every [`v1.variable_scope`](https://www.tensorflow.org/api_docs/python/tf/compat/v1/variable_scope) should be converted to a Python object. Typically this will be one of:
      1. [`tf.keras.layers.Layer`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer)
      2. [`tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model)
      3. [`tf.Module`](https://www.tensorflow.org/api_docs/python/tf/Module)
   4. If you need to aggregate lists of variables (like [`tf.Graph.get_collection(tf.GraphKeys.VARIABLES)`](https://www.tensorflow.org/api_docs/python/tf/Graph#get_collection)), use the `.variables` and `.trainable_variables` attributes of the `Layer` and `Model` objects. These `Layer` and `Model` classes implement several other properties that remove the need for global collections. Their `.losses` property can be a replacement for using the `tf.GraphKeys.LOSSES` collection.

4. Upgrade your training loops

   1. **Use the highest level API that works for your use case.** **Prefer [`tf.keras.Model.fit`](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit) over building your own training loops.**
   2. These high level functions manage a lot of the low-level details that might be easy to miss if you write your own training loop. For example, they automatically collect the regularization losses, and set the `training=True` argument when calling the model.

5. Upgrade your data input pipelines

   1. Use `tf.data` datasets for data input. Two ways of using it.

      ```python
      # method 1
      model.fit(dataset, epochs=5)
      
      # method 2
      for example_batch, label_batch in dataset:
        break
      ```

6. Migrate off `compat.v1` symbols. Skip.

## Converting models

### Low-level variables & operator execution

Below are some tf-1 styled usage of low-level api-calls.

- using variable scopes to control reuse
- creating variables with [`v1.get_variable`](https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable).
- accessing collections explicitly
- accessing collections implicitly with methods like :
  - [`v1.global_variables`](https://www.tensorflow.org/api_docs/python/tf/compat/v1/global_variables)
  - [`v1.losses.get_regularization_loss`](https://www.tensorflow.org/api_docs/python/tf/compat/v1/losses/get_regularization_loss)
- using [`v1.placeholder`](https://www.tensorflow.org/api_docs/python/tf/compat/v1/placeholder) to set up graph inputs
- executing graphs with `Session.run`
- initializing variables manually

In converted code:

- The variables are local Python objects.
- The `forward` function still defines the calculation.
- The `Session.run` call is replaced with a call to `forward`
- The optional [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) decorator can be added for performance.
- The regularizations are calculated manually, without referring to any global collection.
- **No sessions or placeholders.**

Compare:

```python
# Before
in_a = tf.placeholder(dtype=tf.float32, shape=(2))
in_b = tf.placeholder(dtype=tf.float32, shape=(2))

def forward(x):
  with tf.variable_scope("matmul", reuse=tf.AUTO_REUSE):
    W = tf.get_variable("W", initializer=tf.ones(shape=(2,2)),
                        regularizer=tf.contrib.layers.l2_regularizer(0.04))
    b = tf.get_variable("b", initializer=tf.zeros(shape=(2)))
    return W * x + b

out_a = forward(in_a)
out_b = forward(in_b)

reg_loss=tf.losses.get_regularization_loss(scope="matmul")

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  outs = sess.run([out_a, out_b, reg_loss],
                feed_dict={in_a: [1, 0], in_b: [0, 1]})
  
# After
W = tf.Variable(tf.ones(shape=(2,2)), name="W")
b = tf.Variable(tf.zeros(shape=(2)), name="b")

@tf.function
def forward(x):
  return W * x + b

out_a = forward([1,0])
print(out_a)

out_b = forward([0,1])

regularizer = tf.keras.regularizers.l2(0.04)
reg_loss=regularizer(W)
```



### Models based on `tf.layers`

tf-1 replies on `tf.variable_scope` to define and reuse variables.

**After converting**

- The simple stack of layers fits neatly into [`tf.keras.Sequential`](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential). (For more complex models see [custom layers and models](https://www.tensorflow.org/guide/keras/custom_layers_and_models), and [the functional API](https://www.tensorflow.org/guide/keras/functional).)
- The model tracks the variables, and regularization losses.
- The conversion was one-to-one because there is a direct mapping from [`v1.layers`](https://www.tensorflow.org/api_docs/python/tf/compat/v1/layers) to [`tf.keras.layers`](https://www.tensorflow.org/api_docs/python/tf/keras/layers).

```python
# Before
def model(x, training, scope='model'):
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    x = tf.layers.conv2d(x, 32, 3, activation=tf.nn.relu,
          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.04))
    x = tf.layers.max_pooling2d(x, (2, 2), 1)
    x = tf.layers.flatten(x)
    x = tf.layers.dropout(x, 0.1, training=training)
    x = tf.layers.dense(x, 64, activation=tf.nn.relu)
    x = tf.layers.batch_normalization(x, training=training)
    x = tf.layers.dense(x, 10)
    return x

train_out = model(train_data, training=True)
test_out = model(test_data, training=False)

# After
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(0.04),
                           input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(10)
])

train_data = tf.ones(shape=(1, 28, 28, 1))
test_data = tf.ones(shape=(1, 28, 28, 1))

train_out = model(train_data, training=True)
print(train_out)

test_out = model(test_data, training=False)
print(test_out)

len(model.trainable_variables)  # Here are all the trainable variables.

model.losses # Here is the regularization loss.
```

Most arguments stayed the same. But notice the differences:

- The `training` argument is passed to each layer **by the model** when it runs. 
- The first argument to the original `model` function (the input `x`) is gone. This is because object layers separate building the model from calling the model.



### Mixed variables & v1.layers

1. A [`v1.variable_scope`](https://www.tensorflow.org/api_docs/python/tf/compat/v1/variable_scope) is effectively a layer of its own. So rewrite it as a [`tf.keras.layers.Layer`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer). See [the guide](https://www.tensorflow.org/guide/keras/custom_layers_and_models) for details. The general pattern is:
   - Collect layer parameters in `__init__`.
   - Build the variables in `build`.
   - Execute the calculations in `call`, and return the result.

```python
# Before
def model(x, training, scope='model'):
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    W = tf.get_variable(
      "W", dtype=tf.float32,
      initializer=tf.ones(shape=x.shape),
      regularizer=tf.contrib.layers.l2_regularizer(0.04),
      trainable=True)
    if training:
      x = x + W
    else:
      x = x + W * 0.5
    x = tf.layers.conv2d(x, 32, 3, activation=tf.nn.relu)
    x = tf.layers.max_pooling2d(x, (2, 2), 1)
    x = tf.layers.flatten(x)
    return x

train_out = model(train_data, training=True)
test_out = model(test_data, training=False)

# After conversion

# Create a custom layer for part of the model
class CustomLayer(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(CustomLayer, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self.w = self.add_weight(
        shape=input_shape[1:],
        dtype=tf.float32,
        initializer=tf.keras.initializers.ones(),
        regularizer=tf.keras.regularizers.l2(0.02),
        trainable=True)

  # Call method will sometimes get used in graph mode,
  # training will get turned into a tensor
  @tf.function
  def call(self, inputs, training=None):
    if training:
      return inputs + self.w
    else:
      return inputs + self.w * 0.5

# when running in new
custom_layer = CustomLayer()
print(custom_layer([1]).numpy())
print(custom_layer([1], training=True).numpy())

# Use with keras Sequential

train_data = tf.ones(shape=(1, 28, 28, 1))
test_data = tf.ones(shape=(1, 28, 28, 1))

# Build the model including the custom layer
model = tf.keras.Sequential([
    CustomLayer(input_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
])

train_out = model(train_data, training=True)
test_out = model(test_data, training=False)

```

Some things to note:

- Subclassed Keras models & layers need to run in both v1 graphs (no automatic control dependencies) and in eager mode
  - **Wrap the `call()` in a [`tf.function()`](https://www.tensorflow.org/api_docs/python/tf/function) to get autograph and automatic control dependencies**
- **Don't forget to accept a `training` argument to `call`.**
  - Sometimes it is a [`tf.Tensor`](https://www.tensorflow.org/api_docs/python/tf/Tensor)
  - Sometimes it is a **Python boolean.**
- **Create model variables in constructor or [`Model.build`](https://www.tensorflow.org/api_docs/python/tf/keras/Model#build) using `self.add_weight()`.**
  - In [`Model.build`](https://www.tensorflow.org/api_docs/python/tf/keras/Model#build) you have access to the input shape, so can create weights with matching shape.
  - Using [`tf.keras.layers.Layer.add_weight`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#add_weight) allows Keras to track variables and regularization losses.
- Don't keep `tf.Tensors` in your objects.
  - They might get created either in a [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) or in the eager context, and these tensors behave differently.
  - Use [`tf.Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable)s for state, they are always usable from both contexts
  - `tf.Tensors` are **only for intermediate values.**



### A note on Slim & contrib.layers

Skipped



## Training

There are many ways to feed data to a [`tf.keras`](https://www.tensorflow.org/api_docs/python/tf/keras) model. They will accept **Python generators and Numpy arrays as input.**

The recommended way to feed data to a model is to use the [`tf.data`](https://www.tensorflow.org/api_docs/python/tf/data) package, which contains a collection of high performance classes for manipulating data.

### Using dataset

See more from `tfds`

### Using Keras training loops

If you don't need low level control of your training process, using Keras's built-in `fit`, `evaluate`, and `predict` methods is recommended. These methods provide a uniform interface to train the model regardless of the implementation (sequential, functional, or sub-classed).

The advantages of these methods include:

- They accept Numpy arrays, Python generators and, `tf.data.Datasets`
- They apply regularization, and activation losses automatically.
- They support [`tf.distribute`](https://www.tensorflow.org/api_docs/python/tf/distribute) [for multi-device training](https://www.tensorflow.org/guide/distributed_training).
- They support arbitrary callables as losses and metrics.
- They support callbacks like [`tf.keras.callbacks.TensorBoard`](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard), and custom callbacks.
- They are performant, automatically using TensorFlow graphs.



### Write your own loop

1. If the Keras model's training step works for you, but you need more control outside that step, consider using the [`tf.keras.Model.train_on_batch`](https://www.tensorflow.org/api_docs/python/tf/keras/Model#train_on_batch) method, in your own data-iteration loop.

2. Remember: Many things can be implemented as a [`tf.keras.callbacks.Callback`](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback).

3. This method **has many of the advantages of the methods mentioned in the previous section**, but gives the **user control of the outer loop**.

4. You can also use [`tf.keras.Model.test_on_batch`](https://www.tensorflow.org/api_docs/python/tf/keras/Model#test_on_batch) or [`tf.keras.Model.evaluate`](https://www.tensorflow.org/api_docs/python/tf/keras/Model#evaluate) to check performance during training.

```python
# Model is the full model w/o custom layers
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

metrics_names = model.metrics_names

for epoch in range(NUM_EPOCHS):
  #Reset the metric accumulators
  model.reset_metrics()

  for image_batch, label_batch in train_data:
    result = model.train_on_batch(image_batch, label_batch)  # <-- train_on_batch
    print("train: ",
          "{}: {:.3f}".format(metrics_names[0], result[0]),
          "{}: {:.3f}".format(metrics_names[1], result[1]))
  for image_batch, label_batch in test_data:
    result = model.test_on_batch(image_batch, label_batch,
                                 # return accumulated metrics
                                 reset_metrics=False)
  print("\neval: ",
        "{}: {:.3f}".format(metrics_names[0], result[0]),
        "{}: {:.3f}".format(metrics_names[1], result[1]))
```



### Customize the training step

**More flexibility and control**: implementing your own training loop. There are three steps:

1. Iterate over a **Python generator** or [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) to get batches of examples.
2. Use [`tf.GradientTape`](https://www.tensorflow.org/api_docs/python/tf/GradientTape) to collect **gradients**.
3. Use one of the [`tf.keras.optimizers`](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers) to apply weight updates to the model's variables.

Remember:

- Always include a `training` argument on the `call` method of subclassed layers and models.
- Make sure to call the model with the `training` argument set correctly.
- Depending on usage, model variables may not exist until the model is run on a batch of data.
- You need to manually handle things like regularization losses for the model.

Note the simplifications relative to v1:

- There is **no need to run variable initializers**. Variables are initialized on creation.
- There is **no need to add manual control dependencies**. Even in [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) operations act as in eager mode.

```python
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(0.02),
                           input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(10)
])

optimizer = tf.keras.optimizers.Adam(0.001)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

@tf.function
def train_step(inputs, labels):
  with tf.GradientTape() as tape:
    predictions = model(inputs, training=True)
    regularization_loss=tf.math.add_n(model.losses)
    pred_loss=loss_fn(labels, predictions)
    total_loss=pred_loss + regularization_loss

  gradients = tape.gradient(total_loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

for epoch in range(NUM_EPOCHS):
  for inputs, labels in train_data:
    train_step(inputs, labels)
  print("Finished epoch", epoch)
```



### New-style metrics and losses

1. In TensorFlow 2.0, **metrics and losses are objects**. These work both eagerly and in [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function)s. A loss object is callable, and expects the **(y_true, y_pred)** as arguments:

2. A metric object has the following methods:

- [`Metric.update_state()`](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Metric#update_state) — add new observations
- [`Metric.result()`](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Metric#result) —get the current result of the metric, given the observed values
- [`Metric.reset_states()`](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Metric#reset_states) — clear all observations.

3. The object itself is callable. Calling updates the state with new observations, as with `update_state`, and returns the new result of the metric. You don't have to manually initialize a metric's variables, and because TensorFlow 2.0 has automatic control dependencies, you don't need to worry about those either.

```python
# Create the metrics
loss_metric = tf.keras.metrics.Mean(name='train_loss')
accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

@tf.function
def train_step(inputs, labels):
  with tf.GradientTape() as tape:
    predictions = model(inputs, training=True)
    regularization_loss=tf.math.add_n(model.losses)
    pred_loss=loss_fn(labels, predictions)
    total_loss=pred_loss + regularization_loss

  gradients = tape.gradient(total_loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  # Update the metrics
  loss_metric.update_state(total_loss)  # <-- update state here
  accuracy_metric.update_state(labels, predictions)  # <-- update accuracy here


for epoch in range(NUM_EPOCHS):
  # Reset the metrics
  loss_metric.reset_states()  # <-- reset the metrics before iterations
  accuracy_metric.reset_states()  # <-- reset the metrics before iterations

  for inputs, labels in train_data:
    train_step(inputs, labels)
  # Get the metric results
  mean_loss=loss_metric.result()
  mean_accuracy = accuracy_metric.result()

  print('Epoch: ', epoch)
  print('  loss:     {:.3f}'.format(mean_loss))
  print('  accuracy: {:.3f}'.format(mean_accuracy))

```



### Keras optimizers

Lookup the doc for different optimizers

### TensorBoard

TF2 includes major changes to `tf.summary`. Checkout [this link](https://www.tensorflow.org/tensorboard/get_started) for tutorials.



## Saving&Loading

Only introduced the conversion. But for how to save and load models in TF2. Check out [this link](https://www.tensorflow.org/guide/checkpoint)

## Estimators

Skipped for never used an Estimator.

## TensorShape

1. This class was simplified to hold `int`s, instead of [`tf.compat.v1.Dimension`](https://www.tensorflow.org/api_docs/python/tf/compat/v1/Dimension) objects. So there is no need to call `.value()` to get an `int`.

```python
# For TF 1.x
for dim in shape:
  value = dim.value
  print(value)

# For TF 2.x
for dim in shape:
  print(value)
```

2. The boolean value of a [`tf.TensorShape`](https://www.tensorflow.org/api_docs/python/tf/TensorShape) is `True` if the rank is known, `False` otherwise.

## Other Changes

- Remove `tf.colocate_with`: TensorFlow's device placement algorithms have improved significantly. This should no longer be necessary. If removing it causes a performance degredation [please file a bug](https://github.com/tensorflow/tensorflow/issues).
- Replace [`v1.ConfigProto`](https://www.tensorflow.org/api_docs/python/tf/compat/v1/ConfigProto) usage with the equivalent functions from [`tf.config`](https://www.tensorflow.org/api_docs/python/tf/config).



## Conclusions

The overall process is:

1. Run the upgrade script.
2. Remove contrib symbols.
3. Switch your models to an object oriented style (Keras).
4. Use [`tf.keras`](https://www.tensorflow.org/api_docs/python/tf/keras) or [`tf.estimator`](https://www.tensorflow.org/api_docs/python/tf/estimator) training and evaluation loops where you can.
5. Otherwise, use custom loops, but be sure to avoid sessions & collections.

It takes a little work to convert code to idiomatic TensorFlow 2.0, but every change results in:

- Fewer lines of code.
- Increased clarity and simplicity.
- Easier debugging.