# Learning TF-2

##Effective TensorFlow 2

###Summary of changes

####API Cleanup

1. remove `tf.app`, `tf.flags`, and `tf.logging`, changed to `absl-py`.
2. clean up `tf.*` namespace, some moved to subpackages such as `tf.math`.
3. replaced API's: `tf.summary`, `tf.keras.metrics`, and `tf.keras.optimizers`.

####Eager execution

1. `tf.control_dependencies()` is no longer required.

####No more globals

1. No more global variables and the helper functions to help find the defined variables such as:  Variable scopes, global collections, helper methods like `tf.get_global_step()`, `tf.global_variables_initializer()`, optimizers implicitly computing gradients over all trainable variables, and so on.
2. Keep track of your variables! If you lose track of a `tf.Variable()`, it gets garbage collected.

####Functions, not sessions

1. In TF-2, you can decorate a Python function using `tf.function()` to mark it JIT compilation so that TensorFlow runs it as a single graph.
2. AutoGraph converts a subset of Python constructs into their TensorFlow equivalents.
   1. `for/while` -> `tf.while_loop` (break and continue are supported)
   2. `if` -> `tf.cond`
   3. `for _ in dataset` -> `dataset.reduce`.

###Recommendations for idiomatic TF-2

####Refactor code into smaller functions

1. In TF-2 users should refactor their code into smaller functions that are called as needed.
2. In general, it's not necessary to decorate each of these smaller functions with [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function); only use [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) to decorate high-level computations - for example, one step of training or the forward pass of your model.

####Use Keras layers and models to manage variables

1. Keras models and layers offer the convenient `variables` and `trainable_variables` properties, which recursively gather up all dependent variables.
2. Keras layes/models inherit from `tf.train.Checkpointable` and are integrated with `@tf.funciotn`, which makes it possible to directly checkpointor export SavedModels from Keras objects.

####Combine `tf.data.Datasets` and `@tf.function`.

1. When iterating over training data that fits in memory, feel free to use regular Python iteration. 
2. Otherwise, [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) is the best way to stream training data from disk. Datasets are [iterables (not iterators)](https://docs.python.org/3/glossary.html#term-iterable), and work just like other Python iterables in Eager mode. You can fully utilize dataset async __prefetching/streaming__ features by wrapping your code in [`tf.function()`](https://www.tensorflow.org/api_docs/python/tf/function), which replaces Python iteration with the equivalent graph operations using AutoGraph.

####Take advantage of AutoGraph with Python control flow

[fill in this part later after AutoGraph part]

####tf.metrics aggregates data and tf.summary logs them

1. To log summaries, use `tf.summary.(scalar|histogram|...)` and redirect it to a writer using a context manager. (If you omit the context manager, nothing happens.) 

2. Unlike TF 1.x, the summaries are emitted directly to the writer; there is no separate "merge" op and no separate `add_summary()` call, which means that the `step` value must be provided at the callsite.

3. Example

   ```python
   summary_writer = tf.summary.create_file_writer('/tmp/summaries')
   with summary_writer.as_default():
     tf.summary.scalar('loss', 0.1, step=42)
   ```

4. To aggregate data before logging them as summaries, use [`tf.metrics`](https://www.tensorflow.org/api_docs/python/tf/keras/metrics). Metrics are stateful: They accumulate values and return a cumulative result when you call `.result()`. Clear accumulated values with `.reset_states()`.

   ```python
   def train(model, optimizer, dataset, log_freq=10):
     avg_loss = tf.keras.metrics.Mean(name='loss', dtype=tf.float32)
     for images, labels in dataset:
       loss = train_step(model, optimizer, images, labels)
       avg_loss.update_state(loss)
       if tf.equal(optimizer.iterations % log_freq, 0):
         tf.summary.scalar('loss', avg_loss.result(), step=optimizer.iterations)
         # seems like the num of iterations is available from optimizer object
         avg_loss.reset_states()
   ```

####Use `tf.config.experimental_run_functions_eagerly()` when debugging

1. In TensorFlow 2.0, Eager execution lets you run the code step-by-step to inspect shapes, data types and values. 
2. Certain APIs, like [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function), [`tf.keras`](https://www.tensorflow.org/api_docs/python/tf/keras), etc. are designed to use Graph execution, for performance and portability. When debugging, use [`tf.config.experimental_run_functions_eagerly(True)`](https://www.tensorflow.org/api_docs/python/tf/config/experimental_run_functions_eagerly) to use Eager execution inside this code.

