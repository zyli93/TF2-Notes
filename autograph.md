# `tf.function` and AutoGraph



## The `tf.function` decorator

1. When you annotate a function with [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function), you can still call it like any other function. But it will be compiled into a graph, which means you get the benefits of faster execution, running on GPU or TPU, or exporting to SavedModel.
2. If your code uses multiple functions, you don't need to annotate them all - any functions called from an annotated function will also run in graph mode.
3. **Functions can be faster than eager code, for graphs with many small ops.** But for graphs with a few expensive ops (like convolutions), you may not see much speedup.



## Use Python control flow

1. When using data-dependent control flow inside [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function), you can use Python control flow statements and AutoGraph will convert them into appropriate TensorFlow ops. For example, `if` statements will be converted into [`tf.cond()`](https://www.tensorflow.org/api_docs/python/tf/cond) if they depend on a `Tensor`. 

2. AutoGraph supports common Python statements like `while`, `for`, `if`, `break`, `continue` and `return`, with support for nesting. That means you can use `Tensor` expressions in the condition of `while` and `if` statements, or iterate over a `Tensor` in a `for` loop.

3. AutoGraph also provides a low-level API for advanced users. For example we can use it to have a look at the generated code.

   ```python
   print(tf.autograph.to_code(sum_even.python_function))
   ```



## Keras and AutoGraph

AutoGraph is available by default in non-dynamic Keras models. For more information, see [`tf.keras`](https://www.tensorflow.org/api_docs/python/tf/keras).

```python
class CustomModel(tf.keras.models.Model):

  @tf.function
  def call(self, input_data):
    if tf.reduce_mean(input_data) > 0:
      return input_data
    else:
      return input_data // 2


model = CustomModel()

model(tf.constant([-2, -4]))
```



## Side effects

Just like in eager mode, you can use operations with side effects, like **`tf.assign` or [`tf.print`](https://www.tensorflow.org/api_docs/python/tf/print) normally inside [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function)**, and it will insert the necessary control dependencies to ensure they execute in order. [I have question on this section.]

## Debugging

You can call [`tf.config.experimental_run_functions_eagerly(True)`](https://www.tensorflow.org/api_docs/python/tf/config/experimental_run_functions_eagerly) to temporarily enable eager execution inside the `tf.function' and use your favorite debugger:

```python
@tf.function
def f(x):
  if x > 0:
    # Try setting a breakpoint here!
    # Example:
    #   import pdb
    #   pdb.set_trace()
    x = x + 1
  return x

tf.config.experimental_run_functions_eagerly(True)

# You can now set breakpoints and run the code in a debugger.
f(tf.constant(1))

tf.config.experimental_run_functions_eagerly(False)
```



## An example of in-graph training

### Define the model

```python
model = tf.keras.Sequential((
    tf.keras.layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(10)))
model.build()
optimizer = tf.keras.optimizers.Adam()
```

### Define training loop

```python
compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

compute_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()


def train_one_step(model, optimizer, x, y):
  with tf.GradientTape() as tape:
    logits = model(x)
    loss = compute_loss(y, logits)

  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  compute_accuracy(y, logits)
  return loss


@tf.function
def train(model, optimizer):
  train_ds = mnist_dataset()
  step = 0
  loss = 0.0
  accuracy = 0.0
  for x, y in train_ds:
    step += 1
    loss = train_one_step(model, optimizer, x, y)
    if step % 10 == 0:
      tf.print('Step', step, ': loss', loss, '; accuracy', compute_accuracy.result())  # <-- tf.print()
  return step, loss, accuracy

step, loss, accuracy = train(model, optimizer)
print('Final step', step, ': loss', loss, '; accuracy', compute_accuracy.result())
```

### Batching

Skipped

### Re-tracing

Key points:

- Exercise caution when calling functions with non-tensor arguments, or with arguments that change shapes.
- Decorate module-level functions, and methods of module-level classes, and avoid decorating local functions or methods.

1. [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) can give you significant speedup over eager execution, at the cost of a **slower first-time execution**. This is because when executed for the first time, the function is also *traced* into a TensorFlow graph. Constructing and optimizing a graph is usually much slower compared to actually executing it.

2. [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) may also *re-trace* when called with different **non-tensor arguments**.
3. A *re-trace* can also happen when tensor arguments change shape, unless you specified an `input_signature`. See [this link](https://www.tensorflow.org/api_docs/python/tf/function) for what is input signature.

4. In addition, tf.function always creates a new graph function with its own set of traces whenever it is called:

   ```python
   def f():
     print('Tracing!')
     tf.print('Executing')
   
   tf.function(f)()
   tf.function(f)()
   ```

