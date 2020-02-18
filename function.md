# tf.function

Compile a function into a callable TensorFlow graph.

```python
tf.function(
    func=None,
    input_signature=None,
    autograph=True,
    experimental_implements=None,
    experimental_autograph_options=None,
    experimental_relax_shapes=False,
    experimental_compile=None
)
```

- [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) constructs a callable that executes a TensorFlow graph ([`tf.Graph`](https://www.tensorflow.org/api_docs/python/tf/Graph)) created by trace-compiling the TensorFlow operations in `func`, effectively executing `func` as a TensorFlow graph.

- `func` may use data-dependent control flow, including `if`, `for`, `while` `break`, `continue` and `return` statements

- `func`'s closure may include [`tf.Tensor`](https://www.tensorflow.org/api_docs/python/tf/Tensor) and [`tf.Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable) objects:

- `func` may also use ops with side effects, such as [`tf.print`](https://www.tensorflow.org/api_docs/python/tf/print), [`tf.Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable) and others

- **Important:** Any Python side-effects (appending to a list, printing with `print`, etc) **will only happen once**, when `func` is traced. To have side-effects executed into your [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) they need to be written as TF ops

  ```python
  l = [] 
  @tf.function 
  def f(x): 
    for i in x: 
      l.append(i + 1)    # Caution! Will only happen once when tracing 
  f(tf.constant([1, 2, 3])) 
  l 
  # [<tf.Tensor ...>]
  ```

  Instead, use TensorFlow collections like `tf.TensorArray`:

  ```python
  @tf.function 
  def f(x): 
    ta = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True) 
    for i in range(len(x)): 
      ta = ta.write(i, x[i] + 1) 
    return ta.stack() 
  f(tf.constant([1, 2, 3])) 
  # <tf.Tensor: ..., numpy=array([2,3,4], ...)>
  ```

- **IMPORTANT**: *[`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) is polymorphic*

  Internally, [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) can build more than one graph, to support arguments with different data types or shapes, since TensorFlow can build more efficient graphs that are specialized on shapes and dtypes. [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) also treats any pure Python value as opaque objects, and builds a separate graph for each set of Python arguments that it encounters.

  To obtain an individual graph, use the `get_concrete_function` method of the callable created by [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function). It can be called with the same arguments as `func` and returns a special [`tf.Graph`](https://www.tensorflow.org/api_docs/python/tf/Graph) object:

  ```python
  @tf.function 
  def f(x): 
    return x + 1 
  isinstance(f.get_concrete_function(1).graph, tf.Graph) 
  # True
  ```

  **CAUTION**: Passing python scalars or lists as arguments to [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) will always build a new graph. To avoid this, pass numeric arguments as Tensors whenever possible:

  ```python
  @tf.function 
  def f(x): 
    return tf.abs(x) 
  f1 = f.get_concrete_function(1) 
  f2 = f.get_concrete_function(2)  # Slow - builds new graph 
  f1 is f2 
  # False
  f1 = f.get_concrete_function(tf.constant(1)) 
  f2 = f.get_concrete_function(tf.constant(2))  # Fast - reuses f1 
  f1 is f2 
  # True
  ```

  **Python numerical arguments should only be used when they take few distinct values, such as hyperparameters like the number of layers in a neural network.**

- `input_signature`: For Tensor arguments, [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) instantiates a separate graph for **every unique set of input shapes and datatypes. **The example below creates two separate graphs, each specialized to a different shape:

  ```python
  @tf.function 
  def f(x): 
    return x + 1 
  vector = tf.constant([1.0, 1.0]) 
  matrix = tf.constant([[3.0]]) 
  f.get_concrete_function(vector) is f.get_concrete_function(matrix) 
  # False
  ```

  An "input signature" can be optionally provided to [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) to control the graphs traced. The input signature specifies the shape and type of each Tensor argument to the function using a [`tf.TensorSpec`](https://www.tensorflow.org/api_docs/python/tf/TensorSpec) object. More general shapes can be used. This is useful to avoid creating multiple graphs when Tensors have dynamic shapes. It also restricts the dhape and datatype of Tensors that can be used:

  ```python
  @tf.function( 
      input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)]) 
  def f(x): 
    return x + 1 
  vector = tf.constant([1.0, 1.0]) 
  matrix = tf.constant([[3.0]]) 
  f.get_concrete_function(vector) is f.get_concrete_function(matrix) 
  # True
  ```

- Variables may only be created once. [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) only allows creating new [`tf.Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable) objects when it is called for the first time:

  ```python
  class MyModule(tf.Module): 
    def __init__(self): 
      self.v = None 
   
    @tf.function 
    def call(self, x): 
      if self.v is None: 
        self.v = tf.Variable(tf.ones_like(x)) 
      return self.v * x 
  ```

  In general, it is **recommended to create stateful objects like [`tf.Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable) outside of [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) and passing them as arguments.**



Only one thing to notices in `Args`:

* **`func`**: the function to be compiled. If `func` is None, [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) returns a decorator that can be invoked with a single argument - `func`. In other words, `tf.function(input_signature=...)(func)` is equivalent to [`tf.function(func, input_signature=...)`](https://www.tensorflow.org/api_docs/python/tf/function). The former can be used as decorator.



See this [link](https://www.tensorflow.org/api_docs/python/tf/function) for full knowledge of `function`.