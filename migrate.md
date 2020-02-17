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
3. Use Python **objects** to track variables and losses
   1. All name-based variable tracking is strongly discouraged in TF 2.0. Use Python objects to to track variables.
   2. Use [`tf.Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable) instead of [`v1.get_variable`](https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable).
   3. Every [`v1.variable_scope`](https://www.tensorflow.org/api_docs/python/tf/compat/v1/variable_scope) should be converted to a Python object. Typically this will be one of:
      1. [`tf.keras.layers.Layer`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer)
      2. [`tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model)
      3. [`tf.Module`](https://www.tensorflow.org/api_docs/python/tf/Module)
   4. If you need to aggregate lists of variables (like [`tf.Graph.get_collection(tf.GraphKeys.VARIABLES)`](https://www.tensorflow.org/api_docs/python/tf/Graph#get_collection)), use the `.variables` and `.trainable_variables` attributes of the `Layer` and `Model` objects. These `Layer` and `Model` classes implement several other properties that remove the need for global collections. Their `.losses` property can be a replacement for using the `tf.GraphKeys.LOSSES` collection.
4. Upgrade your training loops
   1. Use the highest level API that works for your use case. **Prefer [`tf.keras.Model.fit`](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit) over building your own training loops.**
   2. These high level functions manage a lot of the low-level details that might be easy to miss if you write your own training loop. For example, they automatically collect the regularization losses, and set the `training=True` argument when calling the model.
5. Upgrade your data input pipelines
6. Migrate off `compat.v1` symbols

## Converting models



 

