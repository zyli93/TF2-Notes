# Deep Learining with PyTorch: A 60 Minute Blitz

## Intro

### Operations

1. multiple syntaxes for operations

```python
# add 1
y = torch.rand(5,3)
print(x + y)

# add 2
torch.add(x, y)

# add 3
result = torch.empty(5,3)
torch.add(x, y, out=result)

# add 4
y.add_(x)
```

Any operation that mutates a tensor **in-place** is post-fixed with an `_`. For example: `x.copy_(y)`, `x.t_()`, will change `x`.

2. Reshape: `view()`

### Numpy Bridge

1. Tensor to numpy: `b = a.numpy()`
2. Numpy to Tensor: `b = torch.from_numpy()`



### CUDA Tensors

Tensors can be moved onto any device using the `.to` method.

```python
# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))   # ``.to`` can also change dtype together!
```



## AUTOGRAD

### Tensors

`backward, detach, no_grad, `

1. `torch.Tensor` is the central class of the package. If you set its attribute `.requires_grad` as `True`, it starts to track all operations on it. When you finish your computation you can call `.backward()` and have all the gradients computed automatically. The gradient for this tensor will be accumulated into `.grad` attribute.

2. To stop a tensor from tracking history, you can call `.detach()` to detach it from the computation history, and to prevent future computation from being tracked.

3. To prevent tracking history (and using memory), you can also wrap the code block in `with torch.no_grad():`. This can be particularly helpful when evaluating a model because the model may have trainable parameters with `requires_grad=True`, but for which we don’t need the gradients.

4. There’s one more class which is very important for autograd implementation - a `Function`. `Tensor ` and ` Function` are **interconnected** and build up an acyclic graph, that encodes a complete history of computation. Each tensor has a `.grad_fn` attribute that references a `Function` that has created the `Tensor` **(except for Tensors created by the user - their `grad_fn is None`).**

5. If you want to compute the derivatives, you can call `.backward()` on a `Tensor`. If `Tensor` is a scalar (i.e. it holds a one element data), you don’t need to specify any arguments to `backward()`, however if it has more elements, you need to specify a `gradient` argument that is a tensor of matching shape.
6. `.requires_grad_( ... )` changes an existing Tensor’s `requires_grad` flag in-place. The input flag defaults to `False` if not given.



### Gradients

You can also stop autograd from tracking history on Tensors with `.requires_grad=True` either by wrapping the code block in `with torch.no_grad()`

```python
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)
    
# Output
#   True
# 	True
# 	False
```

Or by using `.detach()` to get a new Tensor with the same content but that does not require gradients:

```python
print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
print(x.eq(y).all())

# Output
# 	True
# 	False
# 	tensor(True)
```



## Neural Network

### Define a network

* You just have to define the `forward` function, and the `backward` function (where gradients are computed) is automatically defined for you using `autograd`. You can use any of the Tensor operations in the `forward` function.

* The learnable parameters of a model are returned by `net.parameters()`. 

* Zero the gradient buffers of all parameters and backprops with random gradients:

  ```python
  net.zero_grad()
  out.backward(torch.randn(1, 10))
  ```

* **`torch.nn` only supports mini-batches.** The entire `torch.nn` package only supports inputs that are a mini-batch of samples, and not a single sample. For example, `nn.Conv2d` will take in a 4D Tensor of `nSamples x nChannels x Height x Width`. If you have a single sample, just use `input.unsqueeze(0)` to add a fake batch dimension.

### Loss Function

* So, when we call `loss.backward()`, the whole graph is differentiated w.r.t. the loss, and all Tensors in the graph that has `requires_grad=True` will have their `.grad` Tensor accumulated with the gradient.

* For illustration, let us follow a few steps backward:

  ```python
  print(loss.grad_fn)  # MSELoss
  print(loss.grad_fn.next_functions[0][0])  # Linear
  print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU
  ```



### Update the weights



```python
import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update
```



### Training on GPU

1. Just like how you transfer a Tensor onto the GPU, you transfer the neural net onto the GPU. Let’s first define our device as the first visible cuda device if we have CUDA available:

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)
```

2. Then these methods will recursively go over all modules and convert their parameters and buffers to CUDA tensors:

```python
net.to(device)  # send all to cuda
```

3. Remember that you will have to send the inputs and targets at every step to the GPU too:

```python
inputs, labels = data[0].to(device), data[1].to(device)
```





## Data Parallelism

It’s very easy to use GPUs with PyTorch. You can put the model on a GPU:

```python
device = torch.device("cuda:0")
model.to(device)
```

Then, you can copy all your tensors to the GPU:

```python
mytensor = my_tensor.to(device)
```

Please note that just calling `my_tensor.to(device)` returns a new copy of `my_tensor` on GPU instead of rewriting `my_tensor`. You need to assign it to a new tensor and use that tensor on the GPU.

It’s natural to execute your forward, backward propagations on multiple GPUs. However, Pytorch will only use one GPU by default. You can easily run your operations on multiple GPUs by making your model run parallelly using `DataParallel`:

```python
model = nn.DataParallel(model)
```





## TensorBoard

### TensorBoard Setup

```python
from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/fashion_mnist_experiment_1')

# The corresponding cmd to run TensorBoard: `tensorboard --logdir=runs`
```

Note that this line alone creates a `runs/fashion_mnist_experiment_1` folder.

### Writing to TensorBoard

Add image to TensorBoard, specifically, grid:

```python
writer.add_image('four_fashion_mnist_images', img_grid)
```

### Inspect the model using TensorBoard

```python
writer.add_graph(net, images)
writer.close()
```

### Adding a "Projector" to TensorBoard

```python
writer.add_embedding(features,
                    metadata=class_labels,
                    label_img=images.unsqueeze(1))
writer.close()
```

###  Tracking model training with TensorBoard

```python
writer.add_scalar('training loss',
                  running_loss / 1000,
                  epoch * len(trainloader) + i)
```

Skipped the `load a figure`.

### Assessing trained models with TensorBoard

```python
# 1. gets the probability predictions in a test_size x num_classes Tensor
# 2. gets the preds in a test_size Tensor
# takes ~10 seconds to run
class_probs = []
class_preds = []
with torch.no_grad():  # <-- use in the test set!
    for data in testloader:
        images, labels = data
        output = net(images)
        class_probs_batch = [F.softmax(el, dim=0) for el in output]
        _, class_preds_batch = torch.max(output, 1)  # <-- ? what else returned?

        class_probs.append(class_probs_batch)
        class_preds.append(class_preds_batch)

test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
test_preds = torch.cat(class_preds)

# helper function
def add_pr_curve_tensorboard(class_index, test_probs, test_preds, global_step=0):
    '''
    Takes in a "class_index" from 0 to 9 and plots the corresponding
    precision-recall curve
    '''
    tensorboard_preds = test_preds == class_index
    tensorboard_probs = test_probs[:, class_index]

    writer.add_pr_curve(classes[class_index],
                        tensorboard_preds,
                        tensorboard_probs,
                        global_step=global_step)  # <-- add PR curve
    writer.close()

# plot all the pr curves
for i in range(len(classes)):
    add_pr_curve_tensorboard(i, test_probs, test_preds)
```

