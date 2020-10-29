# Deep Learning With PyTorch

## Part 1: Core PyTorch

### Chapter 1: Introducing Deep Learning and the PyTorch Library

* Conflate self-awareness with intelligence, but it is not required to perform some tasks which machine learning is good at. 

#### The deep learning revolution

* Before deep learning, machine learning was heavily dependent on *feature engineering*. 
  * Features are transformations on the input data that facilitate downstream algorithms, e.g. classification
  * Deep learning finds the representations automatically to perform the desired task.

<img src="./figures/image-20201019135710990.png" alt="image-20201019135710990" style="zoom:100%;" />

* Steps to execute successful deep learning:
  * Need a way to ingest whatever data we have at hand
  * We need to somehow define the deep learning machine
  * We must have an automated way, *training*, to obtain useful representation and make the machine produce the desired outputs
* Training:
  * Have a *criterion* to measure discrepancy between desired and actual output
  * Modify the machine to minimize the discrepancy

#### PyTorch for deep learning

* Core PyTorch data structure: *tensor* 

#### Why PyTorch?

PyTorch has two features that make it highly attractive for deep learning

* GPU support
* Numerical optimization on generic mathematical expressions

##### The deep learning competitive landscape

Three main libraries currently:

* PyTorch
* TensorFlow
* JAX

#### An overview of how PyTorch supports deep learning projects

* At it's core, PyTorch provides *tensors* (i.e. multidimensional arrays) and operations on them that can be used on the CPU or GPU as well as keeping track of operations on tensors and being able to compute the derivative. 
* Core PyTorch module: `torch.nn`
* To train a model:
  * need source of training data
  * an optimizer to adapt the model to the training data
  * a way to get the model and data to the hardware that will be performing the calculations
* The bridge between custom data and PyTorch tensor is the `Dataset` class in `torch.utils.data` 
  * Since the data varies from problem to problem, will need to write the data sourcing ourselves.
* Want to batch data for parallel computation:
  * `DataLoader` class
* Compare output of model to input data and compute loss using loss functions
  * Found in `torch.nn` 
* `torch.optim` provides optimizers to adjust the model parameters.

### Summary

* Deep learning models automatically learn to associate inputs and desired outputs from examples
* Libraries like PyTorch allow you to build and train neural network models efficiently
* PyTorch defaults to immediate execution for operations

## Chapter 2: Pretrained networks

* Using off-the-shelf models can be a quick way to jump-start a deep learning project.

### A pretrained network that recognizes the subject of an image



