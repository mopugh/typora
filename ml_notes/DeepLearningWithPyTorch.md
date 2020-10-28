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

