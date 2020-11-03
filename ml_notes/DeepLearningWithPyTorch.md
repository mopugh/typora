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
  * Can think of pre-trained model as a program that takes inputs and generates outputs

### A pretrained network that recognizes the subject of an image

![image-20201102214051819](figures/image-20201102214051819.png)

* The input will be images of type `torch.Tensor` with two spatial dimensions and one color dimension (3 color channels). 
* The output will be a `torch.Tensor` of 1000 elements each representing the score associated with a class

#### Obtaining a pretrained network for image recognition

* TorchVision has many datasets and pretrained models
  * Capitalized objects are classes
  * Lowercase objects are convenience functions that return models instantiated from the classes

#### AlexNet

![image-20201102215026455](figures/image-20201102215026455.png)

#### ResNet

```python
from torchvision import models

# see all objects in models
dir(models)

# create an instance of ResNet101
resnet = model.resnet101(pretrained=True)

# print to see details of the model
print(resnet) # one module/layer per line

# Can call resnet like a function to produce output scores.

# Preprocess images
from torchvision import transforms
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]) # need to normalize to what model expects
])

# test on image
# test on image
from pathlib import Path
from PIL import Image
import torch

filename = Path('../dlwpt-code/data/p1ch2/bobby.jpg')
img = Image.open(filename)
img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0)

# running the model is also called inference
# need to put network in eval mode
resent.eval()
out = resnet(batch_t)

# Verify inference
labelfile = Path('../dlwpt-code/data/p1ch2/imagenet_classes.txt')
with open(labelfile) as f:
    labels = [line.strip() for line in f.readlines()]

# Find maximum class score
# torch.max returns maximum value and the index
_, index = torch.max(out, 1) # index is 1d tensor, so need to index into it

# Turns scores into probabilities
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
labels[index[0]], percentage[index[0]].item()

# Use sort function to find other scores
# returns the indices in the original tensor
_, indices = torch.sort(out, descending=True)
[(labels[idx], percentage[idx].item()) for idx in indices[0][:5]] #indices is a 2d tensor
```

### A pretrained model that fakes it until it makes it

##### The GAN game

