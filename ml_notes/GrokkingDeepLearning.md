# Grokking Deep Learning

## Chapter 1: Introducing Deep Learning

* Whenever we encouter a math formula, translate its methods into an intuitive analogy to the real world and break it into parts. 

## Chapter 2: Fundamental Concepts

### What is deep learning?

![image-20201028152137176](figures/image-20201028152137176.png)

### What is machine learning

* Machine learning is a subfield of computer science wherein machines learn to perform tasks for which they were not *explicitly programmed*. 
  * Supervised vs. unsupervised learning

### Supervised machine learning

* Supervised learning is a method for transforming one dataset into another. 
  
  * Useful for taking what you know and transforming it into what you want to know.
    
    ![image-20201028152635622](figures/image-20201028152635622.png)

### Unsupervised machine learning

* Unsupervised learning groups your data.

* Transforms one dataset into another, like supervised learning, but unlike supervised learning, the output data set is not previously known. 
  
  * There is no known "right answer" beforehand
    
    ![image-20201028152854498](figures/image-20201028152854498.png)
  
  ​        ![image-20201028152906330](figures/image-20201028152906330.png)

* All forms of unsupervised learning can be viewed as a form of clustering.

### Parametric vs. nonparametric learning

* Oversimplified: trial-and-error learning vs counting and probability
  
  ![image-20201028153200443](figures/image-20201028153200443.png)

* A parametric model is characterized by having a fixed number of parameters

* A nonparametric model's number of parameters is infinite (determined by the data)

### Supervised parametric learning

* Oversimplified: trial-and-error learning using knobs
  * learning occurs by adjusting the knobs
  * the entirity of what the model has learned is captured by the positions of the knobs
* Three steps to supervised parametric learning
  1. Predict
  2. Compare to the truth
  3. Learn
     * adjust the knobs
     * knobs represent the prediction's sensitivity to different types of input data

### Unsupervised parametric learning

* Use knobs to group data

### Nonparametric learning

* Oversimplified: counting-based methods
* Number of parameters is based on the data (instead of predefined).

## Chapter 3: Introduction to neural prediction

### Step 1: Predict

* Predict, compare, learn paradigm

* **Rule of thumb**: always present enough information to the network, where "enough information" is loosely defined as how much a human might need to make the same prediction.

### A simple neural network making a prediction

```python
### single input, single output, single weight network
weight = 0.1
def neural_network(input, weight):
    prediction = input * weight
    return prediction


number_of_toes = [8.5, 9.5, 10, 9] # input data
input = number_of_toes[0]
pred = neural_network(input, weight)
print(pred)
```

### What is a neural network?

* **What is input data?**: It's a number that you record in the real world

* **What is a prediction?**: It's what the neural network tells you given the input data. 

* **How does the network learn?**: Trial and error. First make a prediction, compare it to the known output, and adjust the weights/parameters to minimize the prediction error. 

* The neural network API is simple: accept the input as information and the weight as knowledge. The network uses the weights/knowledge to interpret the input/information.

* Can think of weights as the networks sensitivity. Large weights have high sensitivity; small weights low sensitivity.

### Making a prediction with multiple inputs

* Neural networks can combine intelligence from multiple datapoints.
  
  ```python
  weights = [0.1, 0.2, 0]
  def neural_network(input, weights):
      pred = w_sum(input, weights)
      return pred
  
  def w_sum(a,b):
      assert(len(a) == len(b))
  
      output = 0
      for i in range(len(a)):
          output += a[i] * b[i]
  
      return output
  
  # multiple inputs
  toes = [8.5, 9.5, 9.9, 9.0]
  wlrec = [0.65, 0.8, 0.8, 0.9]
  nfans = [1.2, 1.3, 0.5, 1.0]
  
  input = [toes[0], wlrec[0], nfans[0]]
  pred = neural_network(input, weights)
  ```

![](/home/mopugh/Documents/typora/ml_notes/figures/2020-10-30-12-19-52-image.png)

    <img title="" src="file:///home/mopugh/Documents/typora/ml_notes/figures/2020-10-30-11-33-51-image.png" alt="" data-align="inline">

### Multiple inputs: What does this neural network do?

* It multiplies three inputs by three weights and sums them: a weighted sum or dot product.

* Need to combine multiple inputs at the same time: single features are not informative enough (e.g. number of toes). 

* store inputs and weights as vectors and matrices.

* elementwise operations perform operations on individual elements of a vector or matrix
  
  ```python
  def elementwise_multiplication(vec_a, vec_b):
      assert(len(vec_a) == len(vec_b))
      return [x[0] * x[1] for x in zip(vec_a, vec_b)]
  
  def elementwise_addition(vec_a, vec_b):
      assert(len(vec_a) == len(vec_b))
      return [x[0] + x[1] for x in zip(vec_a, vec_b)]
  
  def vector_sum(vec_a):
      return sum(vec_a)
  
  def vector_average(vec_a):
      assert(len(a) > 0)
      return vector_sum(vec_a) / len(vec_a)
  ```

* Dot product is essential to understand how neural networks make predictions. The dot product is a notion of similarity.

* Can think of negative weights as a logical `not` 

* Logical interpretation of weights
  
  ```python
  # This interpretation is CRUDE
  weights = [1, 0, 1] => if input[0] or input[2]
  weights = [1, 0, -1] => if input[0] or not input[2]
  weights = [0.5, 0, 1] => if BIG input[0] or input[2]
  ```
  
  * Note that the weight of 0.5 requires the input to be larger to have the same effect. 

* In the previous example, 
  
  * `nfans` is ignored due to the 0 weight
  
  * `ntoes` is dominant because it has the largest product of input and weight

* Cannot shuffle weights: their position determines what inputs they influence.

* The value of the weight and the input determine the overall impact on the final score. 

* Negative weights cause some inputs to reduce the final prediction.

### Multiple inputs: Complete runnable code

Numpy implementation

```python
weights = np.array([0.1, 0.2, 0])
def neural_network(input, weights):
    pred = input.dot(weights)
    return pred

toes = np.array([8.5, 9.5, 9.9, 9.0])
wlrec = np.array([0.65, 0.8, 0.8, 0.9])
nfans = np.array([1.2, 1.3, 0.5, 1.0])

input = np.array([toes[0], wlrec[0], nfans[0]])
pred = neural_network(input, weights)
print(pred)
```

### Making a prediction with multiple outputs

* Neural networks can also make a prediction using only a single input

* ![](/home/mopugh/Documents/typora/ml_notes/figures/2020-10-30-12-45-32-image.png)

* Note in this example, the neural network makes three separate predictions.
  
  ```python
  def neural_network(input, weights):
      return [input * w for w in weights]
  ```

### Predicting with multiple inputs and outputs

* Neural networks can predict multiple outputs given multiple inputs

    ![](/home/mopugh/Documents/typora/ml_notes/figures/2020-10-30-12-49-31-image.png)

![](/home/mopugh/Documents/typora/ml_notes/figures/2020-10-30-12-50-53-image.png)

```python
def dot(a,b):
    vector_sum(elementwise_multiplication(a,b))
    
def vect_mat_mul(vect, matrix):
    # assume matrix is a list of lists
    assert(len(vect) == len(matrix))
    
    output = [dot(vect, row) for row in matrix]
    return output

def neural_network(input, weights):
    return vect_mat_mul(input, weights)
```

### Multiple inputs and outputs: How does it work?

* It performs three independent weighted sums of the input to make three predictions.
  
  * Can think of it as three weights coming out of each input node, or
  
  * Three weights going into each output node.

* Think of list of numbers as a vector and a list of vectors as a matrix
  
  * vector matrix multiplication is the series of weighted sums: take a vector and perform a dot product with every row in a matrix

### Predicting on predictions

* Neural networks can be stacked

    ![](/home/mopugh/Documents/typora/ml_notes/figures/2020-10-30-13-00-28-image.png)

* `numpy` implementation
  
  ```python
  import numpy as np
  
  # toes % wins # fans
  ih_wgt = np.array([
      [0.1, 0.2, -0.1], # hid[0]
      [-0.1, 0.1, 0.9], # hid[1]
      [0.1, 0.4, 0.1], # hid[2]
  ])
  
  # hid[0] hid[1] hid[2]
  hp_wgt = np.array([
      [0.3, 1.1, -0.3], # hurt?
      [0.1, 0.2, 0.0], # win?
      [0.0, 1.3, 0.1], # sad?
  ])
  
  weights = [ih_wgt, hp_wgt]
  
  def neural_network(input, weights):
      hid = input.dot(weights[0])
      pred = hid.dot(weights[1])
      return pred
  
  toes = np.array([8.5, 9.5, 9.9, 9.0])
  wlrec = np.array([0.65, 0.8, 0.8, 0.9])
  nfans = np.array([1.2, 1.3, 0.5, 1.0])
  
  input = np.array([toes[0], wlrec[0], nfans[0]])
  
  pred = neural_network(input, weights)
  print(pred)
  ```

### A quick primer on NumPy