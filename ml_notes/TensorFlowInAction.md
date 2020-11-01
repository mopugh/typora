# TensorFlow In Action

## Chapter 1: The amazing world of TensorFlow

* Machine learning is the gatekepper that lets you cross from the world of data into the realm of information (e.g. actionable insights, useful patterns), bya llowing machines to learn from data.
* What is machine learning?
  * the process of training a computational model to predict some output given the data

### What is TensorFlow

* In its most raw form, TensorFlow provides three basic entities:
  * `tf.Variable`: a mutable data structure that can be used to store model parameters
  * `tf.Tensor`: immutable data structure that can be used to record data and interim and final ouputs of the model
  * `tf.Operation`: various operations provided in TensorFlow that are used to implement actual algorithms (e.g. matrix multiplication)
* Usually do not work at this level but rather use a higher level API such as Keras. 
* When a model is built, TensorFlow creates a data-flow graph and identifies hardware to execute on (e.g. GPU). 
* Tensorboard is a visualization tool to track the model as it trains.
* `SavedModel` used to save and store model
* TensorFlow serving helps deploy trained models and implement an API.

### CPU vs GPU

* CPUs can execute complex sequences of instructions very fast at a small scale (e.g. 8 cores in parallel)
* GPUs are good at executing basic instructions, typically slower than CPUs, but runs at a much larger scale (e.g. thousands of cores in parallel)

### When and when not to use TensorFlow?

#### When to use TensorFlow?

* Prototyping deep learning models
* Implementing models (including non-deep learning) that can run faster on optimized hardware
* Productionized Models / Seriving on cloud
* Monitoring models during training
* Creating heavy-duty data pipelines

#### When not to use TensorFlow?

* Implementing traditional machine learning models. Instead consider:
  * scikit-learn
  * xgboost
  * rapids.ai
* Manipulating and analyzing small-scale structured data
  * If the data can fit in memory, use Pandas or Numpy
* Creating complex natural language processing (NLP) pipelines

### What will this book teach you?

* TensorFlow fundamentals
* Deep learning algorithms
* Monitoring and optimization

### Who is this book for?

* Important question is **not** "how do I use TensorFlow to solve my problem?" but "how do I use TensorFlow effectively to solve my problem?"
* Effective solution:
  * minimizing lines of code without sacrificing readability
  * using latest and greatest features
  * utilizing optimization whenever possible
    * avoid loops
    * use vectorization
* This book: "enabling the reader to write effective TensorFlow solutions"

### Summary

* TensorFlow is an end-to-end machine learning framework that provides an ecosystem facilitating model prototyping, model building, model monitoring and model serving and more.

## Chapter 2: TensorFlow 2

* Tensorflow also features:
  * probabilistic machine learning (tensorflow probability)
  * computer graphics related computations
  * TensorFlow hub to get pre-trained models
  * TensorBoard for visualization/debugging

### TensorFlow 2 vs TensorFlow 1

* A multi-layer perceptron (MLP) is a neural network with an input layer, one or more hidden layers and an output layer, a.k.a. a fully connected network

* Example:

  ![image-20201101062446233](figures/image-20201101062446233.png)

  ```python
  import numpy as np
  import tensorflow as tf
  
  x = np.random.normal(size=[1,4]).astype('float32')
  
  init = tf.keras.initializers.RandomNormal()
  
  w1 = tf.Variable(init(shape=[4,3]))
  b1 = tf.Variable(init(shape=[1,3]))
  
  w2 = tf.Variable(init(shape=[3,2]))
  b2 = tf.Variable(init(shape=[1,2]))
  
  @tf.function
  def forward(x, W, b, act):
      return act(tf.matmul(x,W)+b)
  
  # computing h
  h = forward(x, w1, b1, tf.nn.sigmoid)
  #computing y
  y = forward(h, w2, b2, tf.nn.softmax)
  ```

  * `tf.Variable` are used for weights and biases. Requires an initializer
  * input `x` is a normal numpy array
  * `@tf.function` wrapper tells Python there is tensorflow code
  * `act` is the nonlinearity, e.g. tf.nn.sigmoid
  * `tf.matmul(x, W) + b` performs the linear transformation
  * The intermediate values `h` and `y` are of type `tf.Tensor`

#### How does TensorFlow operate under the hood?

* In the previous example, TensorFlow is 

  * defining a data-flow (computation) graph
  * executing the graph 

* TensorFlow 2 uses *imperative style execution*: declaration (defining the graph) and execution happen simultaneously. This is also known as *eagerly executing* code. 

* A data-flow graph is a DAG where edges are data and nodes are operations.

  * Example for `h = xW + b`

  ![image-20201101063919160](figures/image-20201101063919160.png)

* TensorFlow knows to create the data-flow graph by the `@tf.function` decorator

  * This feature is known as AutoGraph
    * AutoGraph useful when many repeated operations
    * If many different operations, can slow you down due to overhead of generating the graph
    * For loops are unrolled, so can run out of memory
    * numpy arrays and Python lists will be converted to tf.constant objects

  ![image-20201101064547626](figures/image-20201101064547626.png)

#### Visiting an old friend: TensorFlow 1

* Same example in TensorFlow 1

  ```python
  import numpy as np
  import tensorflow as tf
  
  x = tf.placeholder(shape=[1,4], dtype=tf.float32)
  
  w1 = tf.Variable(tf.initializers.truncated_normal()(shape=[4,3]), dtype=tf.float32)
  b1 = tf.Variable(tf.initializers.truncated_normal()(shape=[1,3]), dtype=tf.float32)
  
  w2 = tf.Variable(tf.initializers.truncated_normal()(shape=[3,2]), dtype=tf.float32)
  b2 = tf.Variable(tf.initializers.truncated_normal()(shape=[1,2]), dtype=tf.float32)
  
  h = tf.matmul(x,w1) + b1
  h = tf.nn.sigmoid(h)
  
  y = tf.matmul(x,w2) + b2
  y = tf.nn.softmax(y)
  
  with tf.Session() as sess:
      sess.run(tf.global_variables_initializers())
      res = sess.run(y, feed_dict={x: np.random.normal(size=[1,4])})
      print(res)
  ```

  * Clear distinction between graph definition and graph execution
    * prior to context manager: graph definition
    * in context manager: graph execution
  * In the above: `h` and `y` are symbolic as are variables. `res` is the actual output value
    * Need to initialize variables when executing the graph
  * TensorFlow 1 uses *declarative graph based execution*. 
    * First define data-flow graph using symbolic elements (e.g. placeholder inputs, variables) 
    * Explicitly write code to obtain or evaluate results from the graph
      * Feed in values to the previously defined symbolic elements

  ![image-20201101070233271](figures/image-20201101070233271.png)

### Building blocks in TensorFlow

* In TensorFlow 2, there are three major elements:
  * `tf.Variable`: e.g. `w1`, `b1`
  * `tf.Tensor`: e.g. `h`, `y`
  * `tf.Operation`: e.g. `tf.matmul`

#### Understanding `tf.Variable`

* A typical ML model has two types of data:
  * model parameters, which change as the model is optimized
  * outputs of the model that are static given the input data and model parameters

* `tf.Variable` is ideal for defining model parameters which are initialized and then optimized.

* `tf.Variable` must have three attributes:

  * a `shape`
  * an initial value
  * a data-type

* Example definition: `tf.Variable(initial_value=None, trainable=None, dtype=None)`

  * `initial_value` contains the initial value, usually provided by `tf.keras.initializers`
    * E.g. `tf.keras.initializers.RandomUniform()([4,3])`
  * `trainable`: accepts boolean to determine if value can change during optimization
  * `dtype` determines the data type of the data contained in the variable

* Example: 1-D length 4 constant value of 2:

  `tf.Variable(tf.constant(2.0, shape=[4]), dtype='float32')`

* Can also define from a bumpy array

  `tf.Variable(np.ones(shape=[4,3]), dtype='float32')`

* Use `tf.keras.initializers`

  `tf.Variable(tf.keras.initializers.RandomNormal()(shape=[3,4,5]), dtype='float32')`

* When printing a `tf.Variable`, see:

  * name of the variable
  * shape of the variable
  * data type of the variable
  * the initial value of the variable

* Can convert `tf.Variable` to numpy array: `arr = v1.numpy()` where v1 is of type `tf.Variable`

* Can change the value of elements of a variable of type `tf.Variable`

  ```python
  v = tf.Variable(np.zeros(shape=[4,3]), dtype='float32')
  v = v[0,2].assign(1)
  v = v[2:,1:].assign([[3,3],[3,3]])
  ```

* Exercise: create a `tf.Variable` of type `int16` with values ``[[1,2,3],[4,3,2]]`

  ```python
  ex2 = tf.Variable(np.array([[1,2,3],[4,3,2]]), dtype='int16')
  ```

  

#### Understanding tf.Tensor

![image-20201101104315420](figures/image-20201101104315420.png)

* `tf.Variable` is a mutable data structure whereas `tf.Tensor` is an immutable data structure

  * Can define a constant or the result of an operation on another `tf.Tensor` or `tf.Variable`

* Exercise: Create a `tf.Tensor` of size [4,1,5] that's randomly initialized from a normal distribution

  ```python
  ex3 = tf.constant (np.random.normal(size=[4,1,5]), dtype='float32')
  ```

#### Understanding `tf.Operation`

* Arithmetic and logical comparisons:

  ```python
  import tensorflow as tf
  import numpy as np
  
  a = tf.constant(4, shape=[4], dtype='float32')
  b = tf.constant(2, shape=[4], dtype='float32')
  
  # Arithmetic
  c = a+b
  d = a*b # Element-wise multiplication
  
  a2 = tf.constant([[1,2,3],[4,5,6]])
  b2 = tf.constant([[5,4,3],[3,2,1]])
  
  # element-wise comparison
  # returns boolean tf.Tensor
  equal_check = (a==b)
  leq_check = (a<=b)
  ```

* Reductions

  ```python
  # reductions
  a = tf.constant(np.random.normal(size=[5,4,3]), dtype='float32')
  
  # sum on all elements
  red_a1 = tf.reduce_sum(a)
  
  # element-wise product of each row
  red_a2 = tf.reduce_prod(a, axis=0) # result is of size=[4,3]
  
  # get the minimum over multiple axes
  red_a3 = tf.reduce_min(a, axis=[0,1]) # result is of size=[3]
  ```

  * When performing a reduction on a specific dimension, you lose that dimension

  * Can use `keepdims` parameter if you need to maintain that axis/dimension

    ```python
    # reducing with keepdims=False
    red_a4 = tf.reduce_min(a, axis=1) # result is of size=[5,3]
    # reducing with keepdims=True
    red_a5 = tf.reduce_min(a, axis=1, keepdims=True) # result is of size [5,1,3]
    ```

  * Other useful reductions:
    * `tf.argmax`
    * `tf.argmin`
    * `tf.cumsum`
    * `tf.reduce_mean`

* Exercise: Given an array `[[0.5, 0.2, 0.7], [0.2, 0.3, 0.4], [0.9, 0.1, 0.1]]`, use `tf.reduce_mean` to compute the mean of each column

  ```python
  a = tf.constant(np.array([[0.5, 0.2, 0.7],[0.2, 0.3, 0.4], [0.9, 0.1, 0.1]]), dtype='float32')
  col_mean = tf.reduce_mean(a, axis=1)
  ```

### Neural network related computations in TensorFlow

