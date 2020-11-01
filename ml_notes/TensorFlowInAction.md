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

