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

 