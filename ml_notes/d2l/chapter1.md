# D2L

## Chapter 1: Introduction

### Motivating Example

* **Model**: A fixed set of knobs (parameters)

* **Family of models**: set of distinct programs (input-output mappings) produced by the parameters

* **Learning algorithm**: the meta-program that uses data to choose the parameters<img title="" src="file:///home/mopugh/Documents/typora/ml_notes/figures/b8e47ea963144820d7219370374532a02bc122e0.png" alt="" data-align="left">

* "Programming with data"

### Key Components

1. The **data** that we can learn from

2. A **model** of how to transform the data

3. An **objective function** that quantifies how well (or  badly) the model is doing

4. An **algorithm** to adjust the model's parameters to optimize the objective function

### Kinds of Machine Learning Problems

#### Supervised Learning

* predicting labels given input features

* The supervised learning algorithm takes in the training data set and returns a learned model
  
  ![](/home/mopugh/Documents/typora/ml_notes/figures/f9dae24ed9377d3302cfd808a586801562e3de9a.png)

##### Regression

* the label is an arbitrary  numerical value
  
  * "How much?" problems
  
  * "How many?" problems

##### Classification

* "Which one?" type problems
  
  * assign category (class)

##### Tagging

* Multi-label classification
  
  * E.g. tagging subjects of a blog post

##### Search

* Care about the rankings of the results
  
  * E.g. the results from a search engine

##### Recommender Systems

##### Sequence Learning

* model ingests sequences of inputs or emits sequences of outputs (or both)

* Examples:
  
  * Tagging and parsing
  
  * Automatic speech recognition
  
  * Text to speech
  
  * Machine translation

#### Unsupervised Learning

* clustering

* subspace estimation
  
  * E.g. principal component analysis

* embeddings

* causality and probabilistic graphical models

* generative adversarial networks

#### Interacting with an Environment

* Supervised learning
  
  ![](/home/mopugh/Documents/typora/ml_notes/figures/6db0c90d040ab35bdb96a32aa89e34fba7cecfb8.png)
  
  * offline learning

* Consider interactions with the environment

#### Reinforcement Learning

* policy is a function that maps from observations of the environment to actions
  
  ![](/home/mopugh/Documents/typora/ml_notes/figures/df2d232286ab5f43a43dfcf36e1ff666921e93cb.png)

* Reinforcement learning is very general
  
  * Can recast supervised learning as reinforcement learning
    
    > We could create a reinforcement learning agent with one action corresponding to each class. We could then create an environment which gave a reward that was exactly equal to the loss function from the original supervised learning problem.

* **Credit assignment problem**: determining which actions to credit or blame for an outcome

* Partial observability
  
  * Current observation might not tell you everything about the current state

* Exploitation vs. exploration

* **Markov decision process**: fully observed environment

* **Contextual bandit problem**: the state does not depend on the previous actions

* **Multi-armed bandit**: No stae, just a set of available actions

### Roots

Key principles at the heart of neural networks:

* The alternation of linear and nonlinear processing units, often referred to as layers

* The use of the chain rule (also known as **backpropagation**) for adjusting the parameters in the entire network at once

### The Road to Deep Learning

### Success Stories

### Characteristics

* machine learning can use data to learn transformations between inputs and outputs

* *Deep learning is deep* in precisely the sense that its models learn many layers of transformations
  
  * each layer offers the representation at one level
    
    * e.g. layers near the input may represent more low-level details of the data while layers closer to the classification output may represent more abstract concepts used for discrimination

* *representation learning* aims at finding the representation itself
  
  * *deep learning* can be referred to as multi-level representation learning

* Most significant commonality in deep learning methods is the use of *end-to-end training*
  
  * Rather than assembling a system based on individually tuned components, build a system that tuns the performance jointly
    
    * E.g. replace feature engineering

## Summary

* Whole system optimization is a key component in obtaining high performance
