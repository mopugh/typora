# Probabilistic Graphical Models

## Probabilistic Graphical Models: Principles and Techniques

## [Stanford PGM Course: CS228](https://ermongroup.github.io/cs228-notes/)

### Introduction

- Probabilistic graphical modeling is a branch of machine learning that studies how to use probability distributions to describe the world and to make useful predictions about it.
- Goal of modeling is to be able to describe reality
  - Answer questions
  - Make predictions
- What is a difficulty of using probablistic models?
  - Probabilities are exponential sized objects
  - Example: **Spam Detection**
    - Let $p_{\theta}(y,x_{1},\ldots,x_{n})$ be the joint probability of word occurances in an e-mail and whether or not it is spam. Then these binary variables lead to $2^{n+1}$ possible values.
    - How do you store the function?
    - How do you estimate/predict on such a function?
- **Main simplifying assumption**: Conditional independence
- **Naive Bayes Assumption**: $x_{1},\ldots,x_{n}$ are conditionally independing given $y$
  - $P(y,x_{1},\ldots,x_{n}) = p(y) \Pi_{i=1}^{n} p(x_{i} \vert y)$
  - Each factor $p(x_{i} \vert y)$ can be described using four parameters (when the variables are binary)
  - ![Graphical representation of the Naive Bayes spam classification model. We can interpret the directed graph as indicating a story of how the data was generated: first, a spam/non-spam label was chosen at random; then a subset of $n$ possible English words were sampled independently and at random.](https://ermongroup.github.io/cs228-notes/assets/img/naive-bayes.png)
    - Select $y$ and then sample words independently
  - Want to be able to ask questions about the model.
- **Three main parts of probablistic graphical models:**
  - *Representation* (how to specify a model)
    - The core idea is to assume that the global structure of the probability distribution is determined by composing local structures, each of which is much simpler to handle.
    - There are two main classes of graphical models: directed models (Bayesian networks) and undirected models (Markov random fields).
    - models: directed models (Bayesian networks) and undirected models (Markov random fields). *Factor graphs* are also a convenient representation that can represent either directed or undirected models.
    - the major difference between directed and undirected models is that undirected models have a normalization constant called the *partition function*.
  - *Inference* (how to ask questions of the model)
    - Two main types of questions:
      - marginal inference
      - maximum a posteriori (MAP) inference
        - computing modes of distribution
    - Approximate inference
      - Sampling
        - Markov chaing monte carlo
      - Variational 
        - Optimization
  - *Learning* (how to fit a model to real-world data)
    - *Bayesian Predictive Distribution*: $p(x^{new} \vert \mathbf{x}) = \int p(x^{new} \vert \theta) p(\theta \vert \mathbf{x})d \theta$

### Probability Review



## Bayesian Reasoning and Machine Learning

## Graphical Models, Exponential Families, and Variational Inference





