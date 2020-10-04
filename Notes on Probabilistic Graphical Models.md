# Probabilistic Graphical Models

## Change Log?

* 9/30/20: Added chapter one to "Probabilistic Graphical Models: Prinicples and Techniques".
* 10/1/20: Added notes on Probability Review from cs228 stanford notes upto and including sections 2.3.
* 10/4/20: Finished Section 2 of Probability Review from cs228 stanford notes

## Probabilistic Graphical Models: Principles and Techniques

### Chapter 1: Introduction

* **Declarative Representation**: construction of a model of the system
  * The key property of a declarative representation is the separation of knowledge and reasoning. 
    * Can develop general algorithms
    * Can improve model without modifying algorithms
* Uncertainty arises because of
  * limitations in our ability to observe the world
  * limitations in our ability to model the world
  * possibly innate determinism 
* To obtain meaning conclusions, need to not just reason about what is possible, but what is probable
* ![image-20200930063903786](/Users/mopugh/Library/Application Support/typora-user-images/image-20200930063903786.png)
* A graph is a compact representation of a set of independies of the form X is independent of Y given Z
  * $P(\text{Congestion} \vert \text{Flu, Hayfever, Season}) = P(\text{Congestion} \vert \text{Flu, Hayfever})$
    * Note: Season is not independent of congestion. 
* The graph defines a skeleton for compactly representing a high dimensional distribution
  * Break up joint distribution into a product of smaller factors
  * The graph structure defines the factorization of the distribution P
    * The set of factors and the variables they encompass
* The graph representation of a set of independencies and the graph as a skeleton for factorizing the distribution are equivalent
  * The independence properties of the distribution are what allow it to be represented compactly in a factorized form
  * A particular factorization of the distribution guarantees that certain independencies hold
* Two families of graphical representations
  * Bayesian networks: directed graph
  * Markov networks: undirected graph
  * Differ in the set of independencies they encode and the factorization of the distribution they induce
* Graphical models useful because in practice variables tend to interact directly only with very few other variables
* For graphical models, representation, inference and learning are critical:
  * Need a representation that is a reasonable encoding of the world model
  * Need to use this representation for inference to answer questions of interest
  * Need to acquire the distribution combining expert knowledge and accumulated data (learning)

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

#### Elements of Probability

* **Sample Space** $\Omega$: The set of all outcomes of a random experiment. $\omega \in \Omega$ is an individual outcome.
* **Set of Events (Event Space)** $F$: A set whose elements $A \text{ in } F$ (called events) are subsets of $\Omega$. 
* **Probability Measure**: A function $P: F \rightarrow \mathbb{R}$ satisfying
  * $P(a) \geq 0$ for all $A \in F$
  * $P(\Omega) = 1$
  * If $A_{1}, A_{2}, \ldots$ are disjoint events (i.e. $A_{i} \cap A_{j} = \emptyset$), then $P(\cup_{i} A_{i}) = \sum_{i} P(A_{i})$ 
  * These three properties are the **axioms of probability**
* **Properties**
  * $A \subseteq B \Rightarrow P(A) \leq P(B)$
  * $P(A \cap B) \leq \min (P(A), P(B))$
  * **Union Bound**: $P(A \cup B) \leq P(A) + P(B)$
  * $P(\Omega - A) = 1 - P(A)$
  * **Law of Total Probability**: If $A_{1}, \ldots, A_{k}$ are a set of disjoint events such that $\cup_{i=1}^{k} A_{i} = \Omega$, then $\sum_{i=1}^{k} P(A_{i}) = 1$

#### Conditional Probability

Let $B$ be an event with non-zero probability. Then the conditional probability of any event $A$ given $B$ is 
$$
P(A \vert B) = \frac{P(A \cap B)}{P(B)}
$$

#### Chain Rule

Let $S_{1}, \ldots, S_{k}$ be events and $P(S_{i}) > 0$ for all i. Then the chain rule states:
$$
P(S_{1} \cap S_{2} \cap \cdots \cap S_{k}) = P(S_{1}) P(S_{2} \vert S_{1}) P(S_{3} \vert S_{2} \cap S_{1}) \cdots P(S_{k} \vert S_{1} \cap S_{2} \cap \cdots S_{k-1})
$$
*Outline of Proof*: Example with k = 4
$$
\begin{aligned}
&P(S_{1} \cap S_{2} \cap S_{3} \cap S_{4}) \\
&= P(S_{1}\cap S_{2} \cap S_{3})P(S_{4} \vert S_{1} \cap S_{2} \cap S_{3}) \\
&= P(S_{1} \cap S_{2}) P(S_{3} \vert S_{1} \cap S_{2}) P(S_{4} \vert S_{1} \cap S_{2} \cap S_{3}) \\
&= P(S_{1}) P(S_{2} \vert S_{1}) P(S_{3} \vert S_{1} \cap S_{2}) P(S_{4} \vert S_{1} \cap S_{2} \cap S_{3})
\end{aligned}
$$

#### Independence

* **Definition**: Two events are **independent** if
  * $P(A \cap B) = P(A) P(B)$ or equivalently
  * $P(A \vert B) = P(A)$
* Intuitively, observing the event $B$ does not affect the probability of $A$

### Random Variables

* A random variable $X$ is a function $X: \Omega \rightarrow \mathbb{R}$.
  * Denoted $X(\omega)$ or $X$. 
  * The value that a random variable $X$ takes on is denoted $x$. 
    * $X = x$ means we assign the value of $x \in \mathbb{R}$ to the random variable $X$. 
* **Example**: Number of heads in 10 coin tosses
  * $X(\omega)$ is the number of heads which occur in the sequence of tosses $\omega$
  * $P(X = k) \doteq P(\{\omega : X(\omega) = k \})$
  * **Note**: In this example the possible outcomes are discrete, so $X$ is a *discrete random variable*
* **Example**: Suppose $X(\omega)$ is a random variable denoting the amount of time taken for a radioactive particle to decay. In this case the random variable $X(\omega)$ is continuous, so it is called a *continuous random variable*. 
  * Need to consider the random variable lying in an interval
  * $P(a \leq X \leq b) \doteq P(\{ w : a \leq X(\omega) \leq b \})$
* **Definition**: Indicator function $\mathbf{1}_{A}$ equals 1 if the event $A$ happens and zero otherwise.

#### Cumulative Distribution Functions

* **Definition**: A **cumulative distribution function** (CDF) is a function $F_{X}: \mathbb{R} \rightarrow [0,1]$ such that
  * $F_{X}(x) = P(X \leq x)$
* **Properties**:
  * $0 \leq F_{X}(x) \leq 1$
  * $\lim_{x \rightarrow -\infty} F_{X}(x) = 0$
  * $\lim_{x \rightarrow \infty} F_{X}(x) = 1$
  * $x \leq y \Rightarrow F_{X}(x) \leq F_{X}(y)$ 

#### Probability Mass Function

* When $X$ is a discrete random variable, can specify the probability of outcomes directly. The **probability mass function** (PMF) is a function $p_{X} : \Omega \rightarrow [0,1]$ such that
  * $p_{X}(x) = P(X = x)$
* Let $Val(X)$ denote the possible values $X$ may assume.
* **Properties**:
  * $0 \leq p_{X}(x) \leq 1$
  * $\sum_{x \in Val(X)} p_{X}(x) = 1$
  * $\sum_{x \in A} p_{X}(x) = P(X \in A)$

#### Probability Density Functions

* For continuous random variables which have a continuously differentiable CDF, the **probability density function** is defined as the derivative of the CDF:

  * $f_{X}(x) = \frac{dF_{X}(x)}{dx}$

* For small $\delta x$: $P(x \leq X \leq x + \delta x) \approx f_{X}(x) \delta x$

* **Note**: the value fo the pdf at a point $x$ **is not** the probability of that event, i.e. $f_{X}(x) \neq P(X = x)$ 

* **Properties**

  * $f_{X}(x) \geq 0$
* $\int_{-\infty}^{\infty} f_{X}(x)dx = 1$
  * $\int_{x \in A} f_{X}(x) dx = P(X \in A)$

#### Expectation

* Suppose $X$ is a discrete random variable with PMF $p_{X}(x)$ and $g: \mathbb{R} \rightarrow \mathbb{R}$ is an arbitrary function. Then $g(X)$ is a random variable and the **expectation** or **expected value** of $g(X)$ is 
  $$
  \mathbb{E} [g(X)] = \sum_{x \in Val(X)} g(x) p_{X}(x)
  $$

* If $X$ is a continuous random variable with PDF $f_{X}(x)$, then the expected value of $g(X)$ is 
  $$
  \mathbb{E} [g(X)] = \int_{-\infty}^{\infty} g(x) f_{X}(x) dx
  $$

* Intuitively, the expectation of $g(X)$ is the "weight average" of teh values that $g(x)$ can take on for different values of x, where the weights are given by $p_{X}(x)$ or $f_{X}(x)$. 

  * Special case: $\mathbb{E}[X]$ is taken by letting $g(x) = x$. 
    * This is the **mean** of $X$

* **Properties**

  * $\mathbb{E}[a] = a$ for any constant $a \in \mathbb{R}$
  * $\mathbb{E} [af(X)] = a \mathbb{E} [f(X)]$ for any constant $a \in \mathbb{R}$
  * (Linearity of Expectation) $\mathbb{E} [f(X) + g(X)] = \mathbb{E}[f(X)] + \mathbb{E} [g(X)]$
  * For a discrete random variable $X$, $\mathbb{E} [\mathbf{1}{X = k}] = P(X = k)$

#### Variance

* The variance of a random variable $X$ is a measure of how concentrated the distribution of a random variable $X$ is around its mean.
  $$
  \begin{aligned}
  Var[X] &= \mathbb{E} [(X - \mathbb{E}[X])^{2}] \\
  &= \mathbb{E} [X^{2} - 2 \mathbb{E}[X] X + \mathbb{E} [X]^{2}] \\
  &= \mathbb{E}[X^{2}] - 2 \mathbb{E}[X] \mathbb{E}[X] + \mathbb{E}[X]^{2} \\
  &= \mathbb{E}[X^{2}] - \mathbb{E}[X]^{2}
  \end{aligned}
  $$

* 

* **Properties**
  * $Var[a] = 0$ for any constant $a \in \mathbb{R}$ 
  * $Var[af(X)]$ = $a^{2} Var[f(X)]$ for any constant $a \in \mathbb{R}$ 

#### Some Common Random Variables

##### Discrete Random Variables

* $X \sim \text{Bernoulli}(p)$ where $0 \leq p \leq 1$: E.g. the outcome of a coin flip
  $$
  p(x) = 
  \begin{cases}
  p, &\textrm{if x = 1}\\
  1-p, &\textrm{if x = 0}
  \end{cases}
  $$

* $X \sim \textrm{Binomial}(n,p)$ where $0 \leq p \leq 1$: E.g. number of heads in $n$ independent flips
  $$
  p(x) = {n \choose x} p^{x} (1-p)^{n-x}
  $$

* $X \sim \textrm{Geometric}(p)$ where $p > 0$: E.g. the number of flips until the first heads
  $$
  p(x) = p(1-p)^{x-1}
  $$

* $X \sim \textrm{Poisson}(\lambda)$:
  $$
  p(x) = e^{-\lambda} \frac{\lambda^{x}}{x!}
  $$

##### Continuous Random Variables

* $X \sim \textrm{Uniform}(a, b)$ where $a < b$:
  $$
  p(x) = 
  \begin{cases}
  \frac{1}{b-a}, &\textrm{if } a \leq x \leq b \\
  0, &\textrm{otherwise}
  \end{cases}
  $$

* $X \sim \textrm{Exponential}(\lambda)$ 
  $$
  f(x) = 
  \begin{cases}
  \lambda e^{-\lambda x}, &\textrm{if } x \geq 0\\
  0, &otherwise
  \end{cases}
  $$

* $X \sim \textrm{Normal}(\mu, \sigma^{2})$ 
  $$
  f(x) = \frac{1}{\sqrt{2 \pi} \sigma} e^{-\frac{(x - \mu)^{2}}{2 \sigma^{2}}}
  $$
  

## Bayesian Reasoning and Machine Learning

## Graphical Models, Exponential Families, and Variational Inference





