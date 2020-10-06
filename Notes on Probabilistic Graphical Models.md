# Probabilistic Graphical Models

## Change Log?

* 9/30/20: Added chapter one to "Probabilistic Graphical Models: Prinicples and Techniques".
* 10/1/20: Added notes on Probability Review from cs228 stanford notes upto and including sections 2.3.
* 10/4/20: Finished Section 2 of Probability Review from cs228 stanford notes
* 10/5/20: Finished Probability Review from cs228 standford notes

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
* ![BN__vs_MRF](/Users/mopugh/Documents/typora/typora/figures/pgm/bayes_net_vs_mrf_example.png)
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

### Two Random Variables

Would like to know more than one quantity from a random experiment.

#### Joint and Marginal Distributions

Suppose there are two random variables $X$ and $Y$. 

* Can view them them independently. Thus only need $f_{X}(x)$ and $f_{Y}(y)$. 

* If we want to know the values of $X$ and $Y$ from the same random experiment, need a more compliciated structure: the **joint distribution** 

  * Joint cumulative distribution function of $X$ and $Y$
    $$
    F_{XY}(x,y) = P(X \leq x, Y \leq y)
    $$

  * Knowing the joint distribution function allows one to compute any event involving $X$ and $Y$. 

  * Relationship between joint and individual cumulative distribution functions:

    * $F_{X}(x) = \lim_{y \rightarrow \infty} F_{XY}(x,y)$
    * $F_{Y}(y) = \lim_{x \rightarrow \infty} F_{XY}(x,y)$

    These are the **marginal cumulative distributions functions**

* **Properties**:
  * $0 \leq F_{XY}(x,y) \leq 1$
  * $\lim_{x,y \rightarrow \infty} F_{XY}(x,y) = 1$
  * $\lim_{x,y \rightarrow -\infty} F_{XY}(x,y) = 0$
  * $F_{X}(x) = \lim_{y \rightarrow \infty} F_{XY}(x,y)$

#### Joint and Marginal Probability Mass Functions

If $X$ and $Y$ are discrete random variables, the joint probability mass function $p_{XY}: Val(X) \times Val(Y) \rightarrow [0,1]$ is
$$
p_{XY}(x,y) = P(X = x, Y = y)
$$

* $0 \leq P_{XY}(x,y) \leq 1$ for all $x,y$
* $\sum_{x \in Val(x)} \sum_{y \in Val(y)} P_{XY}(x,y) = 1$
* $p_{X}(x) = \sum_{y \in Val(y)} P_{XY}(x,y)$
  * This it the **marginal probability mass funciton** of $X$
  * The process of computing the marginal w.r.t. one variable is called *marginalization*

#### Joint and Marginal Probability Density Functions

Let $X$ and $Y$ be two continuous random variables with joint distribution function $F_{XY}(x,y)$ and if $F_{XY}(x,y)$ is every differentiable w.r.t. $x$ and $y$, then the **joint probability density function** is defined as
$$
f_{XY}(x,y) = \frac{\partial^{2} F_{XY}(x,y)}{\partial x \partial y}
$$

* **Note**: Like in the single variable case, $f_{XY}(x,y) \neq P(X = x, Y = y)$
* $\int \int_{(x,y) \in A} f_{XY}(x,y) dx dy = P((X,Y) \in A)$
* $0 \leq f_{XY}(x,y)$, but $f_{XY}(x,y)$ can be greater than 1
* $\int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f_{XY}(x,y)dx dy = 1$
* $f_{X}(x) = \int_{-\infty}^{\infty} f_{XY}(x,y) dy $
  * This is the **marginal probability density function** or **marginal density** of X. 

### Conditional Distributions

Try to answer the question: what is the probability distribution over $Y$, when we know that $X = x$

* Discrete case:
  $$
  p_{Y \vert X}(y \vert x) = \frac{p_{XY}(x,y)}{p_{X}(x)}
  $$
  if $p_{X}(x) \neq 0$

* Continuous case:

  * Technical point, the probability of any single point is zero in the continuous case

  * Assuming $f_{X}(x) \neq 0$
    $$
    f_{Y \vert X}(y \vert x) = \frac{p_{XY}(x,y)}{p_{X}(x)}
    $$

### Chain Rule

The chain rule for events applies to random variables
$$
\begin{aligned}
&p_{X_{1},\ldots,X_{n}}(x_{1},\ldots,x_{n})\\
&= p_{X_{1}}(x_{1})p_{X_{2} \vert X_{1}}(x_{2} \vert x_{1}) \cdots p_{X_{n} \vert X_{1}, \ldots, X_{n-1}}(x_{n} \vert x_{1}, \ldots, x_{n-1})
\end{aligned}
$$

### Bayes' Rule

Writing a conditional distribution in terms of a different conditional distribution

* Discrete Case
  $$
  p_{Y \vert X}(y \vert x) = \frac{P_{XY}(x,y)}{P_{X}(x)} = \frac{P_{X \vert Y}(x \vert y) P_{Y}(y)}{\sum_{y' \in Val(Y)} P_{X \vert Y}(x \vert y') P_{Y}(y')}
  $$

* Continuous case
  $$
  f_{Y \vert X}(y \vert x) = \frac{f_{XY}(x,y)}{f_{X}(x)} = \frac{f_{X \vert Y}(x \vert y) f_{Y}(y)}{\int_{-\infty}^{\infty} f_{X \vert Y}(x \vert y') f_{Y}(y') dy'}
  $$

* 

### Independence

Two random variables $X$ and $Y$ are independent if $F_{XY}(x,y) = F_{X}(x)F_{Y}(y)$ for all $x$ and $y$. Equivalently:

* For discrete random variables, $p_{XY}(x,y) = p_{X}(x) p_{Y}(y)$ for all $x \in Val(X)$ and $y \in Val(Y)$ 
* For discrete random variables, $p_{Y \vert X}(y \vert x) = p_{Y}(y)$ whenever $p_{X}(x) \neq 0$ for all $y \in Val(Y)$ 
* For continuous random variables, $f_{XY}(x,y) = f_{X}(x) f_{Y}(y)$ for all $x, y \in \mathbb{R}$ 
* For continuous random variables, $f_{Y \vert X}(y \vert x) = f_{Y}(y)$ whenever $f_{X}(x) \neq 0$ for all $y \in \mathbb{R}$ 

The idea is that two random variables are independent if knowing the value of one variable will never effect the conditional distribution of the other variable

* **Lemma**: If $X$ and $Y$ are independent random variables, then for any subsets $A, B \subseteq \mathbb{R}$
  $$
  P(X \in A, Y \in B) = P(X \in A) P(Y \in B)
  $$

* You can use the previous lemma to prove that if $X$ and $Y$ are independent, then any function of $X$ is independent of any function of $Y$

### Expectation and Covariance

* For two discrete random variables $X$ and $Y$ and a function $g: \mathbb{R}^{2} \rightarrow \mathbb{R}$ 
  $$
  \mathbb{E} [g(X,Y)] = \sum_{x \in Val(X)} \sum_{y \in Val(Y)} g(x,y) p_{XY}(x,y)
  $$

* For two continuous random variables
  $$
  \mathbb{E}[g(X,Y)] = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty}g(x,y)f_{XY}(x,y)dxdy
  $$

* Can use expectations to study the relationship between random variables
  $$
  \begin{aligned}
  Cov[X,Y] &= \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])] \\
  &= \mathbb{E}[XY - X \mathbb{E}[Y] - Y \mathbb{E}[X] + \mathbb{E}[X] \mathbb{E}[Y]] \\
  &= \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y] - \mathbb{E}[Y]\mathbb{E}[X] + \mathbb{E}[X]\mathbb{E}[Y] \\ 
  &= \mathbb{E}[XY] - \mathbb{E}[X] \mathbb{E}[Y]
  \end{aligned}
  $$

  * Note: $\mathbb{E}[X]$ and $\mathbb{E}[Y]$ are constants and can be pulled out of the expectation
  * If $Cov[X,Y] = 0$, the random variables are said to be uncorrelated

* **Properties**

  * (Linearity of expectation) $\mathbb{E}[f(X,Y) + g(X,Y)] = \mathbb{E}[f(X,Y)] + \mathbb{E}[g(X,Y)]$ 
  * $Var[X + Y] = Var[X] + Var[Y] + 2Cov[X,Y]$
  * If $X$ and $Y$ are independent, then $Cov[X,Y] = 0$
  * If $X$ and $Y$ are independent, then $\mathbb{E}[f(X)g(Y)] = \mathbb{E}[f(X)] \mathbb{E}[g(Y)]$ 

### Real-World Applications

#### Images

Suppose $p(\mathbf{x})$ is a distribution over images, where $\mathbf{x}$ is an image represented by a vector of pixels. The distribution assigns high probability to images that look realistic and a low probability to everything else

* generate images by sampling the model
  * the model has far fewer parameters than the amount of data, so the "distilled the essence"
* In-Painting: with a model of $p(\mathbf{x})$ and a patch of an existing image, we can sample $p(\textrm{image} \vert \textrm{patch})$ 
* Image Denoising: use sampling or exact inference on $p(\textrm{original image} \vert \textrm{noisy image})$ 

#### Language Models

Want to construct a probability distribution $p(\mathbf{x})$ over sequences of words or characters that assign high probability to (English) sentences. The distribution can be learned from a variety of sources (e.g. Wikipedia)

* Generation: sample from learned $p(\mathbf{x})$ 
* Translation: build model $p(y \vert x)$ to generate an English sentence $y$ from a Chinese sentence $x$

#### Audio Models

Construct a probability distribution $p(\mathbf{x})$ over audio signals that assigns high probability to ones that sound like human speech.

* Upsampling or Super-Resolution: sample or inference on $p(\mathbf{I} \vert \mathbf{O})$ where $\mathbf{O}$ is the original signal and $\mathbf{I}$ is the intermediate signal
* Speech Synthesis: sample from model to generate speech
* Speech Recognition: Given joint model of speech and language (text), can attempt to infer spoken words from audio signals

#### Applications in Science

* Error Correcting Codes
* Computational Biology
* Ecology
* Economics

#### Applications in Health Care and Medicine

* Medical Diagnosis

### Bayesian Networks

**Representation**: How do we choose a probability distribution to model some interesting aspect of the world?

* Want to learn effective and general technique for parameterizing probability distributions using only a few parameters
* Leverage **directed acyclic graphs** (DAGs)
* Connections between the structure of a DAG and the modeling assumptions of the distribution that it describes. 

The use of DAGs leads to **Bayesian network** models, i.e. a directed graphic model is a Bayesian network

* Bayesian networks show causality
* **Markov random fields** (MRFs) use undirected graphs and cannot show causality

#### Probabilistic Modeling with Bayesian Networks

**Idea**: Recall from the chain rule
$$
p(x_{1},\ldots,x_{n}) = p(x_{1})p(x_{2} \vert x_{1}) \cdots p(x_{n} \vert x_{n-1},\ldots,x_{1})
$$
A Bayesian network is a distribution in which each factor on the right hand side depends only on a small number of *ancestor variables* $x_{A_{i}}$ 
$$
p(x_{i} \vert x_{i-1} \ldots x_{1}) = p(x_{i} \vert x_{A_{i}})
$$

* Example: approximate the factor $p(x_{5} \vert x_{4}, x_{3}, x_{2}, x_{1})$ with $p(x_{5} \vert x_{4}, x_{3})$ and $x_{A_{5}} = \{ x_{4}, x_{3} \}$ 

When the random variables are discrete, the factors $p(x_{i} \vert x_{A_{i}})$ can be viewed as *probability tables* where each row is an assignment of $x_{A_{i}}$ and each column a value of $x_{i}$. If each variable takes $d$ values and has at most $k$ ancestors, then

* The table contains at most $O(d^{k+1})$ entries
* One table per variable, so the entire distribution can be described in $O(nd^{k+1})$ parameters
  * This is compared to $O(d^{n})$ for the naive approach

##### Graphical Representation

Can represent distribution using a directed acyclic graph.

* Vertices are the variables $x_{i}$
* Edges indidcate dependecy relationships
  * The parents of $x_{i}$ are its ancestors $x_{A_{i}}$ 

* Example:

 <img src="/Users/mopugh/Documents/typora/typora/figures/pgm/letter_bayes_net_example.png" alt="letter_bayes_net_example" style="zoom:50%;" />
$$
p(l,g,i,d,s) = p(l \vert g)p(g \vert d, i) p(i) p(d) p(s \vert i)
$$

##### Formal Definition

A Bayesian network is a directed graph $G = (V, E)$ together with

* A random variable $x_{i}$ for each node $i \in V$ 
* One conditional probability distribution $p(x_{i} \vert x_{A_{i}})$ per node specifying the probability of $x_{i}$ conditioned on its parents' values.

A Bayesian network defines a probability distribution on $p$. Conversely a probability distribution $p$ *factorizes* over a DAG $G$ if it can be decomposed into a product of factors as specified by $G$. 

* Can show that a probability distribution $p$ represented by a Bayesian network is valid
* Can show that if $G$ contains cycles, the associated probability may not sum to one

#### The Dependencies of a Bayes Net

Bayesian networks represent probability distribution that can be formed by the products of smaller, local probability distributions (one for each variable). This introduces independence assumptions

* **Question**: Which independence assumptions are being made by using a Bayesian network model with a given structure described by $G$? 
  * We should know precisely what model assumptions are being made
    * Are they correct?

Let $I(p)$ be the set of all independencies that hold for a joint distribution $p$

* Example: if $p(x,y) = p(x)p(y)$, then $x \perp y \in I(p)$ 

##### Independencies Described by Directed Graphs

Bayesian networks described independencies in $I(p)$ by three structures. Consider a Bayesian network $G$ with three nodes: $A, B, C$. There are only three possible structures, which lead to the three different independence assumptions.

* **Common Parent**: If $G$ is of the form $A \leftarrow B \rightarrow C$
  * $B$ observed, then $A \perp C \vert B$. 
  * B unobserved, then $A \not \perp C$ 
  * **Idea**: B contains all the information that determines that outcomes of $A$ and $C$. Once $B$ is observed, there is nothing else that affects the outcomes of $A$ and $C$.
* **Cascade**: If $G$ equals $A \rightarrow B \rightarrow C$
  * $B$ is observed, then $A \perp C \vert B$.
  * B unobserved, then $A \not \perp C$
  * **Idea**: B holds all the information that determines the outcome of C, thus it does not matter what value $A$ takes
* **V-Structure** (Explaining Away): If $G$ is $A \rightarrow C \leftarrow B$, then knowing $C$ couples $A$ and $B$. 
  * $A \perp B$ if C is unobserved
  * $A \not \perp B \vert C$ if C is observed
  * Example: Suppose that $C$ is a Boolean variable that indicates whether our lawn is wet one morning; $A$ and$B$ are two explanations for it being wet: either it rained (indicated by $A$), or the sprinkler turned on (indicated by $B$). If we know that the grass is wet ($C$ is true) and the sprinkler didn’t go on ($B$ is false), then the probability that $A$ is true must be one, because that is the only other possible explanation. Hence, $A$ and$B$ are not independent given $C$.

![Bayesian networks over three variables, encoding different types of dependencies: cascade (a,b), common parent (c), and v-structure (d).](./figures/pgm/3node-bayesnets.png)

* Bayesian networks over three variables, encoding different types of dependencies: cascade (a,b), common parent (c), and v-structure (d).

Can extend the above structures to larger graphs recursively. Let $Q, W, O$ be three sets of nodes in a Bayesian network $G$. $Q$ and $W$ are **d-separated** given $O$ (i.e. the variables $O$ are observed) if $Q$ and $W$ are not connected by an **active path**. An undirected path in G is called **active** given observed variables $O$ if for every consecutive triple of variables $X,Y, Z$ on the path, one of the following holds:

* $X \leftarrow Y \leftarrow X$ and $Y$ is unobserved $Y \notin O$
* $X \rightarrow Y \rightarrow Z$ and $Y$ is unobserved $Y \notin O$
* $X \leftarrow Y \rightarrow Z$ and Y is unobserved $Y \notin O$
* $X \rightarrow Y \leftarrow Z$ and Y or any of its descendants are observed

![In this example, $$X_1$$ and $$X_6$$ are $$d$$-separated given $$X_2, X_3$$.](./figures/pgm/dsep2.png)

* In this example, $$X_1$$ and $$X_6$$ are $$d$$-separated given $$X_2, X_3$$.

![However, $$X_2, X_3$$ are not $$d$$-separated given $$X_1, X_6$$. There is an active pass which passed through the V-structure created when $$X_6$$ is observed.](./figures/pgm/dsep1.png)

* However, $$X_2, X_3$$ are not $$d$$-separated given $$X_1, X_6$$. There is an active pass which passed through the V-structure created when $$X_6$$ is observed

## Bayesian Reasoning and Machine Learning

## Graphical Models, Exponential Families, and Variational Inference





