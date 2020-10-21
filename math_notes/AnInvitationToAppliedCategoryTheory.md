# Seven Sketches In Compositionality: An Invitation To Applied Category Theory

## Chapter 1: Generative Effects: Orders and Galois Connections

Compositionality*: cases in which systems or relationships can be combined to form new systems or relationships.

### More than the sum of their parts

* Observation is inherently "lossy"
  * To extract information from something requires dropping the details. 
* Central theme to category theory: study of structure and structure-preserving maps
  * A map $f: X \rightarrow Y$ is a kind of observation of $X$ via a specified relationship it has with another object $Y$. 
  * Asking what aspects of $X$ one wants to preserve under the observation $f$ becomes the question: "what category are you working in?"
    * E.g. Out of all functions $f: \mathbb{R} \rightarrow \mathbb{R}$, only some preserve order, preserve distance, etc.
* **Definition**: A function $f: \mathbb{R} \rightarrow \mathbb{R}$ is said to be:
  * *order-preserving*: if $x \leq y$ implies $f(x) \leq f(y)$ for all $x, y \in \mathbb{R}$ 
  * *metric-preserving*: if $\vert x - y \vert = \vert f(x) - f(y) \vert$
  * *addition-preserving*: if $f(x+y) = f(x) + f(y)$ 
* In category theory, want to keep control over which aspects are being preseved
  * The less structure that is being preserved, the more "surprises" can occur
    * These surprises are *generative effects* 

#### A first look at generative effects

