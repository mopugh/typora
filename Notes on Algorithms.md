# Notes on Algorithms

## Algorithms by Jeff Erickson

### Chapter 0: Introduction

- What is an algorithm?
  - An explicit, precise, unambiguous, mechanically-executable sequence of elementary instructions, usually intended to accomplish a specific task.

* **Lattice Multiplication**

  * Let input be a pair of arrays of digits $X[0\ldots m-1]$, $Y[0\ldots n-1]$ and the output be z.

  $$
  x = \sum_{i=0}^{m-1} X[i] 10^{i} \\
  y = \sum_{j=0}^{n-1} Y[j] 10^{j} \\
  z = x \cdot y = \sum_{i=0}^{m-1} \sum_{j = 0}^{n-1} X[i]Y[j]10^{i+j}
  $$

  * Runs in $O(mn)$ time.

* **Peasant Multiplication**

  * The four operations for peasant multiplication are:

    * parity
    * addition
    * duplication (doubling)
    * mediation (halving)

  * Algorithm based on the following recursive identity:
    $$
    x \cdot y =
    \begin{cases}
    0 &\textrm{if x = 0}\\
    \left \lfloor{x/2} \right \rfloor \cdot (y+y) &\textrm{if x is even} \\
    \left \lfloor{x/2} \right \rfloor \cdot (y+y) + y &\textrm{if x is odd}
    \end{cases}
    $$

    ```{r, eval=True}
    prod <- 0
    while x > 0
    	if x is odd
    		prod <- prod + y
    	x <- floor(x/2)
    	y <- y + y
    return prod
    ```

  * Runs in $O(mn)$ time.

## Algorithms Illuminated Part 1: The Basics

