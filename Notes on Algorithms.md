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

## Algorithms by Dasgupta, Papadimitriou, Vazirani

### Chapter 0: Prologue

* An algorithm is a set of procedures that are precise, unambiguous, mechanical, efficient and correct.
* Always as of an algorithm:
  * Is it correct?
  * How much time does it take (as a function of n)?
  * Can we do better?

#### Fibonacci Numbers

$$
F_{n} =
\begin{cases}
F_{n-1} + F_{n-2} &\textrm{if n > 1}\\
1 &\textrm{if n = 1} \\
0 &\textrm{if n = 0}
\end{cases}
$$

* Exponential Algorithm

  ```python
  # function fib1(n)
  if n == 0:
      return 0
  elif n == 1:
      return 1
  else:
      return fib1(n-1) + fib1(n-2)
  ```

  Let T(n) be the number of sets needed to compute fib1(n). Then for $n \leq 1$,  $T(n) \leq 2$. For $n \geq 2$, $T(n) = T(n-1) + T(n-2) + 3$ where 3 operations are from checking the value of n and addition. We wee that $T(n) \geq F_{n}$, which is bad!

  ![RecursiveFibonacci](.\figures\algorithms_exponential_fibonacci.png)

  Notice repetition of the function calls in the above tree.

* Polynomial Fibonacci 

  ```python
  # function fib2(n)
  if n == 0:
      return 0
  f = [] # Could also initial an array of length n
  f.append(0)
  f.append(1)
  for i in range(2,n):
      f.append(f[i-1] + f[i-2])
  return f[-1]
  ```

  This algorithm is linear in n since we go through a length n array once.

  * **NOTE**: It is actually note linear because adding two large numbers is not constant time, but proportional to the length of the numbers, so the actually algorithm is proportional to $n^{2}$. 

* Big-O Notation

  Let $f(n)$ and $g(n)$ be function from positive integers to positive reals. We say $f = O(g)$ (which means that "f grows no faster than g") if there is a constant $c > 0$ such that $f(n) \leq c \cdot g(n)$. 

  * Loose analogy: $f = O(g) \approx f \leq g$ 
  * $f = \Omega(g)$ means $g = O(f)$. ($\approx f \geq g$)
  * $f = \Theta(g)$ means $f = O(g)$ and $f = \Omega(g)$ ($\approx f = g$) 

* **Commonsense Rules**:

  * multiplicative constants can be omitted
    * $14n^{2}$ becomes $n^{2}$
  * $n^{a}$ dominates $n^{b}$ if $a > b$
    * $n^{2}$ dominates $n$
  * Any exponential dominates any polynomial
    * $3^{n}$ dominates $n^{5}$ (it even dominates $2^{n}$)
  * Any polynomial dominates any logarithm
    * $n$ dominates $(\log n)^{3}$ 
    * $n^{2}$ dominates $n \log n$ 