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

### Chapter 1: Algorithms with Numbers

#### Basic Arithmetic

##### Addition

* **Fact**: The sum of any three single-digit numbers is at most two digits long

  * Gives algorithm for adding in any base:
    * Align two numbers by their right-hand ends and perofrm a single right-to-left pass in which the sum is computed digit by digit and maintaining the overflow as a carry. By the above fact, *the carry is always a single digit*.
  * **Question**: Given two binary numbers x and y, how long does the algorithm take to add them?
    * Want as a function of the size of the input
      * The number of bits of x and y
    * Suppose x and y are $n$ bits long, then the sum is at most $n+1$ bits and each individual bit gets computed in a fixed amount of time, thus the running time is $O(n)$. 
  * **Question**: Is there a faster algorithm?
    * In order to add two $n$-bit numbers, one must at least read in the bits and write down the answer which requires $n$ operaitons, so upto multiplicative constants, it is optimal.
  * **Question**: Doesn't a computer add numbers for free?
    * Yes, if the length of the numbers is within the word length of the hardware, but for long numbers, it is very much like performing operations bit by bit.

* Change of base:
  $$
  \log_{b}N = \frac{\log_{a}N}{\log_{a}b}
  $$
  * Note: $\log_{a}b$ is a constant

* Properties of $\log_{2}N$
  * The power to raise 2 by to get $N$
  * The number of times you halve $N$ to get down to 1 (more precisely $\lceil \log N \rceil $ )
  * The number of bits in the binary representation of $N$ (more precisely $\lceil \log (N+1) \rceil$)
  * The depth of a complete binary tree with $N$ nodes (more precisely $\lfloor \log N \rfloor$)
  * It is the sum $1 + \frac{1}{2} + \frac{1}{3} + \cdots + \frac{1}{N}$ to within constant factor

##### Multiplication and Division

* Grade-school multiplication algorithm (lattice multiplication)![binary_lattice_multiplication](/Users/mopugh/Documents/typora/typora/figures/algorithms_data_structures/binary_lattice_multiplication.png)
  * Note: left shifting is like multiplying by the base (here 2)
  * Note: right shifting is like dividing by the base, rounding down if needed
  * How long does it take?
    * If x and y are $n$ bits, there are $n$ intermediate rows each of length up to $2n$ bits. Thus, $O(n^{2})$ (essentially n additions of length n)

* Peasant Multiplication 

  ```python
  """
  Function: multiply(x, y)
  Input: two n-bit integers x and y, where y >= 0
  Output: the product of x and y
  """
  
  if y == 0:
    return 0
  z = multiply(x, floor(y/2))
  if y is even:
    return 2 * z
  else:
    return x + 2 * z
  ```

  * In binary, this is equivalent to the above grade-school algorithm in binary. 

  * $$
    x \cdot y = 
    \begin{cases}
    2(x \cdot \lfloor y / 2 \rfloor) &\textrm{if y is even} \\
    x + 2(x \cdot \lfloor y/ 2 \rfloor) &\textrm{if y is odd}
    \end{cases}
    $$

  * *How long does it take*?

    * There are $n$ recursive calls since $y$ is halved at each call (i.e. the number of bits is decreased by 1)
    * Each call requires
      * a division by 2 (right shift)
      * a pairty check (looking at the last bit)
      * a multiplication (left shift)
      * possibly one addition
      * A total of $O(n)$ bit operations
    * Total time: $O(n^{2})$ 

* Division:

  Divide an integer $x$ by an integer $y$: need to find a quotient $q$ and a remainder $r$ such that $x = yq + r$ and $r < y$

  ```python
  """
  Function: divide(x,y)
  Input: Two n-bit integers x and y, y >= 1
  Output: The quotient and remainder of x divided by y
  """
  if x == 0:
    return (q, r) = (0,0)
  (q, r) = divide(floor(x / 2), y)
  q = 2 * q, r = 2 * r
  if x is odd: 
    r = r + 1
  if r >= y:
    r = r - y
    q = q + 1
  return (q, r)
  ```

  #### Modular Arithmetic

  

