# Open Data Structures

## Chapter 1: Introduction

### Interfaces

* Understand difference between a data structure's interface and its implementation
  * interface: describes what a data structure does
  * implementation: describes how a data structure does it
* **Interface** (a.k.a. **abstract data type**) defines the set of operations supported by a data structure and the semantics (meaning) of those operations
  * Does not tell us how the data structure implements those operations.
* **implementation**: includes the internal representation of the data structure as well as definitions of the algorithms that implement the operations

#### The Queue, Stack, and Deque Interfaces

Queue: add elements and remove the next element

* `add(x)`: add the next value `x` to the Queue

* `remove()`: remove the next (previously added) value `y` from the Queue and return `y` 

* The Queue's **queueing discipline** decides which element should be removed.
  
  * FIFO (first-in-first-out): removes items in the same order they were added. 
    
    * `add(x)` sometimes called `enqueue(x)`
    
    * `remove()` sometimes called `dequeue()`
      
      ![image-20201117220503254](figures/image-20201117220503254.png)
  
  * LIFO (last-in-first-out): most recently added element is removed.
    
    * A stack
    
    * `add(x)` = `push(x)`
    
    * `remove()` = `pop()` 
      
      ![image-20201117220816296](figures/image-20201117220816296.png) 
  
  * priority: removes the smallest element, breaking ties arbitrarily. 
    
    * e.g. emergency room treats patients with the most serious conditions
    
    * `remove()` also called `delete_min()`
      
      ![image-20201117220641643](figures/image-20201117220641643.png)

A Deque is a generalization of both the FIFO and LIFO Queues. 

* A sequence of elements with a front and a back
  * can add and remove from either end
    * `add_first(x)`
    * `remove_first()`
    * `add_last(x)`
    * `remove_last()`
  * Stack can be implemented with `add_first(x)` and `remove_first()`
  * FIFO Queue can be implemented with `add_last(x)` and `remove_first()` 

#### The List Interface: Linear Sequences

List interface includes:

* `size()`: return `n`, the length of the list
* `get(i)`: return the value $x_{i}$ 
* `set(i, x)`: set the value of $x_{i}$ to `x` 
* `add(i, x)`: add `x` at position `i` displacing $x_{i}, \ldots, x_{n-1}$. Set $x_{j+1} = x_{j}$ for all $j \in \{n-1, \ldots, i\}$, increment `n` and set $x_{i} = x$. 
* `remove(i)`: remove the value $x_{i}$, displacing $x_{i+1}, \ldots, x_{n-1}$; Set $x_{j} = x_{j+1}$ for all $j \in \{i, \ldots, n-2\}$ and decrement n   

These operations are sufficient to implement the Deque interface:

* `add_first(x)` $\Rightarrow$ `add(0,x)`
* `remove_first()` $ \Rightarrow$ `remove(0)`
* `add_last(x)` $\Rightarrow$ `add(size(), x)`
* `remove_last()` $\Rightarrow$ `remove(size()-1)` 

#### The USet Interface: Unordered Sets

USet interface represents an unordered set of unique elements like a mathematical set.

* all elements are distinct

* no specific order

USet supports the following operations:

* `size()`: return the number `n` of elements in the set

* `add(x)`: add the element `x` to the set if not already present
  
  * Add `x` to the set provided that there is no element `y` in the set such that `x` equals `y`. Return `true` if `x` was added to the set and `false` otherwise.

* `remove(x)`: remove `x` from the set
  
  * find an element `y` in the set such that `x` equals `y` and remove `y`. Return `y` or `nil` if no such element exists

* `find(x)`: find `x` in the set if it exists
  
  * Find an element `y` in the set such that `y` equals `x`. Return `y`, or `nil` if no such element exists.

Note that `x` and `y` can be distinct objects but can be treated as equal. 

* Useful to create **dictionaries** or **maps**

Dictionary/map form compound objects called Pairs, wach with a **key** and a **value**. Two Pairs are treated as equal if their keys are equal.

#### The SSet Interface: Sorted Sets

An SSet stores elements from some total order, so that any two elements x and y can be compared. 

$$
\textrm{compare}(x,y) = \begin{cases}
< 0 \textrm{ if } x < y \\
> = \textrm{ if } x > y \\
= 0 \textrm{ if } x = y 
\end{cases}
$$

SSet supports `size()`, `add(x)`, `remove(x)` methods with exactly the same semantics as USet interface. Only difference is with `find(x)`:

* `find(x)`: locate `x` in the sorted set
  
  * Find the smallest element y in the set such that $y \geq x$. Return y or nil if no such element exists.
  
  * sometimes referred to as **successor search** 
  
  * this is meaningful even if there is no element equal to x in the set
  
  * This usually has a price in terms of running time and complexity compared to USet `find(x)` 

### Mathematical Background

#### Exponentials and Logarithms

* The **base-b logarithm of k**, denoted $log_{b} k$ is the unique value $x$ satisfying $b^{x} = k$. 

* Informally think of logarithm as the number of times to divide k by b before the result is less than or equal to 1. 

* Denote the **natural logarithm** $log_{e} k$ as $\ln k = \int_{1}^{k} \frac{1}{x} dx$ 

* Common manipulations:
  
  * $$
    b^{\log_{b} k} = k
    $$
  
  * Change of base:
    
    $$
    \log_{b} k = \frac{\log_{a} k}{\log_{a} b}
    $$

#### Factorials

* $n!$ counts the number of distinct permutaitons (i.e. orderings) of n elements. 
  
  * **Sterling's Approximation**:
    
    $$
    n! = \sqrt{2 \pi n} \left ( \frac{n}{e} \right )^{n} e^{\alpha(n)}
    $$
    
    where $\frac{1}{12n + 1} < \alpha(n) < \frac{1}{12n}$

* Sterling's Approximation also approximates $\ln(n!)$
  
  $$
  \ln(n!) = n \ln n - n + \frac{1}{2} \ln (2 \pi n) + \alpha(n)
  $$
  
  * Prove Sterling's Approximation by approximating $\ln(n!) = \ln 1 + \ln 2 + \cdots + \ln n$ by the integral $\int_{1}^{n} \ln n dn = n \ln n - n + 1$

* The **Binomial Coefficient** counts the number of subsets of an n element set that have size k, i.e. the number of ways of choosing k distinct integers from the set $\{ 1, \ldots, n \}$:
  
  $$
  n \choose k = \frac{n!}{k! (n-k)!}
  $$
  
  
