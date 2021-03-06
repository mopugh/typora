# Notes on Probability

## Change Log

* 9/30/20: First commit. Started chapter 1 of *Introduction to Probability 2nd Edition*. Worked through and including Section 1.2.

## Introduction to Probability 2nd Edition by Blitzstein and Hwang

### Chapter 1: Probability and Counting

#### Sample Spaces

![PebbleWorld](./figures/probability/pebble_world.png)

* **Definition**: The *sample space* $\mathcal{S}$ of an experiment is the set of all possible outcomes of the experiment
* **Definition**: An *event* is a subset of the sample space $\mathcal{S}$
* The sample space can be finite, countably infinite, or uncountably infinite
* *De Morgan's Laws*:
  * $(A \cup B)^{c} = A^{c} \cap B^{c}$
    * Saying that it is *not* the case that at least one of $A$ and $B$ occur is the same as saying that $A$ does not occur and $B$ does not occur.
  * $(A \cap B)^{c} = A^{c} \cup B^{c}$
    * Saying that it is *not* the case that both occur is the same as saying that at least one does not occur.
* **Example**: Coin flipping Let $A_{1} = {(1,s_{2}, \ldots, s_{10}):s_{j} \in \{0,1\} \text{ for } 2 \leq j \leq 10}$ be the event that the first coin flip is heads. Likewise, let $A_{j}$ be the event that the $j^{th}$ flip is heads for $j = 2,3,\ldots,10$.
  * Let $B$ be the event that at least one flip was heads: $B = \cup_{j=1}^{10} A_{j}$ 
  * Let $C$ be the event that all flips were heads: $C = \cap_{j=1}^{10}A_{j}$
  * Let $D$ be the event that there were at least two consecutive heads: $D = \cup_{j=1}^{9} (A_{j} \cap A_{j+1})$

![SetTerminology](./figures/probability/set_terminology.png)

#### Naive Definition of Probability

* **Definition** (Naive definition of probability): Let $A$ be the event for an experiment with a finite sample space $S$. The *naive probability* of $A$ is
  $$
  P_{naive}(A) = \frac{\vert A \vert}{\vert S \vert} = \frac{\textrm{number of outcomes favorable to }A}{\textrm{total number of outcomes in }S}
  $$

* Note:
  $$
  P_{naive}(A^{c}) = \frac{\vert A^{c} \vert}{\vert S \vert} = \frac{\vert S \vert - \vert A \vert}{\vert S \vert} = 1 - \frac{\vert A \vert}{\vert S \vert} = 1 - P_{naive}(A)
  $$

* *Strategy*: Is it easier to find the probability of an event or of the probability of the complement of the event?

* **Restrictions**: The naive definition of probability assumes:

  * $\vert S \vert$ is finite
  * Each outcome has the same probability

* Naive probability is applicable:

  * symmetry, e.g. symmetry of a coin, cards in a deck
  * by design, e.g. surveys
  * null model

#### How to Count

Calculating naive probabilities requires computing $\vert A \vert$ and $\vert S \vert$. 

##### Multiplication Rule

* Theorem (Multiplication rule): Consider a compound experiment consisting of two sub-experiments, Experiment A and Experiment B. Suppose that A has $a$ possible outcomes and for each of those outcomes Experiment B has $b$ possible outcomes. Then the compound experiment has $ab$ outcomes. 
  * View as a tree: $a$ possible options, each with $b$ follow-up options leading to $b+\cdots+b = ab$ options
  * Often easier to think of experiments as being chronological, but there is not requirement that A be performed before B. 

<img src="figures/Screen Shot 2020-10-08 at 6.11.34 AM.png" alt="Screen Shot 2020-10-08 at 6.11.34 AM" style="zoom:50%;" />

<img src="../../../Desktop/Screen Shot 2020-10-08 at 6.23.14 AM.png" alt="Screen Shot 2020-10-08 at 6.23.14 AM" style="zoom:50%;" />

* Order of flavor and cone doesn't matter
* Doesn't matter if same flavors are available on each cone
  * Just need same number of options
* **Question**: What if want two ice cream cones, but order doesn't matter, i.e. (cakeC, waffleV) is the same as (waffleV, cakeC)?
  * **NOT** $\frac{6\cdot6}{2}=18$ because of examples like (cakeC, cakeC)
  * There are $6\cdot5$ pairs of the form $(x,y)$ where $x \neq y$, so $\frac{6 \cdot 5}{2} = 15$ pairs not considering ordering
  * There are 6 pairs of the form $(x,x)$ for each cone/flavor pair
  * Thus there are a total of $15 + 6 = 21$ optoins
* **Example**: A set with n elements has $2^{n}$ subsets since for each element, you can choose whether to include it or not.
* **Theorem** (Samplng with Replacement): Consider n objects and making k choices from them, one at a time with *replacement* (i.e. choosing a certain object does not preclude it from being chosen again). Then there are $n^{k}$ possible outcomes (where order matters, i.e. choosing object x and then object y is counted as different than choosing object y and then object x)
* **Theorem** (Sampling without Replacement): Consider n objects and making k choices from them, one at a time without replacement (i.e. choosing a certain object precludes it from being chosen again). Then there are $n(n-1)\cdots(n-k+1)$ possible outcomes from $1 \leq k \leq n$, and 0 possibilities if $k > n$ (where order matters). By convention, $n(n-1)\cdots(n-k+1) = n$ for k = 1.
  * Follows from multiplication rule: each sub-experiment has 1 few option.
  * Need $k \leq n$, where as in sampling with replacement, the objects are inexhaustible. 
* **Permutations**: The number of permutations of $n$ objects follows from sampling without replacement: $n!$ 

The sampling theorems are about counting, but when the naive definition of probability applies, we can use them to compute probabilities. 

* **Example** (Birthday Paradox): If there are $k$ people, what's the probability that at least two people share a birthday? 
  * There are $365^k$ ways to assign birthdays
  * Compute the probability that no one shares a birthday - much easier than considering all possibilities of sharing a birthday
    * Assign birthdays to $k$ people such that no one shares a birthday: sampling without replacement
    * $365 \cdot 364 \cdots (365-k+1)$ for $k \leq 365$ 
    * $P(\textrm{no birthday match}) = \frac{365 \cdot 364 \cdots (365 - k + 1)}{365^k}$ 
  * $P(\textrm{at least 1 birthday match}) = 1 - P(\textrm{no birthday match}) = 1 - \frac{365 \cdots (365 - k + 1)}{365^k}$
  * When $k = 366$, guaranteed to have a match
  * Note: the number of pairs, each of which is equally likely to share a birthday, grows as $k \choose 2$ 

* **WARNING**: It is important to think of each object in a population as having a unique identity when sampling, even if the objects are indistinguishable. 
  * Leibniz's Mistake: What is more likely, rolling two dice that sum to 11 or 12? 
    * 11 is more likely since you can get it via (5,6) or (6,5) where as 12 requires (6,6)
    * Leibniz argued they are equally probable since 11 and 12 can only be attained in 1 way
      * Leibniz viewed (5,6) as the same as (6,5), i.e. thought the dice were indistinguishable 
      * Should have labeled the die!

##### Adjusting for Overcounting

* Difficult to count each possibility once and only once

* Count each possibility $c$ times and then divide by $c$

  * **Adjusting for overcounting**

* **Example**: Consider 4 people. 

  * How many ways are there to choose a two person committee? 
    * $4 \cdot 3$, but this double counts since order does not matter, so $4 \cdot 3 / 2 = 6$ 
  * How many ways to break the people into two teams of two?
    * Specify person 1's teammate completely specifies both teams: 3 ways to do this
    * See solution to previous problem and realize it double counts since order of teams does not matter: $6 / 2 = 3$ 

* **Binomial Coefficient**: For any nonnegative integers $k$ and $n$, the *binomial coefficient* $n \choose k$ is the number of subsets of size $k$ for a set of size $n$. 

  * Sets are unordered, so considering without replacement and without distinguishing between different orders.

* **Theorem** (Binomial Coefficient Formula): For $k \leq n$
  $$
  {n \choose k} = \frac{n(n-1) \cdots (n-k+1)}{k!} = \frac{n!}{(n-k)!k!}
  $$
  For $k > n, {n \choose k} = 0$

  * **Proof**: Let $A$ be a set with $\vert A \vert = n$. Any subset of $A$ has at most n elements, so ${ n \choose k} = 0$ for $k > n$. Now let $k \leq n$. By sampling without replacement, there are $n (n-1) \cdots (n -k + 1)$ ways to make an ordered selection of $k$ elements without replacement. This overcoats each subset by a factor of $k!$ since order does not matter. Thus, divide by $k!$. 

* **Warning**: Do not compute binomial coefficient using factorials (numbers get huge quickly)! Use the cancellation formula. 

* **Example**: How many ways are there to permute the letters in 'LALALAAA'? 

  * Determine where the 5 A's go, or equivalently the 3 L's
  * ${8 \choose 5} = {8 \choose 3} = \frac{8 \cdot 7 \cdot 6}{3!} = 56$

* **Example**: How many ways are there to permute the letters in 'STATISTICS'?

  * Could choose where to the the S's, then the T's, etc. 
  * Note there are $10!$ ways to reorder the letters, but this overcounts since we can't differentiate the same letters (i.e multiple S's), so divide this by $3!3!2!$ to account for the 3 S's, 3 T's and 2 I's. 
  * ${10 \choose 3} {7 \choose 3} {4 \choose 2} {2 \choose 1} = \frac{10!}{3!3!2!} = 50400$

* **Example**: Binomial Theorem:
  $$
  (x+y)^{n} = \sum_{k=0}^{n} {n \choose k} x^{k} y^{n-k} 
  $$

  * $(x+y)^{n} = (x+y) \cdot (x+y) \cdots (x+y)$ n times. How many ways are there to choose x k times from the (x+y) terms? $n \choose k$

STOPPED AT EXAMPLE 1.4.20 on page 17 (28 of pdf)

---

