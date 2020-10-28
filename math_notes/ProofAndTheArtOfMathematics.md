# Proof and the Art of Mathematics

## Preface

* A **proof** is any sufficiently detailed convincing mathematical argument that logically establishes the conclusion of a theorem from its premises. 

### A Note to the Student

* Suggest using the *theorem-proof* format

  **Theorem**: *A clear and precise statement of the mathematical claim.* 

  **Proof**: A logically correct, clear, and precise argument that establishes the truth of the claim made in the theorem statement. $\square$ 

* Also include proofs for **lemmas** and **corollaries**. 

  * Lemma: A small theorem-like result, which will be proved separately and then used in other proofs.
  * Corollary: A theorem-like result that follows easily from a previously proved theorem or from details arising in a previous proof.

* Also write **definitions**

  * Definition: A clear and precise statement giving the official meaning of a mathematical term.

* Example

  **Exercise**: Prove that *every* hibdab is hobnob

  Convert this to a precise theorem statement

  **Theorem**: Every hibdad is hobnob.

  **Proof**: And so on with the arguments. $\square$ 

> **Use the theorem-proof format.** In all your mathematical exercises, write in the theorem-proof style. State a clear claim in your theorem statement. State lemmas, corollaries, and definitions as appropriate. Give a separate, clearly demarcated proof for every formally stated mathematical claim.

## Chapter 1: A Classical Beginning

### The number $\sqrt{2}$ is irrational

**Theorem 1**: The number $\sqrt{2}$ is irrational

This is saying that $\sqrt{2}$ cannot be expressed as a fraction.

**Proof**: Suppose toward contradiction that $\sqrt{2}$ is rational. This means that $\sqrt{2} = p / q$ for some integers $p$ and $q$, with $q \neq 0$. We may assume that $p/q$ is in lowest terms, which means that $p$ and $q$ have no nontrivial common factors. By squaring both sides, we conclude that $2 = p^2/ q^2$ and consequently $2q^2 = p^2$. This tells us that $p^2$ must be an even number, which implies that $p$ is also even, since the square of an odd number is odd. So $p = 2k$ for some integer $k$. From this, it follows that $2q^2 = p^2 = (2k)^2 = 4k^2$, and therefore $q^2 = 2k^2$. So $q^2$ is also even, and therefore $q$ must be even. Thus, both $p$ and $q$ are even, which contradicts our assumption that $p/q$ was in lowest terms. So $\sqrt{2}$ cannot be rational. $\square$

This proof proceeded by *contradiction*. In such a proof, one assumes ("toward contradiction") that the desired conclusion is not true, aiming to derive from this assumption a contradiction. Since that consequence cannot be the true state of affairs, the assumption leading to it must have been wrong, and so the derived conclusion must have been true. 

The number $\sqrt{2}$ is *algebraic*: it is the root of a nontrivial integer polynomial: $x^2 - 2 = 0$. *Transcendental* real numbers are non-algebraic numbers, i.e. cannot be expressed as the root of a nontrivial integer polynomial. The complex number $i = \sqrt{-1}$ is a complex algebraic number since it is the root of $x^2 + 1 = 0$. 

#### Revisit proof that $\sqrt{2}$ is irrational

The previous proof that $\sqrt{2}$ is irrational depended on $p/q$ being in lowest terms. We can get around that requirement as follows:

**Slightly modified proof of theorem 1**: Suppose toward contradiction that $\sqrt{2}$ is rational. So $\sqrt{2} = p/q$ for some integers $p$ and $q$, and we may assume that the numerator p is chosen as small as possible for such a representation. It follows as before that $2q^2 = p^2$, and so $p^2$ and hence also $p$ is even. So $p = 2k$, for some $k$, which imples that $q^2 = 2k^2$ as before, so $q^2$, and hence also $q$ is even. So $q = 24$ for some r, and consequently $\sqrt{2} = p / q = (2k)/(2r) = k/r$. We have therefore found a rational representation of $\sqrt{2}$ using a smaller numerator, contradicting our earlier assumption. So $\sqrt{2}$ is not rational. $\square$ 

### Lowest Terms

A fraction $p/q$ is in lowest terms if $p$ and $q$ are *relatively prime*, i.e. that they have no common factor, a number $k > 1$ that divides both. 

* $3/6 = 1/2$ . Notice that two representation mean the same thing. Distinguish between the value of a number and the expression of the number. Distinguish between the description of a number and the number itself. 

**Lemma**: Every fraction can be put in lowest terms. 

**Proof**: Consider any fraction $p/q$, where $p$ and $q$ are integers and $q \neq 0$. Let $p'$ be the smallest nonnegative integer for which there is an integer $q'$ with $\frac{p}{q} = \frac{p'}{q'}$. $p'$ and $q'$ are relatively prime since if they had a common factor, we could divide it out and make an instance of the fraction $\frac{p}{q}$ with a smaller numerator. But $p'$ was chosen to be smallest, and so there is no such common factor. Therefore $\frac{p'}{q'}$ is in lowest terms. $\square$

The previous proof depends on the least number principle: if there is a natural number with a certain property, then there is a smallest such number with that property. In other words, every nonempty set of natural numbers has a least element. 

