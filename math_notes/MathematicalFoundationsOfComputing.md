# Mathematical Foundations of Computing

## Chapter 1: Sets and Cantor's Theorem

### What is a set?

- **Definition:** A _set_ is an _unordered_ collection of _distinct_ elements.
- **Definition:** An _element_ is something contained within a set.
  - $x \in S$, $x \notin S$ 
- **Definition:** The _empty set_ is the set that does not contain any elements.
  - $\emptyset$ , $\forall x, x \notin \emptyset$

### Operations on sets

* **Definition:** The _intersection_ of two sets S and T, denoted S $\cap$ T, is the set of elements contained in both S and T. 
  * The intersection of two sets that have no elements in common is the empty set.
* **Definition**: The *union* of two sets $A$ and $B$, denoted $A \cup B$, is the set of all elements contained in either of the two sets.
* Note: union and intersection are set operations
  * $\{1,2,3\} \cup 4$ and $\{1,2,3\} \cap 4$ are meaningless
  * Did one mean $\{1,2,3\} \cup \{4\}$ or $\{1,2,3\}\cap\{4\}$
* When computing the union or intersection, only care about the elements of the sets and don't care if the elements themselves are sets.
  * $\{\{1,2\}, \{3\}, \{4\}\} \cap \{\{1,2,3\}, \{4\}\} = \{\{4\}\}$

* Note that union and intersection are symmetric
* **Definition**: The *set difference* of $A$ and $B$, denoted $A - B$ or $A \setminus B$, is the set of elements contained in $A$ but not contained in $B$.
  * $\{1,2,3\} - \{3,4,5\} = \{1,2\}$
  * Set difference is not symmetric
* **Definition**: The *set symmetric difference* of two sets $A$ and $B$, denoted $A \triangle B$, is the set of elements contained in exactly one of $A$ or $B$, but not both.
  * $\{1,2,3\} \triangle \{3,4,5\} = \{1,2,4,5\}$

### Special Sets

* **Definition**: The **set of all integers** is denoted $\mathbb{Z}$. Intuitively, it is the set $\{\ldots, -2, -1, 0, 1, 2, \ldots\}$ 
* **Definition**: The **set of all natural numbers**, denoted $\mathbb{N}$, is the set $\mathbb{N} = \{0, 1, 2, 3, \ldots\}$
  * Represent answers to questions of the form "how many"?
* **Definition**: The **set of positive natural numbers** $\mathbb{N}^{+}$ is the set $\mathbb{N}^{+} = \{1, 2, 3,\ldots\}$
* Think of **real numbers** representing arbitrary measurements. 
  * The real numbers are denoted $\mathbb{R}$
* **Definition**: A **finite set** is a set containing only finitely many elements. An **infinite set** is a set containing infinitely many elements. 

### Set-Builder Notation

#### Filtering Sets

Want to be able to define sets by gathering together all the objects that share some common property. 

* Example: (even numbers) $\{n \vert n \in \mathbb{N} \textrm{ and n is even}\}$
  * read as "the set of all n, where n is a natural number and n is even"

Generally: $\{variable \vert \textrm{conditions on that variable}\}$. The variable name does not matter.

**Definition**: A **predicate** is a statement about some object x that is either true of false.

* Example: "x < 0" is a predicate that is true if x is less than zero and false otherwise. 
* It is not required that the predicate be checkable by a computer program. 

Formally, the definition of set-builder notation:

**Definition**: The set $\{x \vert \mathbf{P}(x) \}$ is the set of all $x$ such that $\mathbf{P}(x)$ is true.

*Note*: Set-builder notation can lead to *paradoxical sets* 

#### Transforming Sets

Sometimes easy to consider how an element of a set would be generated rather than describe some property shared by the elements of the set. 

* $\{n \vert \textrm{ there is some } m \in \mathbb{N} \textrm{ such that } n = m^2\}$ vs. $\{n^2 \vert n \in \mathbb{N}\}$ 

In the latter case, we transform the elements of one set to get the elements of the desired set. 

### Relations on Sets

#### Set Equality

**Definition**: If A and B are sets, then $A = B$ precisely when they have teh same elements as one another. This definition is sometimes called the **axiom of extensionality**. 

*Note*: The manner in which two sets are described has no bearing on whether or not they are equal. All that matters is what the two sets contain.

This is why the empty set is unique: any two sets that have no elements must be equal to each other since they have the same elements (namely, no elements).

#### Subsets and Supersets

