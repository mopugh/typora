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

