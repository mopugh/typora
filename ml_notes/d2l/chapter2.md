# Chapter 2: Preliminaries

* Can think about data as a table where rows correspond to examples and columns correspond to attributes
  
  * Linear alebgra

* Machine learning is about optimization of parameters
  
  * Calculus

* Reason about uncertainity
  
  * Probability

## Data Manipulation

* Two important things to do with data
  
  1. acquire data
  
  2. process data

* n-dimensional array: **tensor**

* the tensor object in PyTorch and  Tensorflow is like `ndarray` from NumPy except it supports
  
  * GPU computation
  
  * differentiation

### Getting Started

### Operations

* Can take scalar functions $f: \mathbb{R} \rightarrow \mathbb{R}$ and apply it **element-wise** to tensor to get a vector-valued function $F: \mathbb{R}^{d} \rightarrow \mathbb{R}^{d}$ by **lifting** the scalar-valued function

* Can take a binary scalar functions $f: \mathbb{R} \times \mathbb{R} \rightarrow \mathbb{R}$ and apply it **element-wise** to two tensors to get a vector-valued function $F: \mathbb{R}^{d} \times \mathbb{R}^{d} \rightarrow \mathbb{R}^{d}$ by **lifting** the scalar-valued function
  
  * the two tensors must be the same shape

### Broadcasting Mechanism

* Can apply element-wise operations to tensors that don't have the same shape in certain situations via **broadcasting**
  
  * First, expand one or both arrays by copying elements appropriately so that after this transformation, the two tensors have the same shape. 
  
  * Second, carry out the element-wise operations on the resulting arrays

### Summary

* Main interface to store and manipulate data for deep learning is the tensor (n-dimensional array)

* Provides functionality such as math operations, indexing, slicing, memory saving and conversion to other Python objects

## Data Preprocessing

### Reading the Dataset


