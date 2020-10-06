# Notes on Programming

## Change Log

* 9/30/20: Started Chapter 2 up to page 38.
* 10/1/20: Started chapter 2 of Haskell from First Principles. Fisished upto and including section 2.4

## Python

### Fluent Python

#### Chapter 1: The Python Data Model

#### Chapter 2: An Array of Sequences

* Container sequences: hold references to the objects they contain
  * list
  * tuple
  * collections.deque
* Flat sequences: physically store the value of each item within its own memory space
  * str
  * bytes
  * bytearray
  * memoryview
  * array.array
* Mutable sequences
  * list
  * bytearray
  * array.array
  * collections.deque
  * memoryview
* Immutable sequences
  * tuple
  * str
  * bytes

##### List Comprehensions and Generator Expressions

* List comprehensions do one thing: build a new list

* If list comprehension is multiple lines, consider using a regular for loop

* line breaks are ignored inside paired [], {} or ()

* local variables in a list comprehension do not mask variables from surrounding scope in python3

* List comprehensions build lists from sequences or any other iterable type by filtering and transforming items

  * replace map and filter

* Cartesian product example:

  * ```python
    colors = ['black', 'white']
    sizes = ['S', 'M', 'L']
    tshirts = [(color, size) for color in colors
              							 for size in sizes]
    
    #output
    [('black', 'S'), ('black', 'M'), ('black', 'L'), ('white', 'S'),
    ('white', 'M'), ('white', 'L')]
    ```

* List comprehensions can only generate new lists

* Generator expressions use same syntax as list comprehensions but use paratheses rather than brackets

  * ```python
    symbols = '$¢£¥€¤'
    tuple(ord(symbol) for symbol in symbols)
    # output
    (36, 162, 163, 165, 8364, 164)
    
    import array
    array.array('I', (ord(symbol) for symbol in symbols))
    # output
    array('I', [36, 162, 163, 165, 8364, 164]
    ```

* If a generator expression is the single argument in a function call, there is no need to duplicate the enclosing paratheses.

* Generator expressions yields items one at a time instead of building the entire list which can save memory.

  * ```python
    colors = ['black', 'white']
    sizes = ['S', 'M', 'L']
    for tshirt in ('%s %s' %(c, s) for c in colors for s in sizes):
      print(tshirt)
    # output
    black S
    black M
    black L
    white S
    white M
    white L
    ```

##### Tuples are not just immutable lists

* Tuples can be used as immutable lists or also as recors with no field names

  * position gives meaning

* Tuple Unpacking

  * works with any iterable object

  * requires yielding exactly one item per variable in the receiving tuple

    * Unless you use the * operator 

  * parallel assignment an example

    * ```python
      b, a = a, b
      ```

  * unpacking with * operator example

    * ```python
      t = (20, 8)
      divmod(*)
      # output
      (2, 4)
      
      divmod(20, 8)
      # output
      (2, 4)
      ```

* Using * to grab excess items

  * ```python
    a, b, *rest = range(5)
    a, b, rest
    # output
    (0, 1, [2, 3, 4])
    
    a, b, *rest = range(2)
    a, b, rest
    # output
    (0, 1, [])
    ```

  * parallel assignment example

    * ```python
      a, *body, c, d = range(5)
      a, body, c, d
      # output
      (0, [1, 2], 3, 4)
      ```

* Can use nested tuple unpacking

##### Named Tuples

```python
from collections import namedtuple
City = namedtuple('City', 'name country population coordinates')
tokyo = City('Tokyo', 'JP', 36.933, (35.689722, 139.691667))
tokyo
City(name='Tokyo', country='JP', population=36.933, coordinates=(35.689722,
139.691667))
tokyo.population
36.933
tokyo.coordinates
(35.689722, 139.691667)
tokyo[1]
'JP'
```

* Named uuples require
  * class name
  * list of field names given by an iterable of strings or a single space delimited string

* Methods:
  * _fields: a tuple of field names of the class
  * _make(): allows instantiation from an iterable
  * _asdict(): returns a collections.OrderedDict

##### Tuples as Immutable Lists

* Can't add or remove items from a tuple

##### Slices

* Why to exclude last item in a slice?
  * Easy to get the length of a slice, e.g. range(3) has 3 items
  * Easy to compute length given start and stop: stop - start
  * Easy to split into two parts at an index x: my_list[:x] and my_list[x:]
* s[a : b : c]
  * a is the start index
  * b is the end of the slice
  * c is the stride
  * e.g. s[start:stop:step]
* Can name slices
* Can assign to slices

##### Using + and * with Sequences

* Sequences must be of the same type

* Returns a new sequence

  * **WARNING**: Beware when sequence contains mutable items! 

    * E.g. my_list = [[]] * 3 results in a list with three references to the same inner list

    * Right: 

      ```python
      board = [['_'] * 3 for i in range(3)]
      board
      [['_', '_', '_'], ['_', '_', '_'], ['_', '_', '_']]
      board[1][2] = 'X'
      board
      [['_', '_', '_'], ['_', '_', 'X'], ['_', '_', '_']]
      ```

      Acts like

      ```python
      >>> board = []
      >>> for i in range(3):
      ... 	row = ['_'] * 3 #
      ... 	board.append(row)
      ...
      >>> board
      [['_', '_', '_'], ['_', '_', '_'], ['_', '_', '_']]
      >>> board[2][0] = 'X'
      >>> board #
      [['_', '_', '_'], ['_', '_', '_'], ['X', '_', '_']]
      ```

      

    * Wrong:

      ```python
      weird_board = [['_'] * 3] * 3
      weird_board
      [['_', '_', '_'], ['_', '_', '_'], ['_', '_', '_']]
      weird_board[1][2] = 'O'
      weird_board
      [['_', '_', 'O'], ['_', '_', 'O'], ['_', '_', 'O']]
      ```

      Acts like

      ```python
      row = ['_'] * 3
      board = []
      for i in range(3):
      	board.append(row)
      ```

### First Course on Data Structures in Python

#### Chapter 2: Basic Python

##### Sequence, Selection, and Iteration

* One model for imperative programming: Sequence-Selection-Iteration
  1. **Sequence**: Performing operations one at a time in a specified order
  2. **Selection**: Using conditional statements such as `if` to select which operations to execute
  3. **Iteration**: Repeating some operation using loops or recursion

##### Expressions and Evaluation

* **Expressions** get **evaluated** and produce a **value** 
* **Operator precedence** determines order of operations

##### Variables, Types and State

* **State**: stored information

* Store information in **variables**

* In Python a variable is created by an **assignment** statement

  * `variable name = some_value`
  * The equal sign is doing something (assignment) rather than describing something (equality)
  * RHS of equal sign is evaluated first
    * thus `x = x+1` works
  * An assignment statement is not an expression: it does not have a value

* Every name is associated with a piece of data called an object

* The name of a variable, by itself, is treated as an expression that evaluates to whatever object it is mapped to.

* Every object has a **type**

  * types often determine what you can do with the variable

* **Atomic types**

  * integers
  * floats
  * booleans

* Can inspect the type of a variable using `type()` 

* Objects have three things:

  * **identity**
  * **type**
  * **value**
  * There's a difference between a variable and the object the variable represents

* Can use the `is` keyword to see if two objects are the same

  ```python
  x = [1, 2, 3]
  y = x
  z = [1, 2, 3]
  
  print(x is y) # True
  print(x is z) # False
  print(x == z) # True
  ```

* An object cannot change its identity

* You cannot change the type of an object

* You can reassign a variable to point to a different object

  ```python
  x = 2
  print('x =', x) # x = 2
  print('float(x) =', float(x)) # float(x) = 2.0
  print('x still has type', type(x)) # x still has type <class 'int'>
  
  print('Overwriting x.') # Overwriting x
  x = float(x)
  print('Now, x has type', type(x)) # Now, x has type <class 'float'>
  ```

* A **string** is a sequence of characters.

  * There is no character class
  * Strings are immutable

* If the value of an object can be changed, it is **mutable**

* If the value of an object cannot be changed, it is **immutable**  

##### Collections

###### Strings

* **Strings** are sequences of characters
* You can **concatenate** strings to create a new string with the plus sign
* You can access individual characters using square brackets and an **index**
* The name of the class for strings is **str** 
* You can turn many objects into a string with the `str()` function

###### Lists

* **Lists** are ordered sequences of objects
  * They do not have to be the same type
* The **elements** of the list are separated by commas
* You can append an item to the end of a list `L` by `L.append(newitem)` 
* You can index into a list with square brackets
  * You can use negative indices to count backwards fro the end of the list
* You can overwrite values in a list using regular assignment statements

###### Tuples

* An immutable ordered sequence of objects 
  * Can access items
  * Cannot change what items are in a tuple
    * Same is true of strings

###### Dictionaries

* store key-value pairs
  * key provides value
* Keys can be different types but must be immutable
* Accessing a key that's not in the dictionary is a `KeyError`
* Dictionaries are unordered 
  * **nonsequential collection**

###### Sets

* An unordered collection of items without duplication
  * **nonsequential collection**
* Add an element to a set with the `add` method

##### Common things to do with collections

* `len` function gets the number of elements (length) of the collection
* For sequential types (list, tuples, string) can **slice** a subsequence
  * Slices are half-open (do not include the end index)
  * Slicing a collectoin creates a new object

##### Iterating Over a Collection

* `for` loop 

##### Other Forms of Control Flow

* **Control flow** refers to the commands in a language that affect the order in which operations are executed.
  * `if`
    * Evaluates *predicate* and if `True` then a block of code is executed
    * An example of *selection* 
    * can also have an `else` clause
  * `while` loop
    * repeat block of code until predicate is `False` 
  * `try` block is the way to catch and recover from errors
  * functions change the control flow
    * functions can be treated like any other object (first class)

##### Modules and Imports

* A single `.py` file is called a **module** 
* Import a module using the keyword `import` and use the name of the file without the `.py` extension
* The module has its own namespace
  * ``__name__`` attribute to determine how the module is being used
    * `__main__` means its being run directly
    * `__name__` has the name of the module means its being imported
* Modules are only exectued the first time they are imported
* `from module name import thethingIwanted` imports `thethingIwanted` and does not need to be preced by `modulename` and a dot
* Do not use `from modulename import *` 
* Can rename module after importing, e.g. `import numpy as np` 
  * Shorter name
  * avoid naming conflicts

#### Chapter 3: Object-Oriented Programming

* A primary goal of **object-oriented programming** is to make it possible to write code that is close to the way you think about the things your code represents.
* A **class** is a data type
  * An **object** is an **instance** of a class
    * E.g. ``myList = []``, myList is an object of type list
  * Can use `isinstance` to return boolean 
  * Can use ``type`` to print type
  * *type* and *class* are (mostly) synonymous in Python
* A function defined in a class is called a **method**. 
  * Standard convention to use `self` as the first parameter to a method which is the object to be operated on by the method
    * Do not need to pass in the `self` parameter explicitly
      * E.g. `u.norm()` is translated to `vector.norm(u)` 
* ``__init__`` methd is called the **initializer** 
* methods that start and end with two underscores are **magic methods** or **dunder methods** 
  * Don't write your own dunder methods
  * Dunder methods are usually invoked via other mechanisms
    * i.e. `__init__` is called by using the class name as a function, e.g. `Vector(3,4)`
* `__add__` implements `+` operator for a class (class on LHS)
* `__str__` is called by `print` 
* Can use older formatting `"(%f, %f)" % (self.x, self.y)` 

#### Encapsulation and the Public Interface of a Class

* **Encapsulation** has two related meanings
  * Combining data and methods into a single thing - a class
  * Boundary between inside and outside
* In Python, everything is public
  * Can start things that *ought* to be private with a single underscore
* The collection of public attributes (variables and methods) constitute the **public interface** of the class.
  * Used to help write working code
  * Not security

#### Inheritance and "is a" relationships

* **superclass** and **subclass** 

  * Common attributes in superclass

  ```python
  class Polygon:
    def __init__(self, sides, points):
      self._sides = sides
      self._points = list(points)
        if len(self._points) != self._sides:
        raise ValueError("Wrong number of points.")
        
    def sides(self):
    	return self._sides
    
  class Triangle(Polygon):
  	def __init__(self, points):
  		Polygon.__init__(self, 3, points)
  
    def __str__(self):
  		return "I'm a triangle."
  
  class Square(Polygon):
  	def __init__(self, points):
  		Polygon.__init__(self, 4, points)
  	def __str__(self):
  		return "I'm so square."
  ```

  * In the above, `Polygon` is the superclass and `Square` and `Triangle` are the subclasses

* If a method is called and it is not defined in the class, it looks in the superclass.

* The search for the correct function to call is called the **method resolution order** (MRO)

* The initializer of the superclass is not called automatically when we create a new instance of the subclass (unless the subclass doesn't define `__init__`)

* **Inheritance means is a**

* **DRY**: Don't repeat yourself
* The process of removing duplication by putting common code into a superclass is called **factoring out a superclass** 

#### Duck Typing

* Python has build in (parametric) **polymorphism**, so we can pass any type of object we want to a function.
* **Duck typing**: using objects with the appropriately defined methods
  * a concept related to dynamic **typing**, where the type or the class of an object is less important than the methods it defines.
* not every "is a" relationship needs to be expressed by inheritance
  * Example: `str` function works on any object that implements `__str__`, so `str(t)` for `Triangle t` calls `t.__str__()` which is equivalent to `Triangle.__str__(t)`

#### Composition and "has a " relationships

* **Composition**: one class stores an instance of another class

* **Composition means "has a"** 

  ```python
  class MyLimitedList:
    def __init__(self):
      self._L = []
      
    def append(self, item):
      self._L.append(item)
    
    def __getitem__(self, index):
      return self._L[index]
  ```

  

## Haskell

### Haskell Programming from First Principles

#### Chapter 2: Hello, Haskell!

* :: is a way to write down the type signature
  
  * saying *has the type* 
  
* In GHCI, type
  * ":q" to quit
  * ":l" to load a file
  * ":m" stands for module, to unload a file
  * ":r" reload the same file
  
* Everything in Haskell is an *expression* or a *declaration* 
  * Expressions may be values, combination of values, and/or functions applied to values
  * Expressions evaluate to a result
    * For a literal value, the value evaluates to itself
  
* *Normal Form*: an expression is in normal form when there are no more evaluation steps that can be taken, i.e. the expression is in an irreducible form.
  
  * Reducible expressions are called *redexes*. 
  
* A **function** 
  * maps an input or set of inputs to an output
  * an expression that is applied to an argument and always returns a result
    * In Haskell always takes one argument
      * Multiple arguments are handled by **currying** 
  
* Functions allow abstractions - abstract the parts of the code we want to reuse for different literal values.

* Functions
  * start with the name of the function (function declaration)
    * Function names (and variable names) must start with a lowercase letter
  * followed by the formal parameters of the functions separated by white space
  * followed by an equal sign
  * concluded by an expression that is the body of the function
  
* As with the Lambda calculus, application is evaluation

  * Can replace a function with its definition

* Haskell reduces to Weak Head Normal Form (WHNF)

* Operators are functions that can be used in infix style

* Can somtimes use function infix style

  ```haskell
  10 `div` 4 -- these are equivalent and the answer is 2
  div 10 4
  ```

* alphanumeric function names are prefix by default

* not all prefix functions can be made infix

* If the name is a symbol, it is infix by default and can be made prefix by wrapping it in parentheses, e.g. (+) 

* use ``:info`` command to get associativity and precedence 

  * precedence is on a scale of 0-9, with higher number being higher precedence

* Module names are capitalized

* Indentation is significant in Haskell

* Use spaces and not tabs

* All declarations must start at the same indentation, which is set by the first declaration

# End for 9/30/20: Page 38 Augmented Assignment with Sequences

