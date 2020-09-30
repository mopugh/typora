# Notes on Programming

## Change Log

* 9/30/20: Started Chapter 2 up to page 38.

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

# End for 9/30/20: Page 38 Augmented Assignment with Sequences

