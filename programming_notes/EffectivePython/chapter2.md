# Chapter 2: Lists and Dictionaries

## Item 11: Know how to slice sequences

* Slicing can be extended to any Python class that implements `__getitem__` and `__setitem__` 

* Basic syntax: `somelist[start:end]` where start is inclusive and end is exclusive

  * When slicing from the start, leave out start, e.g. `assert a[:5] == a[0:5]`
  * When slicing to the end, leave out the end, e.g. `assert a[5:] == a[5:]` 
  * Negative indices for slicing relative to the end of the list
  * Can slice beyond end of list (omits missing entries), but indexing beyond the end of a list is an error
    * `a[:20]` works but `a[20]` doesn't if there, for example, there are 10 elements

* **The result of slicing a list is a whole new list**

  * References to the objects from the original list are maintained
  * Shallow copy (verify this)

* ```python
  print('Before ', a)
  a[2:7] = [99, 22, 14]
  print('After  ', a)
  
  >>>
  Before  ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
  After   ['a', 'b', 99, 22, 14, 'h']
  ```

* ```python
  b = a[:]
  assert a == a and b is not a # b is a copy of a
  ```

* Assigning to a list slice replaces that range in the original sequence with what's referenced even if the lengths are different

## Item 12: Avoid striding and slicing in a single expression

*  `somelist[start:end:stride]` 

  ```python
  x = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
  odds = x[::2]
  evens = x[1::2]
  print(odds)
  print(evens)
  
  >>>
  ['red', 'yellow', 'blue']
  ['orange', 'green', 'purple']
  ```

* Reversing a string:

  ```python
  x = '寿司'
  y = x[::-1] # doesn't work for UTF-8 byte string
  print(y)
  
  >>>
  司寿
  ```

* Specifying start, end and stride in a slice can be extremely confusing.

## Item 13: Prefer catch-all unpacking over slicing

* Catch-all unpacking through a **starred expression**

  ```python
  car_ages = [0, 9, 4, 8, 7, 20, 19, 1, 6, 15]
  
  oldest, second_oldest, *others = car_ages_descending
  print(oldest, second_oldest, others)
  
  >>>
  20 19 [15, 9, 8, 7, 6, 4, 1, 0]
  ```

* Starred expressions become list instances in all cases

  * If there are no leftover items, the catch-all will be an empty list 

* Also works for other iterators

## Item 14: Sorty by complex criteria using the `key` parameter

* list has a bulit in sort method

* sort can take a `key` parameter which is a function that takes an element type in the list and returns a value that can be compared

  ```python
  class Tool:
      def __init__(self, name, weight):
          self.name = name
          self.weight = weight
  
      def __repr__(self):
        return f'Tool({self.name!r}, {self.weight})'
  
  tools = [
      Tool('level', 3.5),
      Tool('hammer', 1.25),
      Tool('screwdriver', 0.5),
      Tool('chisel', 0.25),
  ]
  
  print('Unsorted:', repr(tools))
  tools.sort(key=lambda x: x.name)
  print('\nSorted: ', tools)
  
  >>>
  Unsorted: [Tool('level',        3.5),
             Tool('hammer',       1.25),
             Tool('screwdriver',  0.5),
             Tool('chisel',       0.25)]
  
  
  Sorted: [Tool('chisel',         0.25),
           Tool('hammer',         1.25),
           Tool('level',          3.5),
           Tool('screwdriver',    0.5)]
    
   # Compare off of multiple criteria using tuples
  power_tools.sort(key=lambda x: (x.weight, x.name))
  print(power_tools)
  
  >>>
  [Tool('drill',        4),
   Tool('sander',       4),
   Tool('circular saw', 5),
   Tool('jackhammer',   40)]
  ```

* In case you need to sort in multiple directions on different criteria, utilize the fact Python provides a **stable** sorting algorithm

  * Thus can call sort multiple times on the same list to combine different criteria. 

  ```python
  # Need to sort in the opposite order of the desired sorting
  # i.e. want weight then name
  # so sort name first, then weight
  power_tools.sort(key=lambda x: x.name) # Name ascending
  power_tools.sort(key=lambda x: x.weight, # Weight descending
                   reverse=True)
  print(power_tools)
  
  >>>
  [Tool('jackhammer',   40),
   Tool('circular saw', 5),
   Tool('drill',        4),
   Tool('sander',       4)]
  ```

## Item 15: Be cautious when relying on `dict` insertion ordering

* In Python 3.5 and before, iterating over `dict` returns keys in arbitrary order
* In Python 3.6 and beyond, dictionaries preserve insertion order
* Side note: classes uses `dict` type for their instance dictionaries 
* Side note: `collections` has `OrderedDict` class which has better performance if you need to handle a high rate of key insertions and `popitem` calls. 

## Item 16: Prefer `get` over `in` and `KeyError` to handle missing dictionary keys

* Three fundamental operations for interacting with dictionaries

  * accessing
  * assigning
  * deleting keys and their associated values

* Use `get` built in method which automatically checks if the key exists and if it doesn't assigns a default value

  ```python
  count = counters.get(key, 0)
  counters[key] = count + 1
  ```

  ```python
  # for lists
  names = votes.get(key)
  if names is None:
    votes[key] = names = []
  names.append(who)
  
  # or using assignment expressions
  if (names := votes.get(key)) is None:
    votes[key] = names = []
  names.append(who)
  ```

* `setdefault` method for check and default value if it doesn't exist

  ```python
  names = votes.setdefault(key, [])
  names.append(who)
  ```

  * *Note*: the default value in `setdefault` is always called, which can be expensive

* Four common ways to detect and handle missing keys in dictionaries

  * `in`
  * `KeyError`
  * `get`
  * `setdefault`

## Item 17: Prefer `defaultdict` Over `setdefault` to Handle Missing Items in Internal State

```python
from collections import defaultdict

class Visits:
    def __init__(self):
       self.data = defaultdict(set) # accepts function (e.g. constructor)

    def add(self, country, city):
       self.data[country].add(city) # adds to returned set

visits = Visits()
visits.add('England', 'Bath')
visits.add('England', 'London')
print(visits.data)

>>>
defaultdict(<class 'set'>, {'England': {'London', 'Bath'}})
```

* *Note*: The function to `defaultdict` cannot take any arguments.

## Item 18: Know How to Construct Key-Dependent Default Values with `__missing__` 

* Subclass `dict` and implement `__missing__` 

```python
def open_picture(profile_path):
    try:
        return open(profile_path, 'a+b')
    except OSError:
        print(f'Failed to open path {profile_path}')
        raise

class Pictures(dict):
    def __missing__(self, key):
        value = open_picture(key)
        self[key] = value
        return value

pictures = Pictures()
handle = pictures[path]
handle.seek(0)
image_data = handle.read()
```

* `__missing__` must create the new default value for the key, insert it into the dictionary, and return it to the caller.
* `setdefault` is a bad fit when creating the default value has high computational cost or may raise exceptions
* Since the function passed to `defaultdict` cannot have any arguments, it's impossible to have a default value that depends on the key being accessed.