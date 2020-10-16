# Notes on Python Cookbook

------

## Chapter 1: Data Structures and Algorithms

### Unpacking a Sequence into Separate Variables

#### Problem

You have an N-element tuple or sequence that you would like to unpack into a collection of N variables.

#### Solution

* Unpacking by assignment. Requires the number of variables match the structure of the sequence

  ```python
  >>> p = (4, 5)
  >>> x, y = p
  
  >>> data = [ 'ACME', 50, 91.1, (2012, 12, 21) ]
  >>> name, shares, price, date = data
  >>> name, shares, price, (year, mon, day) = data
  ```

  Error if there's a mismatch in the number of elements.

#### Discussion

* Works more generally with any object that is iterable, not just tuples or lists. 
* To discard an item, use the `_` 

### Unpacking Elements from Iterables of Arbitrary Length

#### Problem

You need to unpack N elements from an iterable, but the iterable may be longer than N elements.

#### Solution

* Use "star expressions"

  ```python
  def drop_first_last(grades):
  	first, *middle, last = grades # *middle capture all but the first and last grade
  	return avg(middle)
  
  >>> record = ('Dave', 'dave@example.com', '773-555-1212', '847-555-1212')
  >>> name, email, *phone_numbers = user_record # *phone_numbers captures the phone numbers as a list
  ```

#### Discussion

* `*` unpacking

  ```python
  records = [
  ('foo', 1, 2),
  ('bar', 'hello'),
  ('foo', 3, 4),
  ]
  
  def do_foo(x, y):
  	print('foo', x, y)
  
  def do_bar(s):
  	print('bar', s)
  
  for tag, *args in records:
  	if tag == 'foo':
  		do_foo(*args)
  	elif tag == 'bar':
  		do_bar(*args)
          
  >>> record = ('ACME', 50, 123.45, (12, 18, 2012))
  >>> name, *_, (*_, year) = record
  ```

* Functional Flavor

  ```python
  >>> items = [1, 10, 7, 4, 5, 9]
  >>> head, *tail = items
  >>> head
  1
  >>> tail
  [10, 7, 4, 5, 9]
  
  def sum(items): # Recursion isn't Python' strong suit
      head, *tail = items
      return head + sum(tail) if tail else head
  ```

### Keeping the Last N Items

#### Problem

You want to keep a limited history of the last few items seen during iteration or during some other kind of processing.

#### Solution

Perfect use case for `collections.deque`. 

```python
from collections import deque

def search(lines, pattern, history=5):
	previous_lines = deque(maxlen=history)
	for line in lines:
		if pattern in line:
			yield line, previous_lines
		previous_lines.append(line)

# Example use on a file
if __name__ == '__main__':
	with open('somefile.txt') as f:
		for line, prevlines in search(f, 'python', 5):
			for pline in prevlines:
				print(pline, end='')
			print(line, end='')
			print('-'*20)
```

#### Discussion

When writing code to search for items, its common to use a generator function involving `yield`. This decouples the process of searching from the code that uses the result. 

Using `deque(maxlen=N)` creates a fixed-sized queue. 

```python
>>> q = deque(maxlen=3)
>>> q.append(1)
>>> q.append(2)
>>> q.append(3)
>>> q
deque([1, 2, 3], maxlen=3)
>>> q.append(4)
>>> q
deque([2, 3, 4], maxlen=3)
>>> q.append(5)
>>> q
deque([3, 4, 5], maxlen=3)
```

Can use deque as a simple queue structure. Can add or pop from either end in $O(1)$ time. 

### Finding the Largest or Smallest N Items

#### Problem

You want to make a list of the largest or smallest N items in a collection.

#### Solution

The `heapq` module has two functions, `nlargest()` and `nsmallest()`, that do exactly this

```python
import heapq

nums = [1, 8, 2, 23, 7, -4, 18, 23, 42, 37, 2]
print(heapq.nlargest(3, nums)) # Prints [42, 37, 23]
print(heapq.nsmallest(3, nums)) # Prints [-4, 1, 2]

portfolio = [
	{'name': 'IBM', 'shares': 100, 'price': 91.1},
	{'name': 'AAPL', 'shares': 50, 'price': 543.22},
	{'name': 'FB', 'shares': 200, 'price': 21.09},
	{'name': 'HPQ', 'shares': 35, 'price': 31.75},
	{'name': 'YHOO', 'shares': 45, 'price': 16.35},
	{'name': 'ACME', 'shares': 75, 'price': 115.65}
]

cheap = heapq.nsmallest(3, portfolio, key=lambda s: s['price'])
expensive = heapq.nlargest(3, portfolio, key=lambda s: s['price'])	
```

#### Discussion

When N is small compared to the total length of the collection, this approach is superior. Use `heapq.heappop()` to get the smallest element of the collection.

* If trying to get the minimum or maximum element, `min` and `max` are faster respectively.
* If N is about the same size as the collection, then sorting is faster
  * `nlargest` and `nsmallest` is adaptive and can make these optimization

### Implementing a Priority Queue

#### Problem

You want to implement a queue that sorts items by a given priority and always returns the item with the highest priority on each pop operation.

#### Solution

Use `heapq` module

```python
import heapq

class PriorityQueue:
	def __init__(self):
		self._queue = []
  	self._index = 0
  
  def push(self, item, priority):
  	heapq.heappush(self._queue, (-priority, self._index, item))
  	self._index += 1
  
  def pop(self):
  	return heapq.heappop(self._queue)[-1]
```

#### Discussion

`heapq.push()` and `heapq.heappop()` insert and remove items from a list_queue in such a way that the first item in the list has the smallest priority. Here the queue consists of tuples of the form `(-priority, index, item)`. The priority is negated so we return the highest priority. The index variable orders items with the same priority level. Don't want the following problem:

```python
>>> a = (1, Item('foo'))
>>> b = (5, Item('bar'))
>>> a < b
True
>>> c = (1, Item('grok'))
>>> a < c
Traceback (most recent call last):
File "<stdin>", line 1, in <module>
TypeError: unorderable types: Item() < Item()
```

### Mapping Keys to Multiple Values in a Dictionary

#### Problem

You want to make a dictionary that maps keys to more than one value (so-called "multidict").

#### Solution

A dictionary maps each key to a single value. If you want a key to map to multiple values, you need to store the multiple values in another container, e.g.

```python
d = {
	'a' : [1, 2, 3],
	'b' : [4, 5]
}
e = {
	'a' : {1, 2, 3},
	'b' : {4, 5}
}
```

Can use `collections.defaultdict` module to help with this.

```python
from collections import defaultdict

d = defaultdict(list)
d['a'].append(1)
d['a'].append(2)

e = defaultdict(set)
e['a'].add(1)
e['a'].add(2)
```

Warning: defaultdict will create entries for any key that is accessed. 

#### Discussion

In principle this problem is easy, but initializing the first value is messy.

```python
d = {}
for key, value in pairs:
	if key not in d:
		d[key] = []
	d[key].append(value)
```

vs

```python
d = defaultdict(list)
for key, value in pairs:
	d[key].append(value)
```

### Keeping Dictionaries in Order

#### Problem

You want to create a dictionary and you want to control the order of items when iterating or serializing.

#### Solution

Use `collections.OrderedDict` which preserves the insertion order of the data when iterating.

```python
from collections import OrderedDict

d = OrderedDict()
d['foo'] = 1
d['bar'] = 2
d['spam'] = 3
d['grok'] = 4

# Outputs "foo 1", "bar 2", "spam 3", "grok 4"
for key in d:
	print(key, d[key])
```

Useful for serializing or encoding into a different format.

```python
>>> import json
>>> json.dump(d)
'{"foo": 1, "bar": 2, "spam": 3, "grok": 4}'
```

#### Discussion

`OrderedDict` uses a doubly linked list that orders the keys according to insertion ordered. `OrderedDict` is more than twice as large as a normal dictionary due to this fact. 

### Calculating with Dictionaries

#### Problem

You want to perform various calculations (e.g. minimum value, maximum value, sorting, etc.) on a dictionary of data

#### Solution

The idea is to invert the keys and values using `zip()`

```python
prices = {
	'ACME': 45.23,
	'AAPL': 612.78,
	'IBM': 205.55,
	'HPQ': 37.20,
	'FB': 10.75
}

min_price = min(zip(prices.values(), prices.keys()))
# min_price is (10.75, 'FB')
max_price = max(zip(prices.values(), prices.keys()))
# max_price is (612.78, 'AAPL')
prices_sorted = sorted(zip(prices.values(), prices.keys()))
# prices_sorted is [(10.75, 'FB'), (37.2, 'HPQ'),
# (45.23, 'ACME'), (205.55, 'IBM'),
# (612.78, 'AAPL')]
```

Note: `zip()` creates an iterator that can only be consumed once

#### Dicussion

Don't want the following:

```python
min(prices) # Returns 'AAPL'
max(prices) # Returns 'IBM'
min(prices, key=lambda k: prices[k]) # Returns 'FB'
max(prices, key=lambda k: prices[k]) # Returns 'AAPL'
# Need the extra step to find the price
min_value = prices[min(prices, key=lambda k: prices[k])]
```

### Finding Commonalities in Two Dictionaries

#### Problem

You have two dictionaries and want to find out what they have in common (same keys, same values, etc.)

#### Solution

Idea: Use set operations

```python
a = {
	'x' : 1,
	'y' : 2,
	'z' : 3
}
b = {
	'w' : 10,
	'x' : 11,
	'y' : 2
}

# Find keys in common
a.keys() & b.keys() # { 'x', 'y' }
# Find keys in a that are not in b
a.keys() - b.keys() # { 'z' }
# Find (key,value) pairs in common
a.items() & b.items() # { ('y', 2) }
# Make a new dictionary with certain keys removed
c = {key:a[key] for key in a.keys() - {'z', 'w'}}
# c is {'x': 1, 'y': 2}
```

#### Discussion

`items()` and `keys()` support set operations but `values()` does not since not guaranteed to be unique. 

### Removing Duplicates from a Sequence while Maintaining Order

#### Problem

You want to eliminate the duplicate values in a sequence, but want to preserve the order of the remaining items

#### Solution

If the values in the sequence are hashable, use a set and a generator:

```python
def dedupe(items):
	seen = set()
	for item in items:
		if item not in seen:
			yield item
			seen.add(item)

>>> a = [1, 5, 2, 1, 9, 1, 5, 10]
>>> list(dedupe(a))
[1, 5, 2, 9, 10]
```

It the values are not hashable:

```python
def dedupe(items, key=None):
	seen = set()
	for item in items:
		val = item if key is None else key(item)
		if val not in seen:
			yield item
			seen.add(val)
      
>>> a = [ {'x':1, 'y':2}, {'x':1, 'y':3}, {'x':1, 'y':2}, {'x':2, 'y':4}]
>>> list(dedupe(a, key=lambda d: (d['x'],d['y'])))
[{'x': 1, 'y': 2}, {'x': 1, 'y': 3}, {'x': 2, 'y': 4}]
>>> list(dedupe(a, key=lambda d: d['x']))
[{'x': 1, 'y': 2}, {'x': 2, 'y': 4}]
```

The purpose of the key is to specify a function that converts sequence items into a hashable type for duplicate detection. 

#### Discussion

Can't use `set()` directly if you want to maintain order. Using generator to be as general as possible, e.g.

```python
with open(somefile,'r') as f:
	for line in dedupe(f):
```

