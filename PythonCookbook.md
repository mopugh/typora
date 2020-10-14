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

