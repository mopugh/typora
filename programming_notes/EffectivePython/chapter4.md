# Chapter 4: Comprehensions and Generators

* The result of a call to a generator function can be used anywhere an iterator is appropriate
  * for loops, starred expressions
* Generators can improve performance, reduce memory, and increase readability

## Item 27: Use Comprehensions Instead of `map` and `filter`

```python
even_squares = [x**2 for x in a if x % 2 == 0]

# don't use
alt = map(lambda x: x**2, filter(lambda x: x % 2 == 0, a))
assert even_squares == list(alt)

# dictionary compreshension
even_squares_dict = {x: x**2 for x in a if x % 2 == 0}
# set comprehension
threes_cubed_set = {x**3 for x in a if x % 3 == 0}
```

## Item 28: Avoid More Than Two Control Subexpressoins in Comprehensions

```python
# Convert matrix to a flat list
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flat = [x for row in matrix for x in row]

# element-wise squaring
squared = [[x**2 for x in row] for row in matrix]

# Note
# extends vs appends

# avoid
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
filtered = [[x for x in row if x % 3 == 0]
            for row in matrix if sum(row) >= 10]
# hard to read, error prone
```

## Item 29: Avoid Repeated Work in Comprehensions by Using Assignment Expressions

```python
found = {name: get_batches(stock.get(name, 0), 8)
         for name in order
         if get_batches(stock.get(name, 0), 8)}

# Use assignment expression (walrus operator)
found = {name: batches for name in order
         if (batches := get_batches(stock.get(name, 0), 8))}

# Loop variable leakage
for count in stock.values(): # Leaks loop variable
    pass
print(f'Last item of {list(stock.values())} is {count}')

# Doesn't leak with comprehensions
half = [count // 2 for count in stock.values()]
print(half) # Works
print(count) # Exception because loop variable didn't leak
```

* Use assignment expressions only in condition part
  * If in assignment part, can leak

## Item 30: Consider Generators Instead of Returning Lists

* Generators are produced by functions that use `yield` expressions

  ```python
  # consider using generators instead
  def index_words(text):
      result = []
      if text:
          result.append(0)
      for index, letter in enumerate(text):
          if letter == ' ':
              result.append(index + 1)
     return result
  
  # no interactions with lists
  def index_words_iter(text):
      if text:
          yield 0
      for index, letter in enumerate(text):
          if letter == ' ':
               yield index + 1
  ```

* A generator function does not actually run by instead immediately returns an iterator

  * with each call to the `next` built-in function, the iterator advances the generator to its next `yield` expression
  * Each value passed to `yield` by the generator is returned by the iterator to the caller

  ```python
  # can convert to list if needed
  result = list(index_words_iter(address))
  ```

* Another problem with returning a list vs. generator is the list has to store all the values before returning.

  * Can cause crash for huge inputs

## Item 31: Be Defensive When Iterating Over Arguments

* An iterator produces its results only a single time.

  * Iterate over an iterator or generator that has already raised a `StopIteration` exception, you will not get a result the second time
  * will not get an error

* If you want to iterate mulitple times, create a new container class that implements the **iterator protocol**

  ```python
  class ReadVisits:
    def __init__(self, data_path):
      self.data_path = data_path
      
    def __iter__(self):
      with open(self.data_path) as f:
        for line in f:
          yield int(line)
  ```

* Beware of functions and methods that iterate over input arguments multiple times. 

* Can detect if a value is an iterator if calling `iter` produces the same value. 

  * Or can use `isinstance` built-in along with `collections.abc.Iterator` class

## Item 32: Consider Generator Expressions for Large List Comprehensions

* List comprehensions generate new lists and thus take up potentially a lot of memory

* Consider **generator expressions** which evaluate to an iterator that yields one item at a time

  ```python
  it = (len(x) for x in open('my_file.txt'))
  print(it)
  
  >>>
  <generator object <genexpr> at 0x108993dd0>
  
  print(next(it))
  print(next(it))
  
  >>>
  100
  57
  ```

* Can compose generator expressions

  ```python
  roots = ((x, x**0.5) for x in it)
  ```

## Item 33: Compose Multiple Generators with `yield from`

* `yield from` allows you to yield all values from a nested generator before returning control to the parent generator.

  ```python
  import timeit
  
  def child():
      for i in range(1_000_000):
          yield i
  
  def slow():
      for i in child():
          yield i
  
  def fast():
      yield from child()
  
  baseline = timeit.timeit(
      stmt='for _ in slow(): pass',
      globals=globals(),
      number=50)
  print(f'Manual nesting {baseline:.2f}s')
  comparison = timeit.timeit(
     stmt='for _ in fast(): pass',
     globals=globals(),
     number=50)
  print(f'Composed nesting {comparison:.2f}s')
  
  reduction = -(comparison - baseline) / baseline
  print(f'{reduction:.1%} less time')
  
  >>>
  Manual nesting 4.02s
  Composed nesting 3.47s
  13.5% less time
  ```

* `yield from` allows you to compose multiple nested generators together into a single combined generator

* `yield from` provides better performance than manually iterating nested generators and yielding their outputs.

## Item 34: Avoid Injecting Data into Generators with `send`

* `send` method can be used to provide streaming inputs to a generator at the same time it's yielding outputs.

  ```python
  def wave_modulating(steps):
      step_size = 2 * math.pi / steps
      amplitude = yield             # Receive initial amplitude
      for step in range(steps):
          radians = step * step_size
          fraction = math.sin(radians)
          output = amplitude * fraction
          amplitude = yield output  # Receive next amplitude
  
  def run_modulating(it):
      amplitudes = [
          None, 7, 7, 7, 2, 2, 2, 2, 10, 10, 10, 10, 10]
      for amplitude in amplitudes:
          output = it.send(amplitude)
          transmit(output)
  
  run_modulating(wave_modulating(12))
  
  >>>
  Output is None
  Output:   0.0
  Output:   3.5
  Output:   6.1
  Output:   2.0
  Output:   1.7
  Output:   1.0
  Output:   0.0
  Output:  -5.0
  Output:  -8.7
  Output: -10.0
  Output:  -8.7
  Output:  -5.0
  ```

* providing an input iterator to a set of composed generators is a better approach than the `send` method, which should be avoided.

  ```python
  def wave_cascading(amplitude_it, steps):
      step_size = 2 * math.pi / steps
      for step in range(steps):
          radians = step * step_size
          fraction = math.sin(radians)
          amplitude = next(amplitude_it) # Get next input
          output = amplitude * fraction
          yield output 
  ```

## Item 35: Avoid Causing State Transitions in Generators with `throw`

