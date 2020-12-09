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

