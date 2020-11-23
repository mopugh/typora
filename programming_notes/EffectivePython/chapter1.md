# Chapter 1: Pythonic Thinking

## Item 1: Know which version of Python you're using

Command line:

```bash
python --version
python3 --version
```

Python code:

```python
import sys
print(sys.version_info)
print(sys.version)
```

## Item 2: Follow the PEP 8 style guide

### Whitespace

* In Python, whitespace is syntactically significant

### Naming

* Functions, variables and attributes should be in `lowercase_underscore` format
* Protected instance attributes should be in `_leading_underscore` format
* Private instance attributes should be in `__double_leading_underscore` format
* Classes should be in `CapitalizedWord` format
* Module-level constants should be in `ALL_CAPS` format
* Instance methods in classes should use `self`, which refers to the object, as the name of the first parameter
* Class methods should use `cls`, which refers to the class, as the name of the first parameter

### Expressions and statements

* Do not check for empty contains or sequences by comparing the length to zero. Use `if not somelist` and assume the empty value will implicitly evaluate to `False` 
* If you can't fit an expression on one line, surround it with paratheses and add line breaks and indentation to make it easier to read
* Prefer surrounding multiline expressions with paratheses over using the `\` line continuation character

### Imports

* Always use absolute names for modules when importing them, not names relative to the current module's own path. 
  * E.g. to import the `foo` module from within the `bar` package, you should use `from bar import foo` and not just `import foo`
  * If you must use relative imports, use the explicit syntax `from . import foo` 
* Imports should be in section in the following order:
  * standard library modules
  * third-party modules
  * your own modules
* Each subsection should have imports in alphabetical order
* Use a linter (e.g. Pylint)

## Item 3: Know the difference between `bytes` and `str` 

* Two ways to represent sesquences of character data: `bytes` and `str` 

* `bytes` contain raw, unsigned 8-bit values (often displayed in the ASCII encoding)

  ```python
  a = b'h\x65llo'
  print(list(a))
  print(a)
  
  # output
  [104, 101, 108, 108, 111] # output of print(list(a))
  b'hello' # output of print(a)
  ```

* `str` contain Unicode code points that represent textual characters

  ```python
  a = 'a\u0300 propos'
  print(list(a))
  print(a)
  
  >>>
  ['a', 'ˋ', ' ', 'p', 'r', 'o', 'p', 'o', 's']
  à propos
  ```

* `str` instances do not have an associated binary encoding

  * To convert from Unicode to binary, must call `encode` method on `str`

* `bytes` instances do not have an associated text encoding 

  * To convert from bytes to Unicode data, must call `decode` method on `bytes` 

* Unicode sandwich: use encoding and decoding of Unicode data at the furstest boundary of your interfaces.

  * Core of your program should use the `str` type.

  ```python
  def to_str(bytes_or_str):
      if isinstance(bytes_or_str, bytes):
          value = bytes_or_str.decode('utf-8')
      else:
          value = bytes_or_str
      return value  # Instance of str
  print(repr(to_str(b'foo')))
  print(repr(to_str('bar')))
  
  >>>
  'foo'
  'bar'
  
  def to_bytes(bytes_or_str):
      if isinstance(bytes_or_str, str):
          value = bytes_or_str.encode('utf-8')
      else:
          value = bytes_or_str
      return value  # Instance of bytes
  print(repr(to_bytes(b'foo')))
  print(repr(to_bytes('bar')))
  ```

* `bytes` and `str` are not compatible types (e.g. can't use `+` on both types simultaneously)

  * Comparing `bytes` and `str` instances for equality always evaluates to `False`

* file handles default to requiring Unicode strings instead of `bytes` 

  ```python
  with open('data.bin', 'w') as f: # Fix: use 'wb' for write binary
      f.write(b'\xf1\xf2\xf3\xf4\xf5')
  
  >>>
  Traceback ...
  TypeError: write() argument must be str, not bytes
    
  with open('data.bin', 'r') as f: # Fix: use 'rb'
     data = f.read()
  
  >>>
  Traceback ...
  UnicodeDecodeError: 'utf-8' codec can't decode byte 0xf1 in
  ➥position 0: invalid continuation byte
  ```

* Can explicitly pass encoding

  ```python
  with open('data.bin', 'r', encoding='cp1252') as f:
      data = f.read()
  
  assert data == 'ñòóôõ'
  ```

## Item 4: Prefer interpolated f-strings over C-sytle format strings and `str.format`

* **Formatting** is the process of combining predefined text with data values into a singe human-readable message that's stored as a string.

* Python has four ways to format strings

  * `%` operator
  * `format` function
  * `format` method on `str`
  * f-strings

* Most common is the `%` operator

  ```python
  a = 0b10111011
  b = 0xc5f
  print('Binary is %d, hex is %d' % (a, b))
  
  >>>
  Binary is 187, hex is 3167
  
  # Can use a dictionary for formatting
  key = 'my_var'
  value = 1.234
  
  old_way = '%-10s = %.2f' % (key, value)
  
  new_way = '%(key)-10s = %(value).2f' % {
      'key': key, 'value': value} # Original
  
  reordered = '%(key)-10s = %(value).2f' % {
      'value': value, 'key': key} # Swapped
  
  assert old_way == new_way == reordered
  ```

* **format** built-in function

  ```python
  a = 1234.5678
  formatted = format(a, ',.2f')
  print(formatted)
  
  b = 'my string'
  formatted = format(b, '^20s')
  print('*', formatted, '*')
  
  >>>
  1,234.57
  *     my string     *
  ```

* **format** method on `str` type

  ```python
  key = 'my_var'
  value = 1.234
  
  formatted = '{} = {}'.format(key, value)
  print(formatted)
  
  >>>
  my_var = 1.234
  
  # Another example with format specifiers
  formatted = '{:<10} = {:.2f}'.format(key, value)
  print(formatted)
  
  >>>
  my_var      = 1.23
  
  # Specifying positional index
  formatted = '{1} = {0}'.format(key, value)
  print(formatted)
  
  >>>
  1.234 = my_var
  
  # Repeated positional index
  formatted = '{0} loves food. See {0} cook.'.format(name)
  print(formatted)
  
  >>>
  Max loves food. See Max cook.
  ```

### Interpolated format strings

* f-strings are the best formatting option

* prefix string with `f` character

* allow string to reference all names in the current Python scope

  ```python
  key = 'my_var'
  value = 1.234
  
  formatted = f'{key} = {value}'
  print(formatted)
  
  >>>
  my_var = 1.234
  
  # formatting specifiers
  formatted = f'{key!r:<10} = {value:.2f}'
  print(formatted)
  
  >>>
  'my_var' = 1.23
  
  # Can put in python expressions
  for i, (item, count) in enumerate(pantry):
      old_style = '#%d: %-10s = %d' % (
          i + 1,
          item.title(),
          round(count))
  
      new_style = '#{}: {:<10s} = {}'.format(
          i + 1,
          item.title(),
          round(count))
  
     f_string = f'#{i+1}: {item.title():<10s} = {round(count)}'
  
     assert old_style == new_style == f_string
      
  # Can use python expressions as formatting parameters
  places = 3
  number = 1.23456
  print(f'My number is {number:.{places}f}')
  
  >>>
  My number is 1.235
  ```

## Item 5: Write helper functions instead of complex expressions

* As soon as expressions get complicated, it's time to consider splitting them into smaller pieces and moving logic into helper functions
* **DRY**: Don't repeat yourself

## Item 6: Prefer multiple assignment unpacking over indexing

* **tuples** are immutable, ordered sequence of values

  ```python
  snack_calories = {
      'chips': 140,
      'popcorn': 80,
      'nuts': 190,
  }
  items = tuple(snack_calories.items())
  print(items)
  
  >>>
  (('chips', 140), ('popcorn', 80), ('nuts', 190))
  
  item = ('Peanut butter', 'Jelly')
  first = item[0]
  second = item[1]
  print(first, 'and', second)
  
  >>>
  Peanut butter and Jelly
  
  # Can't re-assign values once created
  pair = ('Chocolate', 'Peanut butter')
  pair[0] = 'Honey'
  
  >>>
  Traceback ...
  TypeError: 'tuple' object does not support item assignment
    
  # Un-packing
  item = ('Peanut butter', 'Jelly')
  first, second = item # Unpacking
  print(first, 'and', second)
  
  >>>
  Peanut butter and Jelly
  
  # Bubblesort with tuple unpacking
  def bubble_sort(a):
      for _ in range(len(a)):
          for i in range(1, len(a)):
              if a[i] < a[i-1]:
                  a[i-1], a[i] = a[i], a[i-1] # Swap with unpacking
  
  names = ['pretzels', 'carrots', 'arugula', 'bacon']
  bubble_sort(names)
  print(names)
  
  >>>
  ['arugula', 'bacon', 'carrots', 'pretzels']
  ```

* Unpacking is generalized in Python and can be applied to any iterable, including many levels of iterables within iterables

## Item 7: Prefer `enumerate` over `range` 

* Often want to iterate over a `list` and also know the index of the current item in the `list`

* `enumerate` wraps any iterator with a lazy generator yielding pairs of the loop index and the next value from the iterator. 

  ```python
  flavor_list = ['vanilla', 'chocolate', 'pecan', 'strawberry']
  
  it = enumerate(flavor_list)
  print(next(it))
  print(next(it))
  
  >>>
  (0, 'vanilla')
  (1, 'chocolate')
  
  for i, flavor in enumerate(flavor_list):
      print(f'{i + 1}: {flavor}')
  
  >>>
  1: vanilla
  2: chocolate
  3: pecan
  4: strawberry
    
  # Specify starting number for enumerate
  for i, flavor in enumerate(flavor_list, 1):
      print(f'{i}: {flavor}')
  ```

* prefer `enumerate` instead of looping over a `range` 

## Item 8: Use `zip` to process iterators in parallel

* `zip` wraps two or more iterators with a lazy generator.

* `zip` yields tuples containing the next value from each iterator.

  * keeps yielding values until any one of the wrapped iterators is exhausted (i.e. the output is as long as the shortest input)

  ```python
  names = ['Cecilia', 'Lise', 'Marie']
  counts = [len(n) for n in names]
  
  for name, count in zip(names, counts):
      if count > max_count:
          longest_name = name
          max_count = count
  ```

  * If the lengths of the lists passed to `zip` may not be equal, consider using `zip_longest` from `itertools` 

## Item 9: Avoid `else` blocks after `for` and `while` loops

```python
a = 4
b = 9

for i in range(2, min(a, b) + 1):
    print('Testing', i)
    if a % i == 0 and b % i == 0:
        print('Not coprime')
        break
else: # runs after loop unless break
    print('Coprime')

>>>
Testing 2
Testing 3
Testing 4
Coprime

# Helper function method
def coprime(a, b):
    for i in range(2, min(a, b) + 1):
        if a % i == 0 and b % i == 0:
            return False
    return True

assert coprime(4, 9)
assert not coprime(3, 6)
```

* Python allows `else` block immediately after `for` and `while` loop interior blocks.
* The `else `block after a loop runs only if the loop body did not encounter a `break` statement
* Avoid using `else` blocks after loops because their behavior isn't intuitive and can be confusing

## Item 10: Prevent reptition with assignment expressions

* Assignment expression, a.k.a. **walrus operator**, is new to Python 3.8

  * Normal assignment written `a = b`
  * a `walrus` b is written as `a := b` 

* Fetching a value, checking to see if it's non-zero, and then using it is a common pattern in Python

  * Assign and then evaluate is the fundamental nature of the walrus operator

  ```python
  fresh_fruit = {
      'apple': 10,
      'banana': 8,
      'lemon': 5,
  }
  
  def make_lemonade(count):
      ...
  def out_of_stock():
      ...
  
  count = fresh_fruit.get('lemon', 0) # Fetch value
  if count: # check if non-zero
      make_lemonade(count) #if non-zero, do something
  else:
      out_of_stock()
      
  # Using walrus operator
  if count := fresh_fruit.get('lemon', 0):
      make_lemonade(count)
  else:
      out_of_stock()
      
  # Another example
  def make_cider(count):
      ...
  
  if (count := fresh_fruit.get('apple', 0)) >= 4: # Note () around walrus operator
      make_cider(count)
  else:
      out_of_stock()
  ```

* Use of walrus operator as an approximation of switch statements

  ```python
  if (count := fresh_fruit.get('banana', 0)) >= 2:
      pieces = slice_bananas(count)
      to_enjoy = make_smoothies(pieces)
  elif (count := fresh_fruit.get('apple', 0)) >= 4:
      to_enjoy = make_cider(count)
  elif count := fresh_fruit.get('lemon', 0):
      to_enjoy = make_lemonade(count)
  else:
      to_enjoy = 'Nothing'
  ```

* No do/while loop in Python

  * Need to do something before the loop to set the initial conditions
  * Another something at the end of the loop

* **Loop-and-a-half** idiom

  * Use an infinite loop and a break statement

    ```python
    bottles = []
    while True:                    # Loop
        fresh_fruit = pick_fruit()
        if not fresh_fruit:        # And a half
            break
    
        for fruit, count in fresh_fruit.items():
            batch = make_juice(fruit, count)
            bottles.extend(batch)
    ```

* Walrus operator obviates the need for the loop-and-a-half idom by allowing `fresh_fruit` variable to be reassigned and then conditionally evaluated each time through the `while` loop. 

  ```python
  bottles = []
  while fresh_fruit := pick_fruit():
      for fruit, count in fresh_fruit.items():
          batch = make_juice(fruit, count)
          bottles.extend(batch) 
  ```

* In general, when you find yourself repeating the same expression or assignment multiple times within a grouping of lines, it's time to consider using assignment expressions

### Things to remember

* Assignment expressions use the walrus operator (`:=`) to both assign and evaluate variable names in a single expression, thus reducing repetition
* When an assignment expression is a subexpression of a larger expression, it must be surrounded with parentheses