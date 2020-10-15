# Python Tricks: The Book

## Patterns for Cleaner Python

### Covering Your A** with Assertions 

* assert statements are a debugging aid that test a condition

  * If the condition is true, nothing happens
  * If false, an ``AssertionError`` exception is raised with an optional error message

* **Example**:

  ```python
  def apply_discount(product, discount):
      price = int(product['price'] * (1.0 - discount))
      assert 0 <= price <= product['price']
      return price
  ```

  Guarantees price is less than original price and positive.

* **Tip** Use currency values as integers in cents

#### Why Not Just Use a Regular Exception?

* The proper use of assertions is to inform developers about *unrecoverable* errors in a program. 
* Assertions are meant to be *internal self-checks* 
  * Declaring some condition to be impossible
* **NOT** meant as a mechanism to handle run-time errors.

#### Python's Assert Syntax

* ```python
  assert_stmt ::= "assert" expression1 ["," expression2]
  ```

  * `expression1` is the condition to test
  * `expression2` is the optional error message

* The above gets converteed to roughly

  ```python
  if __debug__:
      if not expression1:
          raise AssertionError(expression2)
  ```

  * Note the debug condition.

#### Common Pitfalls With Using Asserts in Python

* **Caveat #1: Don't Use Asserts for Data Validation**

  * assertions can be globally disabled with the `-O` and `-OO` command line switches, as well as the `PYTHONOPTIMIZE` environmental variable in CPython

  * **Example** 

    ```python
    def delete_product(prod_id, user):
        assert user.is_admin(), 'Must be admin'
        assert store.has_product(prod_id), 'Unknown product'
        store.get_product(prod_id).delete()
    ```

    * **Checking for admin privileges with an assert statement is dangerous** 
      * If asserts are disabled, any user can delete products (the asserts are never run)
    * **The `has_product()` check is skipped when asserts are disabled** 

  * **TAKE AWAY: NEVER USER ASSERTS TO DO DATA VALIDATION** 

  * Better solution

    ```python
    def delete_product(product_id, user):
        if not user.is_admin():
            raise AuthError('Must be admin to delete')
        if not store.has_product(product_id):
            raise ValueError('Unknown product id')
        store.get_product(product_id).delete()
    ```

    * In addition to be safer, raises the appropriate error

* **Caveat #2: Asserts That Never Fail** 

  * Passing a tuple as the first argument in an assert statement will always evaluate to true and never fail

    * tuples are truthy in Python

    * E.g. of an assertion that will never fail

      ```python
      assert(1 == 2, 'This should fail')
      ```

  * **TIP**: Test to make sure your assertions fail (i.e. work)

### Complacent Comma Placement

End all lines with a comma for lists, dicts, and sets

* Bad:

  ```python
  names = ['Alice', 'Bob', 'Dilbert']
  ```

* Good:

  ```python
  names = [
    'Alice',
    'Bob',
    'Dilbert', # Notice the comma!
  ]
  ```

  * When you add an item, only have to modify (add one line) rather than modify two lines by adding a comma to the last line. Good for version control.

##### Context Managers and the `with` Statement

* `with` statement simplifies exception handling by encapsulating standard uses of try/finally statements in so-called context managers

* most commonly used to manage the safe acquisition and release of system resources

* resources are acquired by the `with` statement and released when execution leaves the `with` context.

* Example:

  ```python
  with open('hello.txt', 'w') as f:
    f.write('hello, world!')
  ```

  Translates to:

  ```python
  f = open('hello.txt', 'w')
  try:
    f.write('hello, world!')
  finally:
    f.close()
  ```

  * The `with` statement makes properly acquiring and releasing resources a breeze

  * Note it is not enough to write

    ```python
    f = open('hello.txt', 'w')
    f.write('hello, world!')
    f.close()
    ```

    because if there was an error in `f.write` then the file wouldn't close and the file descriptor would leak.

* Another example:

  ```python
  some_lock = threading.Lock()
  
  # Harmful:
  some_lock.acquire()
  try:
      # Do something...
  finally:
      some_lock.release()
  
  # Better:
  with some_lock:
      # Do something... 
  ```

* Makes code more readable, and less buggy/leaky

#### Supporting `with` in Your Own Objects

* Context Manager: a "protocol" that an object needs to follow in order to support the `with` statement . Require the following methods:

  * `__enter__`
  * `__exit__` 

* Example:

  ```python
  class ManagedFile:
      def __init__(self, name):
          self.name = name
  
  def __enter__(self):
          self.file = open(self.name, 'w')
          return self.file
  
  def __exit__(self, exc_type, exc_val, exc_tb):
      if self.file:
          self.file.close() 
  ```

  ```python
  >>> with ManagedFile('hello.txt') as f:
  ...    f.write('hello, world!')
  ...    f.write('bye now')
  ```

* Using `contextlib` for the above example

  ```python
  from contextlib import contextmanager
  
  @contextmanager
  def managed_file(name):
      try:
          f = open(name, 'w')
          yield f
      finally:
          f.close()
  
  >>> with managed_file('hello.txt') as f:
  ...     f.write('hello, world!')
  ...     f.write('bye now')
  ```

  `managed_file()` is a generator that first acquires the resource and then suspends its own execution and yields the resource so it can be used by the caller. When the caller leaves the `with` context, the generator continues to execute so that any remaining clean-up steps can occur and the resource can get released back to the system.

* Timer Example

  ```python
  import time
  from contextlib import contextmanager
  
  
  class Timer:
      def __init__(self):
          self.start = 0
          self.end = 0
  
      def __enter__(self):
          self.start = time.time()
          return self
  
      def __exit__(self, exc_type, exc_val, exc_tb):
          self.end = time.time()
          print('Time taken: {} seconds'.format(self.end-self.start))
  
  
  @contextmanager
  def timer2():
      try:
          start = time.time()
          yield
      finally:
          end = time.time()
          print('Time taken: {} seconds'.format(end-start))
  
  
  if __name__ == '__main__':
      with Timer() as t:
          for i in range(1000000):
              pass
  
      with timer2() as t2:
          for i in range(1000000):
              pass
  ```

### Underscores, Dunders, and More

* Five underscore patterns and naming conventions:
  * Single leading underscore: `_var`
  * Single trailing underscore: `var_`
  * Double leading underscore: `__var`
  * Double leading and trailing underscore: `__var__`
  * Single underscore: `_` 

#### Single Leading Underscore: `_var`

* The underscore prefix is meant as a *hint* that the variable or method is intended for internal use
  * Does not affect the behavior of the program
    * Not distinction between public and private
* Will not be imported with *wildcard import* 

#### Single Trailing Underscore: `var_` 

* make a variable name that is a Python keyword
  * E.g. `class_` 

#### Double Leading Underscore: `__var`

* With Python class attributes (variables and methods), a double underscore prefix causes the Python interpreter to rewrite the attribute name in order to avoid naming conflicts in subclasses 

  * a.k.a. *name mangling* 

* Example:

  ```python
  class Test:
      def __init__(self):
          self.foo = 11
          self._bar = 23
          self.__baz = 42
  ```

  ```python
  >>> t = Test()
  >>> dir(t)
  ['_Test__baz', '__class__', '__delattr__', '__dict__',
   '__dir__', '__doc__', '__eq__', '__format__', '__ge__',
   '__getattribute__', '__gt__', '__hash__', '__init__',
   '__le__', '__lt__', '__module__', '__ne__', '__new__',
   '__reduce__', '__reduce_ex__', '__repr__',
   '__setattr__', '__sizeof__', '__str__',
   '__subclasshook__', '__weakref__', '_bar', 'foo']
  ```

  Note: `self.__baz` is not in the list. Was converted to `_Test__baz` 

  ```python
  class ExtendedTest(Test):
      def __init__(self):
          super().__init__()
          self.foo = 'overridden'
          self._bar = 'overridden'
          self.__baz = 'overridden'
  ```

  ```python
  >>> t2 = ExtendedTest()
  >>> t2.foo
  'overridden'
  >>> t2._bar
  'overridden'
  >>> t2.__baz
  AttributeError:
  "'ExtendedTest' object has no attribute '__baz'"
  ```

  ```python
  >>> dir(t2)
  ['_ExtendedTest__baz', '_Test__baz', '__class__',
   '__delattr__', '__dict__', '__dir__', '__doc__',
  '__eq__', '__format__', '__ge__', '__getattribute__',
   '__gt__', '__hash__', '__init__', '__le__', '__lt__',
   '__module__', '__ne__', '__new__', '__reduce__',
   '__reduce_ex__', '__repr__', '__setattr__',
   '__sizeof__', '__str__', '__subclasshook__',
   '__weakref__', '_bar', 'foo', 'get_vars']
  ```

  Note: In `ExtendedTest`, `self.__baz` got converted to `_ExtendedTest__baz`. The original `_Test__baz` is still there:

  ```python
  >>> t2._ExtendedTest__baz
  'overridden'
  >>> t2._Test__baz
  42
  ```

* Also affects method names

* "dunder" means "double underscore"

* *name mangling* also affects global variables

#### Double Leading and Trailing Underscore: `__var__`

* *name mangling* does not apply to attributes that start and end with double underscores
* sometimes attributues that start and end with double underscores are called *magic methods* 

#### Single Underscore: `_`

* Per convention, a single stand-alone underscore is used as a name to indicate that a variable is temporary or insignificant 

* Example:

  ```python
  for _ in range(32):
    print('Hello, World.')
  ```

* Can use in unpacking expressions to ignore particular values

  ```python
  >>> car = ('red', 'auto', 12, 3812.4)
  >>> color, _, _, mileage = car
  
  >>> color
  'red'
  >>> mileage
  3812.4
  >>> _
  12
  ```

* Also the previous calculation in an interpreter session is stored as `_` 

### A Shocking Truth About String Formatting

#### "Old Style" String Formatting

* % operator

  ```python
  >>> errno = 50159747054
  >>> name = 'Bob'
  
  >>> 'Hello, %s' % name
  'Hello, Bob'
  
  >>> 'Hey %s, there is a 0x%x error!' % (name, errno)
  'Hey Bob, there is a 0xbadc0ffee error!'
  
  >>> 'Hey %(name)s, there is a 0x%(errno)x error!' % {
  ...     "name": name, "errno": errno }
  'Hey Bob, there is a 0xbadc0ffee error!'
  ```

#### "New Style" String Formatting

* Use the `format()` function

  ```python
  >>> 'Hello, {}'.format(name)
  'Hello, Bob'
  
  >>> 'Hey {name}, there is a 0x{errno:x} error!'.format(
  ...     name=name, errno=errno)
  'Hey Bob, there is a 0xbadc0ffee error!'
  ```

  Note the `:x` which is a *format spec*, which is a suffix after the variable name 

#### Literal String Interpolation (Python 3.6+)

* *Formatted String Literals*

  ```python
  >>> f'Hello, {name}!'
  'Hello, Bob!'
  
  >>> a = 5
  >>> b = 10
  >>> f'Five plus ten is {a + b} and not {2 * (a + b)}.'
  
  'Five plus ten is 15 and not 30.'
  
  >>> f"Hey {name}, there's a {errno:#x} error!"
  "Hey Bob, there's a 0xbadc0ffee error!"
  ```

  Note the last example allows you to do inline arithmetic. One can embed arbitrary Python expressions. 

#### Template Strings

* Example

  ```python
  >>> from string import Template
  >>> t = Template('Hey, $name!')
  >>> t.substitute(name=name)
  'Hey, Bob!'
  ```

* Template strings do not allow format specifiers 

  ```python
  >>> templ_string = 'Hey $name, there is a $error error!'
  >>> Template(templ_string).substitute(
  ...     name=name, error=hex(errno))
  'Hey Bob, there is a 0xbadc0ffee error!'
  ```

* Why use templates? Possibly use for user input due to security:

  ```python
  >>> SECRET = 'this-is-a-secret'
  >>> class Error:
  ...     def __init__(self):
  ...         pass
  >>> err = Error()
  >>> user_input = '{error.__init__.__globals__[SECRET]}'
  
  # Uh-oh...
  >>> user_input.format(error=err)
  'this-is-a-secret'
  
  >>> user_input = '${error.__init__.__globals__[SECRET]}'
  >>> Template(user_input).substitute(error=err)
  ValueError:
  "Invalid placeholder in string: line 1, col 1"
  ```

#### Dan's Python String Formatting Rule of Thumb

If your format strings are user-supplied, use Template Strings to avoid security issues. Otherwise, use Literal String Interpolation if you’re on Python 3.6+, and “New Style” String Formatting if you’re not.

## Effective Functions

### Python's Functions Are First-Class

* Python functions are first-class

  * assign to variables
  * store them in a data structure
  * pass them to other functions
  * return them as values from functions

* Example that will be used throughout

  ```python
  def yell(text):
      return text.upper() + '!'
  
  >>> yell('hello')
  'HELLO!' 
  ```

#### Functions Are Objects

* All data in a Python program are represented by objects or relations between objects

* Example: Assigning the `yell` function to a variable

  ```python
  bark = yell
  >>> bark('woof')
  'WOOF!'
  ```

  bark is a variable pointing to the function yell

* Function objects and their names are two separate concerns.

  * Can delete `yell` and `bark` still works

* Python attaches a string identifier to every function at creation time

  ```python
  >>> bark.__name__
  'yell'
  ```

* **a variable pointing to a function and the function itself are two separate concerns** 

#### Functions Can Be Stored in Data Structures

```python
>>> funcs = [bark, str.lower, str.capitalize]
>>> funcs
[<function yell at 0x10ff96510>,
 <method 'lower' of 'str' objects>,
 <method 'capitalize' of 'str' objects>]

>>> for f in funcs:
...     print(f, f('hey there'))
<function yell at 0x10ff96510> 'HEY THERE!'
<method 'lower' of 'str' objects> 'hey there'
<method 'capitalize' of 'str' objects> 'Hey there'

>>> funcs[0]('heyho')
'HEYHO!'
```

#### Functions Can Be Passed to Other Functions

```python
def greet(func):
    greeting = func('Hi, I am a Python program')
    print(greeting)

>>> greet(bark)
'HI, I AM A PYTHON PROGRAM!'
```

* Ability to pass function objects as arguments allows one to abstract away and pass around **behavior**. 

* Functions that can accept other functions are also call **higher-order functions**. 

  * They are necessary for the functional programming style

  * Example:

    ```python
    >>> list(map(bark, ['hello', 'hey', 'hi']))
    ['HELLO!', 'HEY!', 'HI!']
    ```

#### Functions Can Be Nested

* Example

  ```python
  def speak(text):
      def whisper(t):
          return t.lower() + '...'
      return whisper(text)
  
  >>> speak('Hello, World')
  'hello, world...'
  ```

  `whisper` does not exist outside speak

* Functions can return functions (i.e. return behaviors)

#### Functions Can Capture Local State

* Inner functions can capture and carry some of the parent function's state

  ```python
  def get_speak_func(text, volume):
      def whisper():
          return text.lower() + '...'
      def yell():
          return text.upper() + '!'
      if volume > 0.5:
          return yell
      else:
          return whisper
  ```

  Notice the inner function captures the text from the outer function

* A **lexical closure** remembers the values from its enclosing lexical scope even when the program flow is no longer in that scope.

  * In practical terms, this means not only can functions return behaviors but they can also pre-configure those behaviors.

  * Example:

    ```python
    def make_adder(n):
        def add(x):
            return x + n
        return add
    
    >>> plus_3 = make_adder(3)
    >>> plus_5 = make_adder(5)
    
    >>> plus_3(4)
    7
    >>> plus_5(4)
    9
    ```

    * `make_adder` serves as a **factory** 

#### Objects Can Behave Like Functions

* Objects can be made **callable** by the `__call__` dunder method

  ```python
  class Adder:
      def __init__(self, n):
           self.n = n
  
      def __call__(self, x):
          return self.n + x
  
  >>> plus_3 = Adder(3)
  >>> plus_3(4)
  7
  ```

  * can use ``callable`` function to determine if an object is callable or not

### Lambdas Are Single-Expression Functions

* Example:

  ```python
  >>> add = lambda x, y: x + y
  >>> add(5, 3)
  8
  
  >>> def add(x, y):
  ...     return x + y
  >>> add(5, 3)
  8
  ```

  * The key difference here is that one does not have to bind the function object to a name before I used it.

    ```python
    >>> (lambda x, y: x + y)(5, 3)
    8
    ```

  * Lambda functions are restricted to a single expression. This means a lambda function can’t use statements or annotations—not even a return statement.

#### Lambdas You Can Use 

```python
>>> tuples = [(1, 'd'), (2, 'b'), (4, 'a'), (3, 'c')]
>>> sorted(tuples, key=lambda x: x[1])
[(4, 'a'), (2, 'b'), (3, 'c'), (1, 'd')]

>>> sorted(range(-5, 6), key=lambda x: x * x)
[0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5]
```

* lambdas also work as lexical closures.

  ```python
  >>> def make_adder(n):
  ...     return lambda x: x + n
  
  >>> plus_3 = make_adder(3)
  >>> plus_5 = make_adder(5)
  
  >>> plus_3(4)
  7
  >>> plus_5(4)
  9
  ```

#### But Maybe You Shouldn't

Lambda functions should be used sparingly and with extraordinary care.

```python
# Harmful:
>>> list(filter(lambda x: x % 2 == 0, range(16)))
[0, 2, 4, 6, 8, 10, 12, 14]

# Better:
>>> [x for x in range(16) if x % 2 == 0]
[0, 2, 4, 6, 8, 10, 12, 14]
```

### The Power of Decorators

* Python's decorators allow you to extend and modify the behavior of a callable (functions, methods, classes) **without** permanently modifying the callable itself.
* Examples: tacking on generic functionality
  * logging
  * enforcing access control and authentication
  * instrumentation and timing functions
  * rate-limiting
  * caching
  * etc.

#### Python Decorator Basics

* A decorator is a callable that takes another callable as input and returns another callable

* Simplest decorator

  ```python
  def null_decorator(func):
      return func
  
  def greet():
      return 'Hello!'
  
  greet = null_decorator(greet)
  
  >>> greet()
  'Hello!'
  ```

* More convenient syntax

  ```python
  @null_decorator
  def greet():
      return 'Hello!'
  
  >>> greet()
  'Hello!'
  ```

  Using the @ syntax is just syntactic sugar

* Note that using the @ syntax decorates the function immediately at definition time. This makes it difficult to access the undecorated original without brittle hacks.

#### Decorators Can Modify Behavior

* Example:

  ```python
  def uppercase(func):
    def wrapper():
      original_result = func()
      modified_result = original_result.upper()
      return modified_result
    return wrapper
  
  @uppercase
  def greet():
    return 'Hello'
  
  >>> greet()
  'HELLO!'
  
  >>> greet
  <function greet at 0x10e9f0950>
  
  >>> null_decorator(greet)
  <function greet at 0x10e9f0950>
  
  >>> uppercase(greet)
  <function uppercase.<locals>.wrapper at 0x76da02f28>
  ```

  * The above example defines a closure and uses it to *wrap* the input function to modify its behavior at call time
  * The `uppercase` decortor returns a different function

#### Applying Multiple Decorators to a Function

* Can apply more than one decorator to a function

* Example:

  ```python
  def strong(func):
      def wrapper():
          return '<strong>' + func() + '</strong>'
      return wrapper
  
  def emphasis(func):
      def wrapper():
          return '<em>' + func() + '</em>'
      return wrapper
  
  @strong
  @emphasis
  def greet():
    return 'Hello'
  
  >>> greet()
  '<strong><em>Hello!</em></strong>'
  ```

  * Ordered applied bottom to top
    * First wrapped by emphasis then strong
    * "decorator stacking"
      * `decorated_greet = strong(emphasis(greet))`

#### Decorating Functions that Accept Arguments

* Use Python's `*args` and `**kwargs` features

  ```python
  def proxy(func):
    def wrapper(*args, **kwargs):
      return func(*args, **kwargs)
    return wrapper
  ```

  * Uses `*` and `**` operators to collect all positional and keyword arguments and store them to variables
  * `wrapper` closure then forwards the collected arguments to the original input function using `*` and `**` "argument unpacking" operators

* Another Example

  ```python
  def trace(func):
      def wrapper(*args, **kwargs):
          print(f'TRACE: calling {func.__name__}() '
                f'with {args}, {kwargs}')
  
          original_result = func(*args, **kwargs)
  
          print(f'TRACE: {func.__name__}() '
                f'returned {original_result!r}')
  
          return original_result
      return wrapper
  
  @trace
  def say(name, line):
      return f'{name}: {line}'
  
  >>> say('Jane', 'Hello, World')
  'TRACE: calling say() with ("Jane", "Hello, World"), {}'
  'TRACE: say() returned "Jane: Hello, World"'
  'Jane: Hello, World'
  ```

#### How to Write "Debuggable" Decorators

* Decorators replace one function with another function. 

  * Decorator hides metadata with the original undecorated function (e.g. original function name, docstring, parameter list)

    ```python
    def greet():
        """Return a friendly greeting."""
        return 'Hello!'
    
    decorated_greet = uppercase(greet)
    
    >>> greet.__name__
    'greet'
    >>> greet.__doc__
    'Return a friendly greeting.'
    
    >>> decorated_greet.__name__
    'wrapper'
    >>> decorated_greet.__doc__
    None
    ```

* Can fix this issue by using `functools.wraps` 

  * copies metadata from the undecorated function to the decorated closure

  ```python
  >>> greet.__name__
  'greet'
  >>> greet.__doc__
  'Return a friendly greeting.'
  
  >>> decorated_greet.__name__
  'wrapper'
  >>> decorated_greet.__doc__
  None
  
  @uppercase
  def greet():
      """Return a friendly greeting."""
      return 'Hello!'
  
  >>> greet.__name__
  'greet'
  >>> greet.__doc__
  'Return a friendly greeting.'
  ```

  * **Recommendation**: Use `functools.wraps` for all decorators. 

### Fun With `*args` and `**kwargs`

* `*args` and `**kwargs` allow a function to accept optional arguments

  * create flexible APIs

  ```python
  def foo(required, *args, **kwargs):
      print(required)
      if args:
          print(args)
      if kwargs:
          print(kwargs)
          
  >>> foo()
  TypeError:
  "foo() missing 1 required positional arg: 'required'"
  
  >>> foo('hello')
  hello
  
  >>> foo('hello', 1, 2, 3)
  hello
  (1, 2, 3)
  
  >>> foo('hello', 1, 2, 3, key1='value', key2=999)
  hello
  (1, 2, 3)
  {'key1': 'value', 'key2': 999}
  ```

  * Above has one required argument: `required` 
  * If additional arguments are provided, `args` will collect the extra positional arguments as a tuple because of the `*` prefix
  * `kwargs` will collect extra keyword arguments as a dictionary because the parameter has a `**` prefix
  * Both `args` and `kwargs` can be empty if no extra arguments are passed to the function.
  * `args` and `kwargs` is a naming convention, can use other variable names

#### Forwarding Optional or Keyword Arguments

* when calling a function, `*` and `**` unpack the positional and keyword arguments

  * As opposed to in a function definition where the arguments are collected into a tuple or dictionary

  ```python
  def trace(f):
      @functools.wraps(f)
      def decorated_function(*args, **kwargs):
          print(f, args, kwargs)
          result = f(*args, **kwargs)
          print(result)
      return decorated_function
  
  @trace
  def greet(greeting, name):
     return '{}, {}!'.format(greeting, name)
  
  >>> greet('Hello', 'Bob')
  <function greet at 0x1031c9158> ('Hello', 'Bob') {}
  'Hello, Bob!'
  ```

### Function Argument Unpacking

* Example:

  ```python
  def print_vector(x, y, z):
    print('<%s, %s, %s>' % (x, y, z))
  
  tuple_vec = (1, 0, 1)
  list_vec = [1, 0, 1]
  dict_vec = {'y': 0, 'z': 1, 'x': 1}
  
  >>> print_vector(*tuple_vec)
  <1, 0, 1>
  >>> print_vector(*list_vec)
  <1, 0, 1>
  >>> print_vector(**dict_vec)
  <1, 0, 1>
  ```

  In the above, the `*` operator unpacked the tuple and the list into the three arguments used by `print_vector`

  * This technique works for any iterable
    * Including generators, in which case all the elemtns are consumed and passed to the function
  * `**` operator unpacks keyword arguments from dictionaries
    * In the above, it unpacks the `x` key to the `x` argument, etc. 
      * If one uses a `*` operator instead, the dictonary keys would be passed to the funciton in random ordered (since dicts are unordered)

### Nothing to Return Here

* Python adds an implicit `return None` statement to the end of any function, so if no return value is specified, the function returns `None` 

  ```python
  def foo1(value):
      if value:
          return value
      else:
          return None
  
  def foo2(value):
      """Bare return statement implies `return None`"""
      if value:
          return value
      else:
          return
  
  def foo3(value):
      """Missing return statement implies `return None`"""
      if value:
          return value
  ```

  * All three functions return `None` if `value` is falsy. 

## Classes & OOP

### Object Comparisons: "is" vs "=="

* `==` checks for *equality*

  * Evaluates to `True` if the objects referred to by the variables are equal (have the same contents)

* `is` compares *identities* 

  * `True` when two variables point to the same (identical) object)

  ```python
  a = [1, 2, 3]
  b = a
  c = list(a) # creates a copy of a
  
  >>> a == b
  True
  
  >>> a is b # a and b point to the same object
  True
  
  >>> a == c
  True
  
  >>> a is c # a and c do not point to the same object
  False
  ```

### String Conversion (Every Class Needs a `__repr__`)

* Trying to print a class without `__str__` or `__repr__` will print the class name and the `id` of the object instance

  ```python
  class Car:
      def __init__(self, color, mileage):
          self.color = color
          self.mileage = mileage
  
      def __str__(self):
          return f'a {self.color} car'
  
  >>> my_car = Car('red', 37281)
  >>> print(my_car)
  'a red car'
  >>> my_car
  <__console__.Car object at 0x109ca24e0>
  ```

  * `__str__` is a "dunder" method that gets called when you try to convert an object to a string through 
    * `print()`
    * str()
    * `{}.format()`
  * Note: In the above the last line still does not print the string

#### `__str__` vs `__repr__`

* Two dunder methods to control how objects are converted to strings in Python 3

  * `__str__`
  * `__repr__`

* Example:

  ```python
  class Car:
      def __init__(self, color, mileage):
          self.color = color
          self.mileage = mileage
  
      def __repr__(self):
          return '__repr__ for Car'
  
      def __str__(self):
          return '__str__ for Car'
  
  >>> my_car = Car('red', 37281)
  >>> print(my_car)
  __str__ for Car
  >>> '{}'.format(my_car)
  '__str__ for Car'
  >>> my_car
  __repr__ for Car
  >>> str([my_car])
  '[__repr__ for Car]'
  ```

  * Notice that containers (e.g. lists, dictionaries) use the result of `__repr__` to represent the objects they contain, even if one uses `str` on the container 
  * For manually choosing representation, use `str` or `repr` instead of calling the dunder method directly

* When to use `__str__` or `__repr__`? 

  * `__str__` should be about *readability*

  * `__repr__` should be about *unambiguity*, such as information needed to debug

  * Example:

    ```python
    >>> import datetime
    >>> today = datetime.date.today()
    
    >>> str(today)
    '2017-02-02'
    >>> repr(today)
    'datetime.date(2017, 2, 2)'
    ```

    * We could copy and paste the string returned by `__repr__` and execute it as valid Python to recreate the original date object. This is a neat approach and a good goal to keep in mind while writing your own reprs.

#### Why Every Class Needs a `__repr__`

* If `__str__` is not implemented, Python falls back to the result of `__repr__`

* **Tip**: Always implemented a `__repr__` method for a class

  * Car Example

    ```python
    def __repr__(self):
      return (f'{self.__class__.__name__}('
              f'{self.color!r}, {self.milage!})')
    ```

    Note the use of `!r` to get the representations of `self.color` and `self.milage`

### Defining Your Own Exception Classes

* Bad Example;

  ```python
  def validation(name):
    if len(name) < 10:
      raise ValueError
  ```

  Not descriptive

* Better Example:

  ```python
  class NameTooShortError(ValueError):
    pass
  
  def validate(name):
    if len(name) < 10:
      raise NameTooShortError(name)
  ```

* It's easier to ask for forgiveness than permission (EAFP) is considered Pythonic

### Cloning Objects for Fun and Profit

* Assignment statements in Python do not create copies of objects

  * they only bind names to an object

* For Python's built-in mutable collections (lists, dicts, sets), copies are made by calling the factory function on an existing collection

  ```python
  new_list = list(original_list)
  new_dict = dict(original_dict)
  new_set = set(original_set)
  ```

  **PROBLEM**: Doesn't work for custom objects and only makes **SHALLOW COPIES**, i.e. creates a new object but populated with references to the child objects found in the original. A **DEEP COPY** is recursive and creates a whole new object. 

* Example

  ```python
  >>> xs = [[1,2,3], [4,5,6], [7,8,9]]
  >>> ys = list(xs) # Make a shallow copy
  
  >>> xs.append(['new sublist'])
  >>> xs
  [[1, 2, 3], [4, 5, 6], [7, 8, 9], ['new sublist']]
  >>> ys
  [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  
  >>> xs[1][0] = 'X'
  >>> xs
  [[1, 2, 3], ['X', 5, 6], [7, 8, 9], ['new sublist']]
  >>> ys
  [[1, 2, 3], ['X', 5, 6], [7, 8, 9]]
  ```

#### Making Deep Copies

Using `copy` module in the Python standard library

```python
>>> import copy
>>> xs = [[1,2,3], [4,5,6], [7,8,9]]
>>> zs = copy.deepcopy(xs)

>>> xs[1][0] = 'X'
>>> xs
[[1, 2, 3], ['X', 5, 6], [7, 8, 9]]
>>> zs
[[1, 2, 3], [4, 5, 6], [7, 8, 9]]
```

* Note: `copy.copy` creates a shallow copy
  * More Pythonic to use factory functions for python mutable collections

#### Copying Arbitrary Objects

`copy.copy` and `copy.deepcopy` can copy arbitrary objects

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f'Point({self.x!r}, {self.y!r})'

>>> a = Point(23, 42)
>>> b = copy.copy(a)
>>> a
Point(23, 42)
>>> b
Point(23, 42)
>>> a is b
False

class Rectangle:
    def __init__(self, topleft, bottomright):
        self.topleft = topleft
        self.bottomright = bottomright

    def __repr__(self):
        return (f'Rectangle({self.topleft!r}, '
                f'{self.bottomright!r})')

rect = Rectangle(Point(0, 1), Point(5, 6))
srect = copy.copy(rect)

>>> rect
Rectangle(Point(0, 1), Point(5, 6))
>>> srect
Rectangle(Point(0, 1), Point(5, 6))
>>> rect is srect
False
>>> rect.topleft.x = 999
>>> rect
Rectangle(Point(999, 1), Point(5, 6))
>>> srect
Rectangle(Point(999, 1), Point(5, 6)) # Changed due to shallow copy
>>> drect = copy.deepcopy(srect)
>>> drect.topleft.x = 222
>>> drect
Rectangle(Point(222, 1), Point(5, 6))
>>> rect
Rectangle(Point(999, 1), Point(5, 6))
>>> srect
Rectangle(Point(999, 1), Point(5, 6))
```

Note because `self.x` and `self.y` are immutable, there's no difference between a shallow and deep copy of `Point`.

### Abstract Base Classes Keep Inheritance in Check

Abstract Base Classes (ABC) ensure that derived classes implement particular methods from the base class. 

```python
from abc import ABCMeta, abstractmethod

class Base(metaclass=ABCMeta):
    @abstractmethod
    def foo(self):
        pass

    @abstractmethod
    def bar(self):
        pass

class Concrete(Base):
    def foo(self):
        pass

    # We forget to declare bar() again...

assert issubclass(Concrete, Base)

>>> c = Concrete()
TypeError:
"Can't instantiate abstract class Concrete with abstract methods bar"
```

TypeError at instantiation time. Without `abc` you'd get an error only when missing method is called. 

### What Namedtuples Are Good For

Regular tuple can only access items via integer indexes. Also ad hoc since it's hard to ensure that two tuples ahve the same fields.

#### Namedtuples to the Rescue

* Immutable like regular tuples

```python
>>> from collections import namedtuple
>>> Car = namedtuple('Car' , 'color mileage') 

# or
>>> Car = namedtuple('Car', [
...     'color',
...     'mileage',
... ]) # passed list of field names

>>> my_car = Car('red', 3812.4)
>>> my_car.color
'red'
>>> my_car.mileage
3812.4

# Also access via indices
>>> my_car[0]
'red'

# tuple unpacking
>>> color, mileage = my_car
>>> print(color, mileage)
red 3812.4
>>> print(*my_car)
red 3812.4

# Immutable
>>> my_car.color = 'blue'
AttributeError: "can't set attribute"
```

* `Car` is the "typename": the name of the new class
* Factory function call `split()` on the field names
* Can view named tuples as a memory-efficient shortcut to defining an immutable class in Python manually

#### Subclassing Namedtuples

Can add methods to a namedtuple object.

```python
Car = namedtuple('Car', 'color mileage')

class MyCarWithMethods(Car):
    def hexcolor(self):
        if self.color == 'red':
            return '#ff0000'
        else:
            return '#000000'

>>> c = MyCarWithMethods('red', 1234)
>>> c.hexcolor()
'#ff0000'
```

#### Built-In Helper Methods

* `_asdict()`: returns contents of the namedtuple as a dictionary

  ```python
  >>> my_car._asdict()
  OrderedDict([('color', 'red'), ('mileage', 3812.4)])
  
  # JSON output
  >>> json.dumps(my_car._asdict())
  '{"color": "red", "mileage": 3812.4}'
  ```

* `_replace()`: creates a shallow copy of a tuple and allows you to selectively replace some of its fields

  ```python
  >>> my_car._replace(color='blue')
  Car(color='blue', mileage=3812.4)
  ```

* `_make()`: create new instances of a namedtuple from a sequence or iterable

  ```python
  >>> Car._make(['red', 999])
  Car(color='red', mileage=999)
  ```

### Class vs. Instance Variable Pitfalls

* Distinction between class methods and instance methods
  * Also a distinction between *class variables* and *instance variables*
* **class variables** are declard inside the class definition (but outside of any instance methods)
  * They are not tied to any particular instance of a class
  * All objects created from a particular class share access to the same set of class variables
    * Modifying a class variable affects all object instances
* **instance variables** are always tied to a particular object instance. 

```python
class Dog:
    num_legs = 4  # <- Class variable

    def __init__(self, name):
        self.name = name  # <- Instance variable

>>> jack = Dog('Jack')
>>> jill = Dog('Jill')
>>> jack.name, jill.name
('Jack', 'Jill')
>>> jack.num_legs, jill.num_legs
(4, 4)
>>> Dog.num_legs
4
>>> Dog.name
AttributeError:
"type object 'Dog' has no attribute 'name'"
>>> jack.num_legs = 6 # Shadows class variable
>>> jack.num_legs, jill.num_legs, Dog.num_legs
(6, 4, 4)
```

* Trying to access an instance variable through the class leads to an `AttributeError`
* The last few lines create a instance variable that shadows the class variable 

#### A Dog-free Example

```python
class CountedObject:
    num_instances = 0

    def __init__(self):
        self.__class__.num_instances += 1
```

`num_instances` is a class variable that serves as a shared counter.

### Instance, Class, and Static Methods Demystified

```python
class MyClass:
    def method(self):
        return 'instance method called', self

		@classmethod
    def classmethod(cls):
        return 'class method called', cls

    @staticmethod
    def staticmethod():
        return 'static method called'
```

#### Instance Methods

In the previous code, `method` is a instance method. Can even modify the class through `self.__class__` attribute

#### Class Methods

Takes a `cls` parameter that points to the class, not the object instance, when it is called. Thus it cannot modify object instance state (which would require access to `self`). Uses the `@classmethod` decorator

#### Static Methods

Uses the `@staticmethod` decorator and does not accept either `self` nor `cls`. Primarily a way to namespace your method as it can't modify the object state or class state.

#### Let's See Them in Action!

Can call a static method on an object instance even though it can't modify the object or class instance

```python
>>> obj.staticmethod()
'static method called'

# Calling methods on class instance
>>> MyClass.classmethod()
('class method called', <class MyClass at 0x11a2>)

>>> MyClass.staticmethod()
'static method called'

>>> MyClass.method()
TypeError: """unbound method method() must
    be called with MyClass instance as first
    argument (got nothing instead)"""
```

#### Delicious Pizza Factories with @classmethod

```python
class Pizza:
    def __init__(self, ingredients):
        self.ingredients = ingredients

    def __repr__(self):
        return f'Pizza({self.ingredients!r})'

		@classmethod
    def margherita(cls):
        return cls(['mozzarella', 'tomatoes'])

    @classmethod
    def prosciutto(cls):
        return cls(['mozzarella', 'tomatoes', 'ham'])

    @staticmethod
    def circle_area(r):
        return r ** 2 * math.pi
      
      
>>> Pizza(['cheese', 'tomatoes'])
Pizza(['cheese', 'tomatoes'])
>>> Pizza.margherita()
Pizza(['mozzarella', 'tomatoes'])

>>> Pizza.prosciutto()
Pizza(['mozzarella', 'tomatoes', 'ham'])
```

`margherita` and `prosciutto` are factory functions taking the `cls` argument to return object instances. Can view as alternate constructors. Python allows only one `__init__` method per class. 

#### When to Use Static Methods

```python
>>> p = Pizza(4, ['mozzarella', 'tomatoes'])
>>> p
Pizza(4, {self.ingredients})
>>> p.area()
50.26548245743669
>>> Pizza.circle_area(4)
50.26548245743669
```

## Common Data Structures in Python

* **ADT** = abstract data type
* Dictionaries are *the* central data structure in Python

### Dictionaries, Maps, and Hashtables

* Dictionaries are also called maps, hashmaps, lookup tables or associative arrays. 

#### `dict` - Your Go-To Dictionary

```python
phonebook = {
    'bob': 7387,
    'alice': 3719,
    'jack': 7052,
}

# dictionary comprehension
squares = {x: x * x for x in range(6)}

>>> phonebook['alice']
3719

>>> squares
{0: 0, 1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
```

* dictionary keys must be hashable. 
  * `__hash__` (hash value never changes during lifetime)
  * `__eq__` 
    * hashable objects which compare as equal must have the same hash value
  * immutable types work well
    * tuples with immutable elements, strings, numbers
* Performance: Expect $O(1)$ lookup, insertion, deletion, update in the average case. 

#### `collections.OrderedDict` - Remember the Insertion Order of Keys

* As of Python 3.7.4, standard dict preserves the insertion order of the keys

```python
>>> import collections
>>> d = collections.OrderedDict(one=1, two=2, three=3)

>>> d
OrderedDict([('one', 1), ('two', 2), ('three', 3)])

>>> d['four'] = 4
>>> d
OrderedDict([('one', 1), ('two', 2),
             ('three', 3), ('four', 4)])

>>> d.keys()
odict_keys(['one', 'two', 'three', 'four'])
```

#### `collections.defaultdict` - Return Default Values for Missing Keys

defaultdict accepts a callable in its constructor whose reutrn value will be used if a requsted key cannot be found. 

```python
>>> from collections import defaultdict
>>> dd = defaultdict(list)

# Accessing a missing key creates it and
# initializes it using the default factory,
# i.e. list() in this example:
>>> dd['dogs'].append('Rufus')
>>> dd['dogs'].append('Kathrin')
>>> dd['dogs'].append('Mr Sniffles')

>>> dd['dogs']
['Rufus', 'Kathrin', 'Mr Sniffles']
```

#### `collection.ChainMap` - Search Multiple Dictionaries as a Single Mapping

Groups multiple dictionaries into a single mapping. 

```python
>>> from collections import ChainMap
>>> dict1 = {'one': 1, 'two': 2}
>>> dict2 = {'three': 3, 'four': 4}
>>> chain = ChainMap(dict1, dict2)

>>> chain
ChainMap({'one': 1, 'two': 2}, {'three': 3, 'four': 4})

# ChainMap searches each collection in the chain
# from left to right until it finds the key (or fails):
>>> chain['three']
3
>>> chain['one']
1
>>> chain['missing']
KeyError: 'missing'
```

#### `types.MappingProxyType` - A Wrapper for Making Read-Only Dictionaries

Provides a read-only view into the wrapped dicitonary's data.

```python
>>> from types import MappingProxyType
>>> writable = {'one': 1, 'two': 2}
>>> read_only = MappingProxyType(writable)

# The proxy is read-only:
>>> read_only['one']
1
>>> read_only['one'] = 23
TypeError:
"'mappingproxy' object does not support item assignment"

# Updates to the original are reflected in the proxy:
>>> writable['one'] = 42
>>> read_only
mappingproxy({'one': 42, 'two': 2})
```

### Array Data Structures

* Arrays consist of fixed-size data records that allow each element to be efficiently located based on its index.
* They are *contiguous* data structures
* $O(1)$ access to an element

#### `list` - Mutable Dynamic Arrays

* Use *dynamic arrays*: when an element is added or removed, the list will automatically adjust the backing store that holds these elements by allocating or releasing memory.
* Python lists can hold arbitrary elements
  * structure takes up more space than since it can hold "anything"

#### `tuple` - Immutable Containers

* Unlike lists, tuples are immutable
  * elements can't be added or removed dynamically
  * all elements in a tuple must be defined at creation time
* Can hold elements of arbitrary type.

#### `array.array` - Basic Typed Arrays

* Provides space-efficient storage of basic C-style data types
  * e.g. 32-bit integers, floating point numbers, etc.
* Mutable
* "typed array": can only hold one data type

```python
import array

>>> arr = array.array('f', (1.0, 1.5, 2.0, 2.5))
>>> arr[1]
1.5

# Arrays have a nice repr:
>>> arr
array('f', [1.0, 1.5, 2.0, 2.5])

# Arrays are mutable:
>>> arr[1] = 23.0
>>> arr
array('f', [1.0, 23.0, 2.0, 2.5])

>>> del arr[1]
>>> arr
array('f', [1.0, 2.0, 2.5])

>>> arr.append(42.0)
>>> arr
array('f', [1.0, 2.0, 2.5, 42.0])

# Arrays are "typed":
>>> arr[1] = 'hello'
TypeError: "must be real number, not str"
```

#### `str`- Immutable Arrays of Unicode Characters

Python 3.x uses `str` objects to store textual data as immutable sequences of Unicode characters.

```python
>>> arr = 'abcd'
>>> arr[1]
'b'

>>> arr
'abcd'

# Strings are immutable:
>>> arr[1] = 'e'
TypeError:
"'str' object does not support item assignment"

>>> del arr[1]
TypeError:
"'str' object doesn't support item deletion"

# Strings can be unpacked into a list to
# get a mutable representation:
>>> list('abcd')
['a', 'b', 'c', 'd']
>>> ''.join(list('abcd'))
'abcd'

# Strings are recursive data structures:
>>> type('abc')
"<class 'str'>"
>>> type('abc'[0])
"<class 'str'>"
```

#### `bytes`- Immutable Arrays of Single Bytes

Bytes are immutable sequences of single bytes (integers between 0 and 255). 

```python
>>> arr = bytes((0, 1, 2, 3))
>>> arr[1]
1
# Bytes literals have their own syntax:
>>> arr
b'\x00\x01\x02\x03'
>>> arr = b'\x00\x01\x02\x03'

# Only valid "bytes" are allowed:
>>> bytes((0, 300))
ValueError: "bytes must be in range(0, 256)"

# Bytes are immutable:
>>> arr[1] = 23
TypeError:
"'bytes' object does not support item assignment"

>>> del arr[1]
TypeError:
"'bytes' object doesn't support item deletion"
```

#### `bytearray`- Mutable Arrays of Single Bytes

A mutable sequence of integers in the range 0 to 255 (bytes).

```python
>>> arr = bytearray((0, 1, 2, 3))
>>> arr[1]
1

# The bytearray repr:
>>> arr
bytearray(b'\x00\x01\x02\x03')

# Bytearrays are mutable:
>>> arr[1] = 23
>>> arr
bytearray(b'\x00\x17\x02\x03')

>>> arr[1]
23

# Bytearrays can grow and shrink in size:
>>> del arr[1]
>>> arr
bytearray(b'\x00\x02\x03')

>>> arr.append(42)
>>> arr
bytearray(b'\x00\x02\x03*')

# Bytearrays can only hold "bytes"
# (integers in the range 0 <= x <= 255)
>>> arr[1] = 'hello'
TypeError: "an integer is required"

>>> arr[1] = 300
ValueError: "byte must be in range(0, 256)"

# Bytearrays can be converted back into bytes objects:
# (This will copy the data)
>>> bytes(arr)
b'\x00\x02\x03*'
```

#### Summary

* Need to store arbitrary objects potentially with mixed data types: use a list or tuple depending on mutability
* Numerical data and tight packing and performance are important: use array.array or numpy or pandas
* Textual representation as Unicode characters: use str unless you need a "mutable string" in which case use an array of characters
* Want a contiguous block of bytes: use bytes or byte array

### Records, Structs, and Data Transfer Objects

#### `dict` - Simple Data Objects

Possible issue is that fields can be added and removed at any time.

#### `tuple` - Immutable Groups of Objects

Possible issue is you can only access data stored in a tuple through integer indices

#### Writing a Custom Class - More Work, More Control

```python
class Car:
    def __init__(self, color, mileage, automatic):
        self.color = color
        self.mileage = mileage
        self.automatic = automatic

>>> car1 = Car('red', 3812.4, True)
>>> car2 = Car('blue', 40231.0, False)

# Get the mileage:
>>> car2.mileage
40231.0

# Classes are mutable:
>>> car2.mileage = 12
>>> car2.windshield = 'broken'

# String representation is not very useful
# (must add a manually written __repr__ method):
>>> car1
<Car object at 0x1081e69e8>
```

#### `collections.namedtuple` - Convenient Data Objects

```python
>>> from collections import namedtuple
>>> Car = namedtuple('Car' , 'color mileage automatic')
>>> car1 = Car('red', 3812.4, True)

# Instances have a nice repr:
>>> car1
Car(color='red', mileage=3812.4, automatic=True)
# Accessing fields:
>>> car1.mileage
3812.4

# Fields are immtuable:
>>> car1.mileage = 12
AttributeError: "can't set attribute"
>>> car1.windshield = 'broken'
AttributeError:
"'Car' object has no attribute 'windshield'"
```

#### `typing.NamedTuple` - Improved Namedtuples

* Type annotations are not enforced without a separate type-checking tool lik mypy. 

```python
>>> from typing import NamedTuple

class Car(NamedTuple):
    color: str
    mileage: float
		automatic: bool

>>> car1 = Car('red', 3812.4, True)

# Instances have a nice repr:
>>> car1
Car(color='red', mileage=3812.4, automatic=True)

# Accessing fields:
>>> car1.mileage
3812.4

# Fields are immutable:
>>> car1.mileage = 12
AttributeError: "can't set attribute"
>>> car1.windshield = 'broken'
AttributeError:
"'Car' object has no attribute 'windshield'"

# Type annotations are not enforced without
# a separate type checking tool like mypy:
>>> Car('red', 'NOT_A_FLOAT', 99)
Car(color='red', mileage='NOT_A_FLOAT', automatic=99)
```

#### `struct.Struct` - Serialized C Structs

#### `types.SimpleNamespace` - Fancy Attribute Access

### Sets and Multisets

* A *set* is an unordered collection of objects that does not allow duplicates
  * test for membership, insert, delete: $O(1)$
  *  union, intersections: $O(n)$
* To create an empty set: `set()` 

#### `set` - Your Go-To Set

* Sets are mutable and allow dynamic insertions and deletions
* Any hashable object can be stored in a set

#### `frozenset`- Immutable Sets

* Immutable version of set
  * Cannot be altered after construction. (No insertions or deletions)
  * They are hashable and can be used as keys in a dictionary or as objects in another set. 

#### `collections.Counter` - Multisets

* Use if you want to keep track of how many times something is included in a set

```python
>>> from collections import Counter
>>> inventory = Counter()

>>> loot = {'sword': 1, 'bread': 3}
>>> inventory.update(loot)
>>> inventory
Counter({'bread': 3, 'sword': 1})
>>> more_loot = {'sword': 1, 'apple': 1}
>>> inventory.update(more_loot)
>>> inventory
Counter({'bread': 3, 'sword': 2, 'apple': 1})

>>> len(inventory)
3  # Unique elements

>>> sum(inventory.values())
6  # Total no. of elements
```

### Stacks (LIFOs)

