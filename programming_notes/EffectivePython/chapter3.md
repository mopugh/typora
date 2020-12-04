# Chapter 3: Functions

Functions enable you to break large programs into smaller, simplier pieces with names to represent their intent.

## Item 19: Never Unpack More Than Three Variables When Functions Return Multiple Values

```python
def get_avg_ratio(numbers):
    average = sum(numbers) / len(numbers)
    scaled = [x / average for x in numbers]
    scaled.sort(reverse=True)
    return scaled

longest, *middle, shortest = get_avg_ratio(lengths) # *middle unpacks entries of list

print(f'Longest: {longest:>4.0%}')
print(f'Shortest: {shortest:>4.0%}')

>>>
Longest:  108%
Shortest:  89%
```

```python
# Bad Example!!!
def get_stats(numbers):
    minimum = min(numbers)
    maximum = max(numbers)
    count = len(numbers)
    average = sum(numbers) / count

    sorted_numbers = sorted(numbers)
    middle = count // 2
    if count % 2 == 0:
        lower = sorted_numbers[middle - 1]
        upper = sorted_numbers[middle]
        median = (lower + upper) / 2
    else:
        median = sorted_numbers[middle]

    return minimum, maximum, average, median, count

minimum, maximum, average, median, count = get_stats(lengths) # So much unpacking!

print(f'Min: {minimum}, Max: {maximum}')
print(f'Average: {average}, Median: {median}, Count {count}')

>>>
Min: 60, Max: 73
Average: 67.5, Median: 68.5, Count 10
```

* Never use more than three variables when unpacking the multiple return values from a function.
  * If more are required consider:
    * a class
    * `namedtuple`

## Item 20: Prefer Raising Exceptions to Returning `None`

* Be careful of `False`-equivalent values, i.e. `None` and `0` 

  ```python
  # Use exception rather than returning None
  def careful_divide(a, b):
      try:
          return a / b
      except ZeroDivisionError as e:
          raise ValueError('Invalid inputs')
          
  x, y = 5, 2
  try:
      result = careful_divide(x, y)
  except ValueError:
      print('Invalid inputs')
  else:
      print('Result is %.1f' % result)
  
  >>>
  Result is 2.5
  
  # Using types
  def careful_divide(a: float, b: float) -> float:
      """Divides a by b.
  
      Raises:
          ValueError: When the inputs cannot be divided.
      """
  
      try:
          return a / b
      except ZeroDivisionError as e:
          raise ValueError('Invalid inputs')
  ```

  ## Item 21: Know How Closures Interact with Variable Scope

  ```python
  # Example
  def sort_priority(values, group):
      def helper(x): # closure! Able to access group
        	if x in group:
              return (0, x)
          return (1, x)
      values.sort(key=helper)
  
  numbers = [8, 3, 1, 2, 5, 4, 7, 6]
  group = {2, 3, 5, 7}
  sort_priority(numbers, group)
  print(numbers)
  
  >>>
  [2, 3, 5, 7, 1, 4, 6, 8] 
  ```

  * **Closures**: functions that refer to variables from the scope in which they were defined.

  * Python functions are **first-class**: we passed `sort` the `helper` function

    * Functions can be assigned to variables, passed to other functions, etc.

  * For comparing sequences, Python first compres the first elements and if they are equal, compares the second elements, and so on.

    * This is why the returned tuple from `helper` works

    ```python
    def sort_priority2(numbers, group):
        found = False        # Scope: 'sort_priority2'
        def helper(x):
            if x in group:
                found = True # Scope: 'helper' -- Bad!
                return (0, x)
            return (1, x)
    
        numbers.sort(key=helper)
        return found
    
    # Called scoping bug
    
    # Use nonlocal
    def sort_priority3(numbers, group):
        found = False
        def helper(x):
            nonlocal found # Added
            if x in group:
                found = True
                return (0, x)
            return (1, x)
        numbers.sort(key=helper)
        return found
    
    # Be careful of using nonlocal
    # Consider using a class
    class Sorter:
        def __init__(self, group):
    
            self.group = group
            self.found = False
    
        def __call__(self, x):
            if x in self.group:
                self.found = True
                return (0, x)
        return (1, x)
      
      sorter = Sorter(group)
    numbers.sort(key=sorter)
    assert sorter.found is True
    ```

    * Scoping prevents local variables from polluting the module 
    * Use `nonlocal` to get out of closure
      * Does not go upto module level

    ## Item 22: Reduce Visual Noise with Variable Positional Arguments

    * positional arguments also called *varargs* or *star args* 

      ```python
      # Example
      def log(message, values):
          if not values:
              print(message)
          else:
              values_str = ', '.join(str(x) for x in values)
              print(f'{message}: {values_str}')
      
      log('My numbers are', [1, 2])
      log('Hi there', [])
      
      >>>
      My numbers are: 1, 2
      Hi there
      
      # Use varargs
      def log(message, *values): # The only difference
          if not values:
             print(message)
          else:
              values_str = ', '.join(str(x) for x in values)
             print(f'{message}: {values_str}')
      
      log('My numbers are', 1, 2)
      log('Hi there') # Much better
      
      >>>
      My numbers are: 1, 2
      Hi there
      ```

    ## Item 23: Provide Optional Behavior with Keyword Arguments

    * Keyword arguments can be passed in any order as long as all the required positional arguments are specified.

      ```python
      def remainder(number, divisor):
          return number % divisor
      
      remainder(20, 7)
      remainder(20, divisor=7)
      remainder(number=20, divisor=7)
      remainder(divisor=7, number=20)
      
      # Keyword arguments must be after positional arguments
      remainder(number=20, 7)
      
      >>>
      Traceback ...
      SyntaxError: positional argument follows keyword argument
        
      # Each argument can be specified only once
      remainder(20, number=7)
      
      >>>
      Traceback ...
      TypeError: remainder() got multiple values for argument 'number'
        
      # Can pass in dictionary using ** operator
      my_kwargs = {
          'number': 20,
          'divisor': 7,
      }
      assert remainder(**my_kwargs) == 6
      
      ## Can receive named keyword arguments using the **kwargs catch-all parameter
      def print_parameters(**kwargs):
          for key, value in kwargs.items():
              print(f'{key} = {value}')
      
      print_parameters(alpha=1.5, beta=9, gamma=4)
      
      >>>
      alpha = 1.5
      beta = 9
      gamma = 4
      ```

      * Can set default values in function definition
      * Best practice to specify optional arguments using keyword names and not pass them as positional arguments
      * Function arguments can be specified by position or by keyword

    ## Item 24: Use `None` and Docstrings to Specify Dynamic Default Arguments

    ```python
    from time import sleep
    from datetime import datetime
    
    # this doesn't work because datetime.now() is only executed when the function is defined
    def log(message, when=datetime.now()):
        print(f'{when}: {message}')
    
    log('Hi there!')
    sleep(0.1)
    log('Hello again!')
    
    >>>
    2019-07-06 14:06:15.120124: Hi there!
    2019-07-06 14:06:15.120124: Hello again!
          
    # Try this using None
    def log(message, when=None):
        """Log a message with a timestamp.
    
        Args:
            message: Message to print.
            when: datetime of when the message occurred.
                Defaults to the present time.
        """
        if when is None:
            when = datetime.now()
        print(f'{when}: {message}')
    ```

    * Default arguments in functions are only evaluated once per module load

    ## Item 24: Enforce Clarity with Keyword-Only and Positional-Only Arguments

    