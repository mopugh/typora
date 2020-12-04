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

