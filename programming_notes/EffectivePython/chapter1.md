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

