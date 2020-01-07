# Python

## Python Basics
``` python
17 / 3                      # 5.666667
17 // 3                     # 5
5 ** 2                      # 25

str1 = r'raw string \n will not be escapted'
str2 = """\
multiline
string
"""
3 * 'a' + 'b'               # aaab
word[0] = 'J'               # Error, strings are immutable

range(0, 10, 3)             # 0, 3, 6, 9
print(range(10))            # range(0, 10), this is an iterable object
list(range(5))              # [0, 1, 2, 3, 4]

'''
Loop statements may have an else clause.
It is executed when the loop terminates through exhaustion of the list (with for)
or when the condition becomes false (with while),
but not when the loop is terminated by a break statement.
'''
for n in range(2, 10):
    for x in range(2, n):
        if n % x == 0:
            print(n, 'equals', x, '*', n//x)
            break
    else:
        # loop fell through without finding a factor
        print(n, 'is a prime number')

'''
Below code prints 5.
The default values are evaluated at the point of function definition in the defining scope.
'''
i = 5
def f(arg=i):
    print(arg)
i = 6
f()

'''
As list is an mutable object, below code will print:
[1]
[1, 2]
'''
def f(a, L=[]):
    L.append(a)
    return L
print(f(1))
print(f(2))

'''
Write the function as below to avoid the default to be shared between subsequent calls.
'''
def f(a, L=None):
    if L is None:
        L = []
    L.append(a)
    return L

'''
Function Arguments
'''
def no_side_effects(cities):
    cities = cities + ["Birmingham", "Bradford"]
locations = ["London", "Leeds", "Glasgow", "Sheffield"]
no_side_effects(locations) # ['London', 'Leeds', 'Glasgow', 'Sheffield']

def side_effects(cities):
    cities += ["Birmingham", "Bradford"]
locations = ["London", "Leeds", "Glasgow", "Sheffield"]
# ['London', 'Leeds', 'Glasgow', 'Sheffield', 'Birmingham', 'Bradford']
side_effects(locations)

'''
Unpacking arguments
'''
def concat(*args, sep="/"):
    return sep.join(args)
concat("earth", "mars", "venus", sep=".")       # 'earth.mars.venus'

args = [3, 6]
list(range(*args))                              # [3, 4, 5]

def parrot(voltage, state='a stiff', action='voom'):
    print("-- This parrot wouldn't", action, end=' ')
    print("if you put", voltage, "volts through it.", end=' ')
    print("E's", state, "!")
d = {"voltage": "four million", "state": "bleedin' demised", "action": "VOOM"}
parrot(**d)

'''
Lambda Expressions can be used to create small anonymous functions
'''
def make_incrementor(n):
    return lambda x: x + n
f = make_incrementor(42)
f(0)                                        # 42

'''
Map, Filter, Reduce
'''
map(func, seq)
odd_numbers = list(filter(lambda x: x % 2, numbers))
reduce(lambda a,b: a if (a > b) else b, [47,11,42,102,13])  # 102
```

## Data Structures
``` python
# Add an item to the end of the list. Equivalent to a[len(a):] = [x].
list.append(x)

# Extend the list by appending all the items from the iterable.
# Equivalent to a[len(a):] = iterable.
list.extend(iterable)

# Insert an item at a given position.
# The first argument is the index of the element before which to insert,
# so a.insert(0, x) inserts at the front of the list,
# and a.insert(len(a), x) is equivalent to a.append(x).
list.insert(i, x)

# Remove the first item from the list whose value is equal to x.
# It raises a ValueError if there is no such item.
list.remove(x)

# Remove the item at the given position in the list, and return it.
# If no index is specified, a.pop() removes and returns the last item in the list.
# (The square brackets around the i in the method signature denote that
# the parameter is optional, not that you should type square brackets at that position.)
list.pop([i])

# Remove all items from the list. Equivalent to del a[:].
list.clear()

# Return zero-based index in the list of the first item whose value is equal to x.
# Raises a ValueError if there is no such item.
list.index(x[, start[, end]])

# Return the number of times x appears in the list.
list.count(x)

# Sort the items of the list in place (the arguments can be used for
# sort customization, see sorted() for their explanation).
list.sort(key=None, reverse=False)

# Reverse the elements of the list in place.
list.reverse()

# Return a shallow copy of the list. Equivalent to a[:].
# In shallow copy, if any of the fields of the object are
# references to other objects, just the reference addresses are copied
list.copy()

'''
Using Lists as Stacks - last-in, first-out
'''
stack = [3, 4, 5]
stack.append(6)                             # [3, 4, 5, 6]
stack.pop()                                 # 6
stack.pop(i)

'''
Using Lists as Queues - first-in, first-out
'''
from collections import deque
queue = deque(["Eric", "John", "Michael"])
queue.append("Terry")                       # Terry arrives
queue.popleft()                             # "Eric"

'''
List Comprehensions
'''
squares = list(map(lambda x: x**2, range(10)))
squares = [x**2 for x in range(10)]
[(x, y) for x in [1,2,3] for y in [3,1,4] if x != y]

matrix = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
]
[[row[i] for row in matrix] for i in range(4)]      # equivalent to list(zip(*matrix))

'''
The del statement
'''
a = [1, 2, 3, 4, 5, 6]
del a[2:4]                                  # [1, 2, 5, 6]
del a

'''
Tuples - immutable
'''
v = ([1, 2, 3], [3, 2, 1])
v[0][1]=22                      # tuples can contain mutable objects in them

'''
Sets - unordered collection with no duplicate elements
'''
basket = {'apple', 'orange', 'apple', 'pear', 'orange', 'banana'}
print(basket)                   # {'orange', 'banana', 'pear', 'apple'}
'orange' in basket              # True
a = set('abracadabra')          # Unique letters in a {'a', 'r', 'b', 'c', 'd'}
b = set('alacazam')
a - b                           # letters in a but not in b
a | b                           # letters in a or b or both
a & b                           # letters in both a and b
a ^ b                           # letters in a or b but not both

'''
Dict - Only strings, numbers, or tuples (of str or num) can be keys
'''
tel = {'jack': 4098, 'guido': 4127, 'sape': 4139}
tel['jack']                     # 4098, raise Error if key does not exist
tel.get('jack')                 # 4098, return None if key does not exist
list(tel)                       # ['jack', 'guido', 'sape']
sorted(tel)                     # ['guido', 'jack', 'sape']
'guido' in tel                  # True
dict([('sape', 4139), ('guido', 4127), ('jack', 4098)])
# {'sape': 4139, 'guido': 4127, 'jack': 4098}, same as dict
(sape=4139, guido=4127, jack=4098)
{x: x**2 for x in (2, 4, 6)}    # {2: 4, 4: 16, 6: 36}
dict1.update(dict2)
# merge keys and values in dict2 into dict1, overwritting the existed keys

'''
Looping Techniques
'''
for k, v in dict.items():       # loop on dict
    pass
for i, v in enumerate([1, 2, 3]):       # loop on list
    pass
for q, a in zip(sequence1, sequence2):
# loop on two sequences; zip() returns a iterator,
# which exhaust itself after use - become empty after used for one time
    pass
for i in reversed(range(1, 10, 2)):     # loop on reversed order
    pass

'''
Comparing Sequences - first the first two items are compared,
and if they differ this determines the outcome of the comparison;
if they are equal, the next two items are compared, and so on,
until either sequence is exhausted
'''
'ABC' < 'C' < 'Pascal' < 'Python'
(1, 2, 3) == (1.0, 2.0, 3.0)
(1, 2, ('aa', 'ab')) < (1, 2, ('abc', 'a'), 4)
```

## Deep Copy and Shallow Copy
``` python
# copy a list - creating a reference to the same object
lst1 = ['a','b',['ab','ba']]
list2 = list1
# two variables references to the same object, update in one list will impact the other one

# shallow copy - creating references to the objects inside the object
list2 = list1[:]    # copied the first two elements but not the 3rd one
dict1 = dict2. copy()   # this is shallow copy as well

# deep copy - copy the entire object, including nested objects
from copy import deepcopy 
lst2 = deepcopy(lst1)
```

## Open Files
``` python
# It is good practice to use the with keyword when dealing with file objects. 
# The advantage is that the file is properly closed after its suite finishes,
# even if an exception is raised at some point.
# 'r' when the file will only be read
# 'w' for only writing (an existing file with the same name will be erased)
# 'a' opens the file for appending
# 'r+' opens the file for both reading and writing
with open('workfile', 'w') as f:
    read_data = f.read()        # return as a string
    f.readline()                # read a single line from the file
    f.readlines()               # read all lines
    for line in f:              # loop on lines, memory efficient and fast
        pass
```

## Errors and Exceptions
``` python
'''
Try Except Else - The use of the else clause is better than
adding additional code to the try clause because it avoids
accidentally catching an exception that wasn’t raised
by the code being protected by the try … except statement.
'''
try:
    f = open(arg, 'r')
except OSError:
    print('cannot open', arg)
else:                               # run if no except is raised
    print(arg, 'has', len(f.readlines()), 'lines')
    f.close()

try:
    raise NameError('HiThere')
except NameError:
    print('An exception flew by!')
    raise                           # reraise the same exception

'''
User-defined Exceptions
'''
class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class InputError(Error):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message

'''
Finally - the finally clause will execute as the last task before the try statement completes
1. If an exception occurs during execution of the try clause,
the exception may be handled by an except clause.
If the exception is not handled by an except clause,
the exception is re-raised after the finally clause has been executed.

2. An exception could occur during execution of an except or else clause.
Again, the exception is re-raised after the finally clause has been executed.

3. If the try statement reaches a break, continue or return statement,
the finally clause will execute just prior to the break,
continue or return statement’s execution.

4. If a finally clause includes a return statement,
the finally clause’s return statement will execute before,
and instead of, the return statement in a try clause.
'''
try:
    raise KeyboardInterrupt
finally:
    print('Goodbye, world!')

# Goodbye, world!
# KeyboardInterrupt
# Traceback (most recent call last):
#   File "<stdin>", line 2, in <module>

def bool_return():
    try:
        return True
    finally:
        return False
bool_return()                   # False

def divide(x, y):
    try:
        result = x / y
    except ZeroDivisionError:
        print("division by zero!")
    else:
        print("result is", result)
    finally:
        print("executing finally clause")
divide(2, 1)
# result is 2.0
# executing finally clause
divide(2, 0)
# division by zero!
# executing finally clause
divide("2", "1")
# executing finally clause
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
#   File "<stdin>", line 3, in divide
# TypeError: unsupported operand type(s) for /: 'str' and 'str'
```

## Scopes and Namespaces
``` python
def scope_test():
    def do_local():
        spam = "local spam"

    def do_nonlocal():
        nonlocal spam       # used in nested functions
        spam = "nonlocal spam"

    def do_global():
        global spam
        spam = "global spam"

    spam = "test spam"
    do_local()              # spam == "test spam"
    do_nonlocal()           # spam == "nonlocal spam"
    do_global()             # spam == "nonlocal spam"

scope_test()                # spam == "global spam"
```

## Decorators
``` python
'''
Simple example, below code will print:
Before calling succ
11
After calling succ
'''
def our_decorator(func):
    def function_wrapper(x):
        print("Before calling " + func.__name__)
        res = func(x)
        print(res)
        print("After calling " + func.__name__)
    return function_wrapper

@our_decorator
def succ(n):
    return n + 1

succ(10)

'''
Arguments in Decorator - 
Hi, foo returns:
42
'''
def greeting(expr):
    def greeting_decorator(func):
        def function_wrapper(x):
            print(expr + ", " + func.__name__ + " returns:")
            func(x)
        return function_wrapper
    return greeting_decorator

@greeting("Hi")
def foo(x):
    print(42)

foo("Hi")
```

## Classes
``` python
'''
Class and Instance Variables
'''
class Dog:
    tricks = []                     # mistaken use of a class variable
    def __init__(self, name):
        self.name = name
    def add_trick(self, trick):
        self.tricks.append(trick)
d = Dog('Fido')
e = Dog('Buddy')
d.add_trick('roll over')            # unexpectedly shared by all dogs
e.tricks                            # ['roll over']
d.tricks = ['new']                  # assigning new value to d.tricks, will not impact e.tricks

# correct design
class Dog:
    def __init__(self, name):
        self.name = name
        self.tricks = []            # creates a new empty list for each dog
    def add_trick(self, trick):
        self.tricks.append(trick)

'''
Data Encapsulation - the bundling of data with the methods that operate on that data.
Data Hiding - some internal information or data is "hidden", so that it can't be accidentally changed.
Data Abstraction - Data Encapsulation + Data Hiding.

name: private attributes, should only be used by the owner, i.e. inside of the class definition itself.
_name: protected attributes, should only be used under certain conditions.
__name: public attributes, can and should be freely used.
'''
# Data Encapsulation Example - encapsulate "__name"
class Robot:
    def __init__(self, name=None):
        self.set_name(name)   

    def say_hi(self):
        if self.__name:
            print("Hi, I am " + self.__name)
        else:
            print("Hi, I am a robot without a name")

    def set_name(self, name):
        self.__name = name
        
    def get_name(self):
        return self.__name

# properties for setter and getter
class Robot:
    def __init__(self, val=None):
        self.name = val         # This is calling the setter

    @property
    def name(self):             # Two methods with the same name, but different arguments
        return self.__name

    @name.setter    
    def name(self, val):
        self.__name = val

'''
Class Methods, Static Methods and Instance Methods
'''
# Instance Methods - have to create instances to know the class information
class Pet:
    _class_info = "pet animals"

    def about(self):
        print("This class is about " + self._class_info + "!")   

class Dog(Pet):
    _class_info = "man's best friends"

p = Pet()
p.about()           # This class is about pet animals!
Dog.about(Dog)      # This class is about man's best friends!

# Static Methods
class Pet:
    _class_info = "pet animals"

    @staticmethod
    def about():
        print("This class is about " + Pet._class_info + "!")   

class Dog(Pet):
    _class_info = "man's best friends"

Pet.about()         # This class is about pet animals!
Dog.about()         # This class is about pet animals!

# Class Methods
class Pet:
    _class_info = "pet animals"

    @classmethod
    def about(cls):
        print("This class is about " + cls._class_info + "!")  

class Dog(Pet):
    _class_info = "man's best friends"

Pet.about()         # This class is about pet animals!
Dog.about()         # This class is about man's best friends!

'''
Inheritance
override - method in child class overwrites the method in parent class with the same name
overload - not needed in python
'''
class Robot:
    def __init__(self, name):
        self.name = name

    def say_hi(self):
        print("Hi, I am " + self.name)

class PhysicianRobot(Robot):
    def say_hi(self):       # override
        super().say_hi()
        print("and I am a physician!")

'''
Use of super()
MRO - method resolution order, break the inheritance tree structure into a linear order
'''
# m of D called
# m of B called
# m of A called
# m of C called
# m of A called
class A:
    def m(self):
        print("m of A called")

class B(A):
    def m(self):
        print("m of B called")
        A.m(self)
    
class C(A):
    def m(self):
        print("m of C called")
        A.m(self)

class D(B,C):
    def m(self):
        print("m of D called")
        B.m(self)
        C.m(self)
x = D()
x.m()

# m of D called
# m of B called
# m of C called
# m of A called
class A:
    def m(self):
        print("m of A called")

class B(A):
    def m(self):
        print("m of B called")
        super().m()
    
class C(A):
    def m(self):
        print("m of C called")
        super().m()

class D(B,C):
    def m(self):
        print("m of D called")
        super().m()
x = D()
x.m()

# Example
import random
 
class Robot():
    __illegal_names = {"Henry", "Oscar"}
    __crucial_health_level = 0.6
 
    def __init__(self, name):
        self.name = name  #---> property setter
        self.health_level = random.random()
 
    @property
    def name(self):
        return self.__name
 
    @name.setter
    def name(self, name):
        if name in Robot.__illegal_names:
            self.__name = "Marvin"
        else:
            self.__name = name
 
    def __str__(self):
        return self.name + ", Robot"
 
    def __add__(self, other):
        first = self.name.split("-")[0]
        second = other.name.split("-")[0]
        return type(self)(first + "-" + second)         # return the proper type after add for child classes
     
    def needs_a_nurse(self):
        if self.health_level < Robot.__crucial_health_level:
            return True
        else:
            return False
 
    def say_hi(self):
        print("Hi, I am " + self.name)
        print("My health level is: " + str(self.health_level))

class NursingRobot(Robot):
    def __init__(self, name="Hubert", healing_power=None):
        super().__init__(name)
        if healing_power:
            self.healing_power = healing_power
        else:
            self.healing_power = random.uniform(0.8, 1)
    
    def say_hi(self):
        print("Well, well, everything will be fine ... " + self.name + " takes care of you!")
 
    def say_hi_to_doc(self):
        Robot.say_hi(self)
 
    def heal(self, robo):
        if robo.health_level > self.healing_power:
            print(self.name + " not strong enough to heal " + robo.name)
        else:
            robo.health_level = random.uniform(robo.health_level, self.healing_power)
            print(robo.name + " has been healed by " + self.name + "!")

'''
Abstract Methods - classes cannot be instantiated without implement the abstract class
'''
class AbstractClassExample(ABC):
 
    def __init__(self, value):
        self.value = value
        super().__init__()
    
    @abstractmethod
    def do_something(self):
        pass

class DoAdd42(AbstractClassExample):
    pass

x = DoAdd42(4)          # raise an error as the abstract class is not implemented

class DoAdd42(AbstractClassExample):
    def do_something(self):
        return self.value + 42

x = DoAdd42(10)         # correct implementation

'''
Slots - prevent the dynamic creation of attributes
'''
class S(object):
    __slots__ = ['val']         # defines the attributes can be created for the class

    def __init__(self, v):
        self.val = v

x = S(42)
print(x.val)                    # 42
x.new = "not possible"          # raise an error

'''
__iter__: Create a Iterable Class
'''
class Reverse:
    """
    Creates Iterators for looping over a sequence backwards.
    """
    
    def __init__(self, data):
        self.data = data
        self.index = len(data)
    
    # If the class contains a __next__, it is enough for the __iter__ method to return self
    def __iter__(self):
        return self

    def __next__(self):
        if self.index == 0:
            raise StopIteration
        self.index = self.index - 1
        return self.data[self.index]

lst = [34, 978, 42]
lst_backwards = Reverse(lst)        # Create an iterable object
for el in lst_backwards:
    print(el)

'''
__call__
'''
class A:
    def __init__(self):
        print("An instance of A was initialized")
    
    def __call__(self, *args, **kwargs):
        print("Arguments are:", args, kwargs)
              
x = A()
x(3, 4, x=11, y=10)     # Arguments are: (3, 4) {'x': 11, 'y': 10}

'''
__str__ and __repr__
str will call on __str__ or falls back on __repr__ if __str__ does not exist;
repr will only call on __repr__;
repr: obj == eval(repr(obj))
'''
class Robot:
    def __init__(self, name, build_year):
        self.name = name
        self.build_year = build_year

    def __repr__(self):
        return "Robot(\"" + self.name + "\"," +  str(self.build_year) +  ")"

    def __str__(self):
        return "Name: " + self.name + ", Build Year: " +  str(self.build_year)
```

## Recursive Function
``` python
'''
Fibonacci sequence - fib2 is much more efficient than fib1
'''
def fib1(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib(n-1) + fib(n-2)

def fib2(n):
    old, new = 0, 1
    if n == 0:
        return 0
    for i in range(n-1):
        old, new = new, old + new
    return new

'''
Pascal's triangle
'''
def pascal(n):
    if n == 1:
        return [1]
    else:
        p_line = pascal(n-1)
        line = [ p_line[i]+p_line[i+1] for i in range(len(p_line)-1)]
        line.insert(0,1)
        line.append(1)
    return line

'''
Finding all prime numbers up to a specified integer
'''
from math import sqrt

def primes(n):
    if n == 0:
        return []
    elif n == 1:
        return []
    else:
        p = primes(int(sqrt(n)))
        no_p = [j for i in p for j in range(i*2, n + 1, i)]
        p = [x for x in range(2, n + 1) if x not in no_p]
        return p
```
