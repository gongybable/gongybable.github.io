# Python

1. Max Pairwise Product
``` python
def max_pairwise_product(numbers):
    n = len(numbers)
    max1 = -1
    max2 = -1
    for i in range(n):
        if i == 0:
            max1 = 0
        elif i == 1:
            if numbers[max1] > numbers[i]:
                max2 = i
            else:
                max2 = max1
                max1 = i
        else:
            if numbers[i] >= numbers[max1]:
                max2 = max1
                max1 = i
            elif numbers[i] >= numbers[max2]:
                max2 = i
    return numbers[max1] * numbers[max2]
```

2. Fibonacci sequence
``` python
def fib2(n):
    old, new = 0, 1
    if n == 0:
        return 0
    for i in range(n-1):
        old, new = new, old + new
    return new


#For a fibonacci sequence, get a new sequence from Fn % m
#we can see that it is a pisano period sequence starts with 0, 1.

def get_fibonacci_huge(n, m):
    a = 0
    b = 1
    pisano_period_length = 0
    while not (a == 0 and b == 1 and pisano_period_length > 0):
        a, b = b, (a+b)%m
        pisano_period_length += 1

    pisano_period_index = n % pisano_period_length

    if pisano_period_index <= 1:
        return pisano_period_index

    previous = 0
    current  = 1
    for i in range(pisano_period_index-1):
        previous, current = current, (previous + current)%m

    return current


#Sum of fibonacci sequence:
#F0 + F1 + F2 + ... Fn = F(n+2) - 1
#F0^2 + F1^2 + F2^2 + ... Fn^2 = FnF(n+1)
```

3. Greatest Common Divisor
``` python
def gcd(a, b):
    while (a != 0 and b != 0):
        if a >= b:
            return gcd(a%b, b)
        else:
            return gcd(b%a, a)
    if a == 0:
        return b
    if b == 0:
        return a
```

4. Greedy Algorithm
``` python
# maximize the value for given capacity
def get_optimal_value(capacity, weights, values):
    value = 0.
    obj = dict()
    for i in range(len(weights)):
        obj[i] = values[i] / weights[i]

    sorted_obj = sorted(obj, key=obj.get, reverse=True)

    for i in range(len(weights)):
        item = sorted_obj[i]
        item_weight = weights[item]
        item_value = values[item]

        if capacity >= item_weight:
            capacity = capacity - item_weight
            value =  value + item_value
        else:
            value = value + item_value / item_weight * capacity
            return value

    return value

# Car Fueling problem
def compute_min_refills(distance, tank, stops):
    n = 0
    current_spot = -1
    current_distance = 0
    stops.append(distance)

    i = 0
    while i < len(stops):
        if stops[i] - current_distance > tank:
            i = i - 1
            if current_spot == i:
                return -1
            current_spot = i
            current_distance = stops[i]
            n += 1
        i += 1
    return n

# covering segments
from collections import namedtuple
Segment = namedtuple('Segment', 'start end')

def optimal_points(segments):
    points = []
    sorted_segments = sorted(segments)
    i = 0
    while i < len(sorted_segments) - 1:
        if (sorted_segments[i].end >= sorted_segments[i+1].start):
            new_segment = Segment(sorted_segments[i+1].start, min(sorted_segments[i+1].end, sorted_segments[i].end))
            sorted_segments[i+1] = new_segment
        else:
            points.append(sorted_segments[i].end)

        i = i + 1
    points.append(sorted_segments[i].end)
    return points

# for given number, find max number of differnt numbers which will sum up to the number
def optimal_summands(n):
    summands = []
    #write your code here
    current_sum = 0
    i = 1
    while i <= n:
        if current_sum + i == n:
            summands.append(i)
            return summands
        elif current_sum + i < n:
            current_sum = current_sum + i
            summands.append(i)
            i = i + 1
        else:
            current_sum = current_sum - (i - 1)
            summands[-1] = n - current_sum
            return summands

# combine numbers together to get the largest number
from functools import cmp_to_key
def compare_two_numbers(a, b):
    str_a = str(a)
    str_b = str(b)
    if str_a + str_b > str_b + str_a:
        return -1
    else:
        return 1

def largest_number(a):
    sorted_a = sorted(a, key=cmp_to_key(compare_two_numbers))
    #write your code here
    res = ""
    for x in sorted_a:
        res += str(x)
    return res
```

4. Divide-and-Conquer: Recursive