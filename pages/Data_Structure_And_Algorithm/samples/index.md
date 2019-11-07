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

'''
For a fibonacci sequence, get a new sequence from Fn % m

we can see that it is a pisano period sequence starts with 0, 1.
'''
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

'''
Sum of fibonacci sequence:
F0 + F1 + F2 + ... Fn = F(n+2) - 1

F0^2 + F1^2 + F2^2 + ... Fn^2 = FnF(n+1)
'''

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