## Q1. 

1. implement a key value store
    * get(key) -> value
    * set(key, value) -> None

2. Add setting timestamp support
    * get(key, timestamp=None) -> value (match the timestamp)
    * set(key, value, timestamp) -> None

3. Add getting nearest timestamp support
    * get(key, timestamp) -> value (get the closet timestamp for the key)
    * set(key, value, timestamp) -> None

## Q2. Password
1. Given a 2D matrix and a list of tuples, determine the password

    Tuple: coordinate X, Y, position of the char in the password
    Note: coordinate 0, 0 is bottom left of matrix

    Example:
    3
    A B C
    D E F
    G H I
    3
    (0, 0, 0)
    (2, 1, 2)
    (0, 2, 1)

    out: GBF

    Input to parse:

    first row is number of rows in matrix
    then 3 rows of matrix
    fifth row is number of tuples
    then 3 tuples

2. Input is many chunks of input above, seperated by an empty line, return a sequence of passwords, a new password starts when a duplicate position is encountered in the previous password

    Example 
    2
    A B 
    C D
    3
    0 0 2
    1 1 1
    1 0 0

    2
    E F
    G H
    3
    0 1 3
    1 0 1
    1 1 0

    output: ["DBCG", "FH"] # FH is a new pwd as position 1 is already set in the previous password


## Q3. Card Game
Given a list of non-duplicate cards, find any winning hand if exists.

A card has following properties:
1. Sign: + - or =
2. Letter A, B or C
3. count 1, 2 or 3 (all letters are the same)

for example:
+AA, -B, =CCC, ...

A winning hand is 3 cards and:
1. all signs are the same or all signs are different
2. all letters are same or all letters are different
3. all count are the same or all count are different

For example:
-AAA, =BBB, +CCC
C, CCC, CC
A, BB, CCC




# LeetCode
```python
def exact_match(arr, val):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if arr[mid] == val:
            return mid
        elif arr[mid] > val:
            hi = mid - 1
        else:
            lo = mid + 1
    return -1

def lower_bound(arr, val):
    lo, hi = 0, len(arr) - 1
    res = -1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if arr[mid] == val:
            res = mid
            hi = mid - 1
        elif arr[mid] > val:
            hi = mid - 1
        else:
            lo = mid + 1
    return res

def upper_bound(arr, val):
    lo, hi = 0, len(arr) - 1
    res = -1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if arr[mid] == val:
            res = mid
            lo = mid + 1
        elif arr[mid] > val:
            hi = mid - 1
        else:
            lo = mid + 1
    return res

def largest_samller(arr, val):
    lo, hi = 0, len(arr) - 1
    res = -1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if arr[mid] > val:
            hi = mid - 1
        elif arr[mid] < val:
            res = mid
            lo = mid + 1
        else:
            return mid
    return res

def smallest_larger(arr, val):
    lo, hi = 0, len(arr) - 1
    res = -1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if arr[mid] == val:
            return mid
        elif arr[mid] > val:
            res = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return res

def min_diff(res, val):
    lo, hi = 0, len(arr) - 1
    while lo < hi-1:
        mid = lo + (hi - lo) // 2
        if arr[mid] > val:
            hi = mid
        else:
            lo = mid
    return min(abs(res[lo] - val), abs(res[hi] - val))
```