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

```python
def get_wpd():
    matrix_row = int(input())
    matrix = []
    for _ in range(matrix_row):
        row = input().strip().split(" ")
        matrix.append(row)

    matrix = matrix[::-1]

    pwd_len = int(input())
    pwd = [""] * pwd_len
    for _ in range(pwd_len):
        row = input().strip().split(",")
        i = int(row[0][1:])
        j = int(row[1])
        pos = int(row[2][:-1])
        pwd[pos] = matrix[i][j]

    res = "".join(pwd)
    return res

if __name__ == '__main__':
    print(get_wpd())
```


2. Input is many chunks of input above, seperated by an empty line, return a sequence of passwords, a new password starts when a duplicate position is encountered in the previous password

    Example 
    2
    A B 
    C D
    3
    (0, 0, 2)
    (1, 1, 1)
    (1, 0, 0)

    2
    E F
    G H
    3
    (0, 1, 3)
    (1, 0, 1)
    (1, 1, 0)

    output: ["DBCG", "FH"] # FH is a new pwd as position 1 is already set in the previous password

```python
if __name__ == '__main__':
    res = []
    pwd_arr = []
    pos_set = set()
    while True:
        try:
            matrix_row = int(input())
            matrix = []
            for _ in range(matrix_row):
                row = input().strip().split(" ")
                matrix.append(row)

            matrix = matrix[::-1]

            pwd_len = int(input())
            pwd_arr = pwd_arr + [""] * pwd_len
            for i in range(pwd_len):
                row = input().strip().split(",")
                i = int(row[0][1:])
                j = int(row[1])
                pos = int(row[2][:-1])
                if pos in pos_set:
                    pwd = "".join(pwd_arr)
                    res.append(pwd)

                    pos_set = set()
                    pwd_arr = [""] * (pwd_len - i)

                pos_set.add(pos)
                pwd_arr[pos] = matrix[i][j]                    
            
            input() # empty line
        except EOFError:
            pwd = "".join(pwd_arr)
            res.append(pwd)
            print(res)
            break
        except Exception as e:
            raise(e)
```
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

```python
from collections import defaultdict

def card_to_tuple(card):
    sign = card[0]
    letter = card[1]
    count = len(card) - 1
    return (sign, letter, count)

def tuple_to_card(s, l, c):
    return s+l*c

def get_valid_pairs(s, l, c):
    signs = []
    letters = []
    counts = []
    pairs = []

    for sign in "+-=":
        if sign != s:
            signs.append(sign)

    for letter in "ABC":
        if letter != l:
            letters.append(letter)

    for count in range(1, 4):
        if count != c:
            counts.append(count)
    
    # 1 different
    pairs.append(((signs[0], l, c), (signs[1], l, c)))
    pairs.append(((s, letters[0], c), (s, letters[1], c)))
    pairs.append(((s, l, counts[0]), (s, l, counts[1])))

    # 2 different
    pairs.append(((signs[0], letters[0], c), (signs[1], letters[1], c)))
    pairs.append(((signs[0], letters[1], c), (signs[1], letters[0], c)))

    pairs.append(((s, letters[0], counts[0]), (s, letters[1], counts[1])))
    pairs.append(((s, letters[0], counts[1]), (s, letters[1], counts[0])))

    pairs.append(((signs[0], l, counts[0]), (signs[1], l, counts[1])))
    pairs.append(((signs[0], l, counts[1]), (signs[1], l, counts[0])))

    # 3 different
    pairs.append(((signs[0], letters[0], counts[0]), (signs[1], letters[1], counts[1])))
    pairs.append(((signs[0], letters[0], counts[1]), (signs[1], letters[1], counts[0])))
    pairs.append(((signs[0], letters[1], counts[0]), (signs[1], letters[0], counts[1])))
    pairs.append(((signs[1], letters[0], counts[0]), (signs[0], letters[1], counts[1])))

    return pairs

if __name__ == '__main__':
    cards = defaultdict(int)
    while True:
        try:
            card = input().strip()
            card_tuple = card_to_tuple(card)
            cards[card_tuple] += 1

            valid_pairs = get_valid_pairs(*card_tuple)
            if cards[card_tuple] == 3:
                print([card]*3)

            for pair in valid_pairs:
                if cards[pair[0]] >=1 and cards[pair[1]] >=1:
                    print([card, tuple_to_card(*pair[0]), tuple_to_card(*pair[1])])
            
        except EOFError:
            break
        except Exception as e:
            raise(e)
```

## Q4 HTTP Requests
https://realpython.com/python-requests/

```python
import requests

response = requests.get(
    'https://api.github.com/search/repositories',
    params={'q': 'requests+language:python'},
    headers={'Accept': 'application/vnd.github.v3.text-match+json'},
)

# View the new `text-matches` array which provides information
# about your search term within the results
json_response = response.json()
repository = json_response['items'][0]
print(f'Text matches: {repository["text_matches"]}')
```

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