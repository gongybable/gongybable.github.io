* **Maximum sum of a square no larger than k**

```python
Binary search on the length of the square along with 2D cumulative sum
```

* **fill matrix for sudoku**
<details>

```python
def myrow(pos, n):
    # Returns a list of indices of all elements in the row containing position
    output = []
    pos = pos % n
    return list(range(pos,n**2,n))
    
def mycol(pos, n):
    # Returns a list of indices of all elements in the column containing position
    output = []
    pos = (pos // n)*n
    return list(range(pos, pos +n))

def top_diagonal(n):
    # Returns a list of indices of all elements in the top left to bottom right diagonal
    return [i*n + i for i in range(n)]

def other_diagonal(n):
    # Returns a list of indices of all elements in top right to bottom left diagonal
    return [i*n + n - i - 1 for i in range(n)]

def issafe(board, pos, x, n):
    # Returns True if x can be placed on the board
    global global_sum
    if(board[pos] is not None):
        return False
    if(x in board):
        return False
    if(pos == myrow(pos, n)[-1]):
        if(x + sum([board[r] for r in myrow(pos, n)[:-1]]) != global_sum):
            return False
    if(pos == mycol(pos, n)[-1]):
        if(x + sum([board[r] for r in mycol(pos, n)[:-1]]) != global_sum):
            return False
    if(pos == top_diagonal(n)[-1]):
        if(x + sum([board[r] for r in top_diagonal(n)[:-1]]) != global_sum):
            return False
    if(pos == other_diagonal(n)[-1]):
        if(x + sum([board[r] for r in other_diagonal(n)[:-1]]) != global_sum):
            return False
    return True

def fill(board, pos, n):
    #Backtracking through the board
    if(pos >= n**2):
        return True
    for num in range(1, n**2+1):
        if(issafe(board, pos, num, n)):
            board[pos] = num
            if(fill(board, pos+1, n)):
                return True
            else:
                board[pos] = None
    return False

n = 3
global_sum = n*(n**2+1)/2
board = [None]*(n**2)
val = fill(board, 0, n)
output = []
for i in range(n):
    output.append(board[i*n:i*n+n])
print(output)
print(val)
```
</details>

* **cut cake**
<details>

```python
# Complexity O((H*W*K)*(H+W))
def solution(cake, k):
    R = len(cake)
    C = len(cake[0])
    straw = [[0] * (C+1) for _ in range(R+1)]
    dp = [[[0] * (k+1) for i in range(C+1)] for _ in range(R+1)]
    
    for i in range(R-1, -1, -1):
        for j in range(C-1, -1, -1):
            if i == R-1 and j == C-1:
                straw[i][j] = 1 if cake[i][j] == 'v' else 0
            elif i == R-1:
                straw[i][j] = straw[i][j+1] + 1 if cake[i][j] == 'v' else 0
            elif j == C-1:
                straw[i][j] = straw[i+1][j] + 1 if cake[i][j] == 'v' else 0
            else:
                straw[i][j] = straw[i+1][j] + straw[i][j+1] - straw[i+1][j+1] + 1 if cake[i][j] == 'v' else 0
    
    for i in range(R-1, -1, -1):
        for j in range(C-1, -1, -1):
            dp[i][j][0] = 0
            dp[i][j][1] = 1 if straw[i][j] > 0 else 0
            if i == R-1 and j == C-1:
                continue
            for cuts in range(2, k+1):
                curr = straw[i][j]
                for row in range(i, R):
                    if straw[row+1][j] < curr:
                        dp[i][j][cuts] += dp[row+1][j][cuts-1]

                # Cutting Ways vertically
                for col in range(j, C):
                    if straw[i][col+1] < curr:
                        dp[i][j][cuts] += dp[i][col+1][cuts-1]
    return dp[0][0][k]
```
</details>

* **given set of numbers, get all possible multiplication numbers from the set**
<details>

```python
def solution(arr):
    res = [1]
    for num in arr:
        l = len(res)
        for i in range(l):
            res.append(res[i]*num)
    return res
```
</details>

* **repeating integers, frequency**
<details>

```python
def hasRepeats(nums, repeats):
    histogram = {}
    for num in nums:
        if num not in histogram:
            histogram[num] = 0
        histogram[num] += 1

    budget = list(histogram.values())

    def helper(budget,costs):

        if len(costs) == 0:
            return True

        c = costs.pop()
        for i in range(len(budget)):
            if budget[i] < c:
                continue
            
            budget[i] -= c
            if helper(budget,costs)
                return True
            budget[i] += c
        
        return False
    
    return helper(budget,repeats)
```
</details>

* **Longest row of dominoes**
<details>

```python
def solution(dominoes):
    def helper(dominoes, sol):
        if len(sol) > len(res):
            res = sol[:]

        for i in range(len(dominoes)):
            if not sol or sol[-1][1] == dominoes[i][0]:
                sol.append(dominoes[i])
                new_dominoes = dominoes[:i] + dominoes[i+1:]
                helper(new_dominoes, sol)
                sol.pop()

    res = []
    helper(dominoes, [])
    return res
```
</details>

* **split arry into min number of decreasing subsequence**
```python
same as getting longest increasing subsequence in the array
```

* **pizza shop - pizza price, and 0/1/2 toppings**
<details>

```python
def closestPrice(pizzas, toppings, x):
    import bisect
    closest = float('inf')
    new_toppings = [0]
# Generate combinations for 0, 1, and 2 toppings
    for i in range(len(toppings)):
        new_toppings.append(toppings[i])
        for j in range(i+1, len(toppings)):
            new_toppings.append(toppings[i] + toppings[j])
    new_toppings.sort()
    for pizza in pizzas:
        idx = bisect.bisect_left(new_toppings, x - pizza)
        for j in range(idx-1, idx+2):
            if 0 <= j < len(new_toppings):
                diff = abs(pizza + new_toppings[j] - x)
                if diff == abs(closest - x):
                    closest = min(closest, pizza + new_toppings[j]) # When two are equal, take the lowest one according to example 3
                elif diff < abs(closest - x):
                    closest = pizza + new_toppings[j]
    return closest
```
</details>

* **binary searchable**
<details>

```python
def bs(arr):
    res = [False] * len(arr)
    def helper(left, right, upper, lower):
        if left <= right:
            mid = left + (right - left) // 2
            if arr[mid] > lower and arr[mid] < upper:
                res[mid] = True

            if left < right:
                helper(mid+1, right, upper, max(lower, arr[mid]))
                helper(left, mid-1, min(upper, arr[mid]), lower)
                
    helper(0, len(arr)-1, float('inf'), float('-inf'))
    
    return res
```
</details>

* **design shopping cart**
<details>

```java
public class ShoppingCart {
	// map1: offerID: prodID, price
    // map2: prodID: price, offerID (BST)
	public void addOffer(productID, offerID, price) {
	}

	public void removeOffer(offerID) {
	}

	public long getClosestOffer(productID, price) {
	}

}
```
</details>

* **Given an Array A, find the minimum amplitude you can get after changing up to 3 elements. Amplitude is the range of the array (basically difference between largest and smallest element).**

```python
O(n) to find 4 maximums and 4 minimums
There are 4 options, remove all 3/2/1/0 maximums
```

* **Given a string S, we can split S into 2 strings: S1 and S2. Return the number of ways S can be split such that the number of unique characters between S1 and S2 are the same.**

```python
O(n)
count the S, set distance to be the different chars
loop on S until it contains all the chars and number of chars is smaller than total count
```

* **a string is strickly smaller than another if the count of the smallest char is samller. e.g. b < aa, a < bb. A and B contains multiple strings with length less than 10. return total counts in A that are less than B.**
<details>

```python
def solve(A, B):
    wordsA = A.split(",")
    wordsB = B.split(",")
    freqCounter = [0] * 11
    
    for w in wordsA:
        minFreq = w.count(min(w))
        freqCounter[minFreq] += 1
    
    toReturn = []
    for w in wordsB:
        minFreq = w.count(min(w))
        toReturn.append(sum(freqCounter[:minFreq]))
    
    return toReturn
```
</details>

* **Array is larger if the first non-match element is larger. Give a array A and int K, find the largest subarray with length K.**
<details>

```python
# for unique values:
def largest_subarray(a, k):
    first_idx = 0
    for x in range(1, len(a) - k + 1):
        if a[first_idx] < a[x]:
            first_idx = x

    return a[first_idx:first_idx+k]

# non unique values
def largest_subarray(a, k):
    first_idx = 0
    for x in range(1, len(a) - k + 1):
        for i in range(k):
            if a[first_idx + i] < a[x + i]:
                first_idx = x
                break
            elif a[first_idx + i] > a[x + i]:
                break

    return a[first_idx:first_idx+k]
```
</details>

* **maximum time**
<details>

```python
def giveMeMaxTime(time):
    time = list(time)

    if time[0] == '?':
        if time[1] <= '3' or time[1] == '?':
            time[0] = '2'
        else:
            time[0] = '1'

    time[1] == '?':
        if time[0] == '2':
            time[1] = '3'
        else:
            time[1] = '9'

    if time[3] == '?':
        time[3] = '5'
    if time[4] == '?':
        time[4] = '9'

    return "".join(time)
```
</details>



https://leetcode.com/problems/binary-tree-postorder-traversal/


https://github.com/stephengrice/education/blob/master/BST/bst.py



