* **Maximum sum of a square no larger than k**

```python
Binary search on the length of the square along with 2D cumulative sum
```

## backtrack

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

```Java
// Complexity O((H*W*K)*(H+W))
int main() {
    int H = 10;
    int W = 10;
    int k = 5;
    vector<vector<char>> cake = {
        {'v', 'v', 'v', 'v', 'v', 'v', 'v', 'v', 'v', 'v'},
        {'v', 'v', 'v', 'v', 'v', 'v', 'v', 'v', 'v', 'v'},
        {'v', 'v', 'v', 'v', 'v', 'v', 'v', 'v', 'v', 'v'},
        {'v', 'v', 'v', 'v', 'v', 'v', 'v', 'v', 'v', 'v'},
        {'v', 'v', 'v', 'v', 'v', 'v', 'v', 'v', 'v', 'v'},
        {'v', 'v', 'v', 'v', 'v', 'v', 'v', 'v', 'v', 'v'},
        {'v', 'v', 'v', 'v', 'v', 'v', 'v', 'v', 'v', 'v'},
        {'v', 'v', 'v', 'v', 'v', 'v', 'v', 'v', 'v', 'v'},
        {'v', 'v', 'v', 'v', 'v', 'v', 'v', 'v', 'v', 'v'},
        {'v', 'v', 'v', 'v', 'v', 'v', 'v', 'v', 'v', 'v'}
    };
    
    vector<vector<int>> staw(H+1, vector<int>(W+1, 0));
    
    staw[H-1][W-1] = (cake[H-1][W-1] == 'v') ? 1 : 0;
    
    for(int i = H-1;i>=0;i--) {
        for(int j = W-1;j>=0;j--) {
            if (i == H-1 && j == W-1) {
                staw[H-1][W-1] = (cake[H-1][W-1] == 'v') ? 1 : 0;
            } else if (i == H-1) {
                staw[i][j] = (cake[i][j] == 'v' ? 1 : 0) + staw[i][j+1];
            } else if (j == W-1) {
                staw[i][j] = (cake[i][j] == 'v' ? 1 : 0) + staw[i+1][j];
            } else {
                staw[i][j] = (cake[i][j] == 'v' ? 1 : 0) + staw[i+1][j] + staw[i][j+1] - staw[i+1][j+1];
            }
        }
    }
    
    vector<vector<vector<int>>> dp(H+1, vector<vector<int>>(W+1, vector<int>(k+1, 0)));
    
    
    for(int i = H-1;i>=0;i--) {
        for(int j = W-1;j>=0;j--) {
            dp[i][j][0] = 0;
            dp[i][j][1] = ((staw[i][j] > 0) ? 1 : 0);
            if (i == H-1 && j == W-1) {
                continue;
            }
            for(int cuts = 2;cuts <= k; cuts++) {
                int curr = staw[i][j];
                for(int row = i;row < H-1;row++) {
                    if (staw[row+1][j] < curr){
                        dp[i][j][cuts] += dp[row+1][j][cuts-1];
                    }
                }
                // Cutting Ways vertically
                for(int col = j;col < W-1;col++) {
                    if (staw[i][col+1] < curr){
                        dp[i][j][cuts] += dp[i][col+1][cuts-1];
                    }
                }
            }
        }
    }
    
    cout<<dp[0][0][k];
}
```
</details>

* **given set of numbers, get all possible multiplication numbers from the set**
<details>

```java
vector<int> Get(vector<int> A) {
  vector<int> ans{1};
  for (int i = 0; i < A.size(); ++i) {
    int size = ans.size();
    for (int j = 0; j < size; ++j) {
      ans.emplace_back(A[i] * ans[j]);
    }
  }

  return ans;
}
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
        result = False
        for i in range(len(budget)):
            if budget[i] < c:
                continue
            
            budget[i] -= c
            result = result or helper(budget,costs)
            if result:
                return True
            budget[i] += c
        
        return result
    
    return helper(budget,repeats)
```
</details>

* **Longest row of dominoes**
<details>

```python
def solution(dominoes):
    def helper(dominoes, sol):
        nonlocal res
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


## heapq

* **Min Cost to Hire K Workers with quality**
<details>

```python
def mincostToHireWorkers(quality, wage, K):
    workers = sorted((w/q, q)
                        for q, w in zip(quality, wage))

    ans = float('inf')
    pool = []
    sumq = 0
    for ratio, q in workers:
        heapq.heappush(pool, -q)
        sumq += q

        if len(pool) > K:
            sumq += heapq.heappop(pool)

        if len(pool) == K:
            ans = min(ans, ratio * sumq)

    return float(ans)
```
</details>

* **min cost to keep employees**
<details>

```python
def solver(cost,salary,severance,nums):
    dp = {0:0}
    for req in nums:
        tmp = collections.defaultdict(lambda: float('inf'))
        for key in dp:
            if key >= req:
                for i in range(req,key+1):
                    tmp[i] = min(tmp[i],dp[key]+i*salary+(key-i)*severance)
            else:  tmp[req] = min(tmp[req],dp[key]+req*salary+(req-key)*cost)
        dp = tmp
    return min(dp.values())
```
</details>

## others

* **split arry into min number of decreasing subsequence**
```python
same as getting longest increasing subsequence in the array
```

* **Maximum Subarray sum**
<details>

```python
def maxSubArray(nums):
    curr = nums[0]
    res = nums[0]
    for i in range(1, len(nums)):
        curr = max(nums[i], nums[i]+curr)
        res = max(res, curr)
    return res

# divide and conquer
def cross_sum(nums, left, right, p): 
        if left == right:
            return nums[left]

        left_subsum = float('-inf')
        curr_sum = 0
        for i in range(p, left - 1, -1):
            curr_sum += nums[i]
            left_subsum = max(left_subsum, curr_sum)

        right_subsum = float('-inf')
        curr_sum = 0
        for i in range(p + 1, right + 1):
            curr_sum += nums[i]
            right_subsum = max(right_subsum, curr_sum)

        return left_subsum + right_subsum   

def helper(nums, left, right): 
    if left == right:
        return nums[left]
    
    p = (left + right) // 2
        
    left_sum = helper(nums, left, p)
    right_sum = helper(nums, p + 1, right)
    cross_sum = cross_sum(nums, left, right, p)
    
    return max(left_sum, right_sum, cross_sum)
    
def maxSubArray(nums):
    return helper(nums, 0, len(nums) - 1)
```
</details>

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

* **best fruit - n fruits and m ppl vote for the fruits, find best fruit; rank**
<details>

```python
# O(n*(m+n))
from collections import defaultdict
import math

def solve(N, M, A):
    favorites = [row[::-1] for row in A]
    remaining = set(range(1, N + 1))

    for r in range(N - 1):
        candidate_votes = {num: 0 for num in remaining}

        for row in favorites:
            while row[-1] not in remaining:
                row.pop()

            candidate_votes[row[-1]] += 1

        eliminate_num = -1
        eliminate_votes = math.inf

        for num, votes in candidate_votes.items():
            if votes < eliminate_votes or (
                votes == eliminate_votes and num < eliminate_num
            ):
                eliminate_num, eliminate_votes = num, votes

        remaining.discard(eliminate_num)

    return remaining.pop()
```

```python
def rankTeams(votes):
    n = len(votes[0])
    t = list(votes[0])
    d = collections.defaultdict(lambda:[0]*n)
    for i in votes:
        for j in range(n):
            d[i[j]][j] += 1
    t.sort(key=lambda x: [-i for i in d[x]] + [x])
    return "".join(t)
```
</details>

* **box stacking**
<details>

```Java
private static int solve1(int[][] nums) {
	Arrays.sort(nums, (a, b) -> a[0] == b[0] ? a[1] == b[1] ? a[2] - b[2] : a[1] - b[1] : a[0] - b[0]);
	int res = 0;
	int dp[] = new int[nums.length];
	for (int i = 0; i < nums.length; i++) {
		dp[i] = 1;
		for (int j = 0; j < i; j++) {
			if (nums[j][0] < nums[i][0] && nums[j][1] < nums[i][1] && nums[j][2] < nums[i][2])
				dp[i] = Math.max(dp[i], dp[j] + 1);
		}
		res = Math.max(dp[i], res);
	}
	return res;
}
```
</details>

* **Product of the Last K Numbers**
<details>

```python
class ProductOfNumbers:
    def __init__(self):
        self.product_table = [1]
        
    def add(self, num):
        if num != 0:
            self.product_table.append( num * self.product_table[-1] )
        else:
            self.product_table = [1]
            
    def getProduct(self, k):
        if k >= len( self.product_table ):
            return 0
        else:
            return self.product_table[-1] // self.product_table[-(k+1)]
```

```java
public class SlidingWindow {
    private LinkedList<Integer> storage;
    private int size;
    private int numZeros;
    private Integer product;

    public SlidingWindow(int k) {
        storage = new LinkedList<Integer>();
        size = k;
        product = new Integer(1);
    }

    public void add(int val) {
        if (size < 1) {
            return;
        }
        if (storage.size() >= size) {
            int divisor = storage.pollFirst();
            if (divisor == 0) {
                --numZeros;
            } else {
                product /= divisor;
            }
        }
        if (val == 0) {
            ++numZeros;
        } else {
            product *= val;
        }
        storage.addLast(val);
    }

    public int getProduct() {
        if (size == 0 || numZeros > 0) {
            return 0;
        }
        return product;
    }
}
```
</details>

* **Substring with all alphabets in sequence**

https://leetcode.com/problems/minimum-window-subsequence/

<details>

```java
class ShortestOrderSeq {
    public String shortestSeq(String s) {
        int[] alphabet = new int[26];
        for (int i=0; i<26; i++) alphabet[i] = -1;

        int start = 0; int end = Integer.MAX_VALUE;
        for (int i=0; i<s.length(); i++) {
            int ch = s.charAt(i) - 'a';

            if (ch < 0 || ch > 25) continue;

            if (ch == 25 && alphabet[24] != -1 && i+1-alphabet[24] < end-start) {
                start = alphabet[24]; end = i+1;
            }

           // if char is 'a' assign alphabet[0] with new position value
           // else assigning alphabet[ch] as alphabet[ch-1] ensures that alphabets are in proper sequence
            alphabet[ch] = (ch == 0) ? i : alphabet[ch-1];
        }

        return end==Integer.MAX_VALUE ? "" : s.substring(start, end);
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        String s = sc.nextLine();
        System.out.println(new ShortestOrderSeq().shortestSeq(s));
    }
}
```
</details>

* **Divide Array in Sets of Consecutive Numbers**

https://leetcode.com/problems/divide-array-in-sets-of-k-consecutive-numbers/

https://leetcode.com/problems/split-array-into-consecutive-subsequences/

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












https://leetcode.com/problems/binary-tree-longest-consecutive-sequence/




https://leetcode.com/problems/optimize-water-distribution-in-a-village/


https://leetcode.com/problems/binary-tree-postorder-traversal/


https://github.com/stephengrice/education/blob/master/BST/bst.py



