## Tree and Graph

* **The distance between 2 binary strings is the sum of their lengths after removing the common prefix. Given a list of binary strings, pick a pair that gives you maximum distance among all possible pair and return that distance.**
<details>

```Java
// "static void main" must be defined in a public class.
class BinaryTrieNode {
	private char ch;

	BinaryTrieNode left;
	BinaryTrieNode right;

	public BinaryTrieNode(char c) {
		ch = c;
	}

	public void set(char ch, BinaryTrieNode node) {
		if (ch == '0') {
			left = node;
		} else if (ch == '1') {
			right = node;
		}
	}

	public BinaryTrieNode get(char ch) {
		if (ch == '0') {
			return left;
		} else if (ch == '1') {
			return right;
		}
		return null;
	}
}

class BinaryTrie {
	private BinaryTrieNode root;
	private int maxDiff;

	public BinaryTrie(List<String> binaries) {
		root = new BinaryTrieNode('\0');
		maxDiff = 0;

        // insert each binary string into Trie
		for (String str : binaries) {
			BinaryTrieNode curr = root;

			for (char ch : str.toCharArray()) {
				BinaryTrieNode child = curr.get(ch);

				if (child == null) {
					child = new BinaryTrieNode(ch);
					curr.set(ch, child);
				}

				curr = child;
			}
		}
	}

	public int getMaxDiff() {
		getMaxDepth(root);

		return maxDiff;
	}

    // helper method to calculate depth of a trie node.
	private int getMaxDepth(BinaryTrieNode root) {
		if (root == null)
			return 0;

        // calculate left child depth
		int leftDepth = getMaxDepth(root.left);
        
        // calculate rightt child depth
		int rightDepth = getMaxDepth(root.right);
        
		if (leftDepth > 0 && rightDepth > 0) {
			maxDiff = Math.max(maxDiff, leftDepth + rightDepth);
		}

        // send max depth between left and right to upper recursive level
		return 1 + Math.max(leftDepth, rightDepth);
	}
}

public class Main {
    public static void main(String[] args) {
        List<String> binaries = new ArrayList(Arrays.asList("1011100", "1011011","1001111"));

		BinaryTrie trie = new BinaryTrie(binaries);
        
		System.out.println(trie.getMaxDiff()); // gives 10 (1011100, 1001111) differ by 10.
        
        binaries = new ArrayList(Arrays.asList("101", "111","000"));
        
        trie = new BinaryTrie(binaries);
        
		System.out.println(trie.getMaxDiff()); // return 6 (101, 000)
    }
}
```
</details>

* **minimum height tree / Minimum Distance To The Farthest Node**
<details>

```python
def findMinHeightTrees(n, edges):
        if n == 1:
            return [0]
        connections = defaultdict(set)
        
        for u, v in edges:
            connections[u].add(v)
            connections[v].add(u)
        
        leaves = set(node for node in connections if len(connections[node]) == 1)
        
        while len(connections) > 2:
            new_leaves = set()
            for leaf in leaves:
                nbor = connections[leaf].pop()
                connections[nbor].remove(leaf)
                if len(connections[nbor]) == 1:
                    new_leaves.add(nbor)
                del connections[leaf]
            leaves = new_leaves
        
        return list(connections.keys())
```
</details>

* **Minimum Cost to Make at Least One Valid Path in a Grid**

https://leetcode.com/problems/minimum-cost-to-make-at-least-one-valid-path-in-a-grid/

<details>

```python
def minCost(self, grid):
    m, n = len(grid), len(grid[0])

    def neighborhood(y, x):
        if x + 1 < n:
            yield y, x + 1, int(grid[y][x] != 1)
        if x > 0:
            yield y, x - 1, int(grid[y][x] != 2)
        if y + 1 < m:
            yield y + 1, x, int(grid[y][x] != 3)
        if y > 0:
            yield y - 1, x, int(grid[y][x] != 4)

    min_cost = collections.defaultdict(lambda: float('inf'))
    min_cost[(0, 0)] = 0
    queue = collections.deque([(0, 0, 0)])

    while queue:
        cost, y, x = queue.popleft()

        if cost != min_cost[y, x]:
            continue

        if y == m - 1 and x == n - 1:
            return cost


        for y2, x2, step_cost in neighborhood(y, x):
            cost2 = cost + step_cost
            if cost2 < min_cost[y2, x2]:
                min_cost[y2, x2] = cost2

                if not step_cost:
                    queue.appendleft((cost2, y2, x2))
                else:
                    queue.append((cost2, y2, x2))
```
</details>

* **minimum time to wet all the nodes**
<details>

```python
def networkDelayTime(times, N, K):
    g = collections.defaultdict(list)
    for s, d, time in times:
        g[s].append((d, time))
    
    hq = [(0, K)]
    dist = {}
    res = 0
    while hq:
        d, node = heapq.heappop(hq)
        
        if node in dist:
            continue
        dist[node] = d
        res = max(res, d)
        
        for n, d2 in g[node]:
            if n not in dist:
                heapq.heappush(hq, (d2+d, n))

    if len(dist) != N:
        return -1
    return res
```
</details>

* **break walls: break maximum k walls**

https://leetcode.com/problems/shortest-path-in-a-grid-with-obstacles-elimination/

* **Lights Out Puzzle**

https://leetcode.com/problems/minimum-number-of-flips-to-convert-binary-matrix-to-zero-matrix/

<details>

```java
// Time: 2^(MN)
public class Program {
    int count;
    static int min = Integer.MAX_VALUE;

    public static void main(String[] args) {
        Program p = new Program();
        int[][] matirx = new int[][]{
                {0, 1, 0},
                {1, 1, 1},
                {0, 0, 0}
        };
        p.helper(matirx, 3, 3, 0);
        if (min == Integer.MAX_VALUE) System.out.println(-1);
        else System.out.println(min);
    }
    private void helper(int[][] matrix, int cols, int rows, int pos) {
        int i = pos / cols;
        int j = pos % cols;
        if (i >= rows) {
            boolean oneExist = false;
            for (int m = 0; m < rows; m++) {
                for (int n = 0; n < cols; n++) {
                    if (matrix[m][n] == 1) {
                        oneExist = true;
                        break;
                    }
                }
                if (oneExist) break;
            }
            if (!oneExist) {
                min = Math.min(min, count);
                return;
            } else {
                return;
            }

        }
        // not flip;
        helper(matrix, cols, rows, pos + 1);
        // flip
        filp(matrix, i, j, cols, rows);
        count++;
        helper(matrix, cols, rows, pos + 1);
        filp(matrix, i, j, cols, rows);
        count--;

    }

    private void filp(int[][] matrix, int i, int j, int cols, int rows) {
        int[] dr = new int[]{0, 0, -1, 0, 1};
        int[] dc = new int[]{0, -1, 0, 1, 0};
        for (int k = 0; k < 5; k++) {
            int r = i + dr[k];
            int c = j + dc[k];
            if (r >= 0 && r < rows && c >= 0 && c < cols) {
                matrix[r][c] = matrix[r][c] == 1 ? 0 : 1;
            }
        }
    }
}
```
</details>

* **house robber, Find max sum of elements in a Binary Tree, such that you don't select the adjuscent nodes**
<details>

```python
def rob(self, root: TreeNode) -> int:
    def dfs(node):
        if not node:
            return 0
        
        if node in resMap:
            return resMap[node]
        
        res = node.val
        
        if node.left:
            res += dfs(node.left.left) + dfs(node.left.right)
            
        if node.right:
            res += dfs(node.right.left) + dfs(node.right.right)
            
        
        res = max(res, dfs(node.left) + dfs(node.right))
        
        resMap[node] = res            
        return res
    
    
    resMap = dict()
    return dfs(root)
```
</details>

* **war house / binary tree cameras**

https://leetcode.com/problems/binary-tree-cameras/

<details>

```python
from typing import List, Tuple
from enum import Enum

class RoomCoverage(Enum):
	NOT_COVERED = 1
	IS_COVERED = 2
	CONTAINS_BOMB = 3

class Room:
	def __init__(self, number: int, adjacent: List[int]):
		self.number = number
		self.adjacent = adjacent

def min_bomb_coverage(rooms: List[Room]):
	if not rooms:
		return 0
		
	def dfs(room_number, parent_number):
		room = rooms[room_number]
		
		if not room.adjacent or (len(room.adjacent) == 1 and room.adjacent[0] == parent_number):
			return 0, RoomCoverage.NOT_COVERED
		
		count = 0
		coverage = RoomCoverage.NOT_COVERED
		
		for child_count, child_coverage in (
			dfs(child_number, room_number)
			for child_number in room.adjacent
			if child_number != parent_number
		):
			count += child_count
			
			if child_coverage is RoomCoverage.CONTAINS_BOMB and coverage is RoomCoverage.NOT_COVERED:
				coverage = RoomCoverage.IS_COVERED
			elif child_coverage is RoomCoverage.NOT_COVERED:
				coverage = RoomCoverage.CONTAINS_BOMB
			
		if coverage is RoomCoverage.CONTAINS_BOMB:
			count += 1
		
		return count, coverage
	
	count, coverage = dfs(0)
	
	if coverage is RoomCoverage.NOT_COVERED:
		count += 1
	
	return count
```
</details>

* **Friend Suggestion, return the person that has most friends in common**
<details>

```java
public Person friendSuggestion(Person p) {
    int max = -1;
    Person output = null;

    Map<Person, Integer> map = new HashMap<>();

    for (Person friend : p.friends) {
      for (Person mutual : friend.friends) {
        if (mutual.id != p.id && !p.friends.contains(mutual)) {
          map.put(mutual, map.getOrDefault(mutual, 0) + 1);
        }
      }
    }

    for (Map.Entry<Person, Integer> mutual : map.entrySet()) {
      if (mutual.getValue() > max) {
        max = mutual.getValue();
        output = mutual.getKey();
      }
    }

    return output;
  }
```
</details>

* **Currency Conversion / Evaluate Division**

https://leetcode.com/problems/evaluate-division/

## DP

* **split array into two with min difference**
<details>

```python
def lastStoneWeightII(stones):
    total = 0
    for s in stones:
        total += s
    
    target = total // 2
    
    dp = [[0] * (target +1) for i in range(len(stones) +1)]
    
    for i in range(1, len(stones)+1):
        for j in range(1, target+1):
            stone = stones[i-1]
            if stone > j:
                dp[i][j] = dp[i-1][j]
            else:
                dp[i][j] = max(dp[i-1][j], dp[i-1][j-stone] + stone)
    
    return total - 2*dp[-1][-1]
```
</details>

* **light bulb k-empty slots**
<details>

```python
def kEmptySlots(flowers, k):
    days = [0] * len(flowers)
    for day, position in enumerate(flowers):
        days[position - 1] = day+1


    ans = float('inf')
    left, right = 0, K+1
    while right < len(days):
        for i in range(left + 1, right):
            if days[i] < days[left] or days[i] < days[right]:
                left, right = i, i+K+1
                break
        else:
            ans = min(ans, max(days[left], days[right]))
            left, right = right, right+K+1

    return ans if ans < float('inf') else -1
```
</details>

* **min days to bloom: Given an array of roses. roses[i] means rose i will bloom on day roses[i]. Also given an int k, which is the minimum number of adjacent bloom roses required for a bouquet, and an int n, which is the number of bouquets we need. Return the earliest day that we can get n bouquets of roses.**
<details>

```python
# DP O(nL)
def minDaysBloomByDp(roses, k, n):
    windowKmax = [0] * (len(roses) - k + 1);
    fillMax(windowKmax,roses,k);
    dp = [[float('inf')] * (len(roses) + 1) for i in range(n+1)]

    for i in range(len(roses) + 1):
        dp[0][i] = 0

    for i in range(1, n+1):
        for j in range(k, len(roses) + 1):
            dp[i][j] = min(dp[i][j - 1], max(dp[i - 1][j - k],windowKmax[j - k]));

    return dp[-1][-1]

def fillMax(windowKmax, r, k):
    dq = collections.deque()
    for i in range(len(r)):
        if (i >= k && r[i - k] == dq.peekFirst()):
            dq.popleft();
        while (!dq.isEmpty() && r[i] > dq.peekLast()):
            dq.pop();
        dq.offerLast(r[i]);
        if (i >= k - 1):
            windowKmax[i - k + 1] = dq.peekFirst();

# BS O(L*log(max-min))  
int minDaysBloomByBS(int[] roses, int k, int n) {
    int min = Integer.MAX_VALUE, max = -1;
    for (int r : roses) {
        max = Math.max(r,max);
        min = Math.min(r,min);
    }
    int[] windowKmax = new int[roses.length - k + 1];
    fillMax(windowKmax,roses,k);
    int s = min, e = max;
    while (s <= e) {
        int mid = (e - s)/2 + s;
        if (search(windowKmax,n,k,mid)) {
            e = mid - 1;
        } else {
            s = mid + 1;
        }
    }
    return e + 1;
}

boolean search(int[] win,int n,int k,int day) {
    for (int i = 0; i < win.length; ) {
        if (day >= win[i]) {
            n--;
            i+=k;
        } else {
            i++;
        }
    }
    return n <= 0;
}

# could u do it better, if n*k nearly equal to L time complexity O(n * (L - n * k))
int minDaysBloomByDp(int[] roses, int k, int n) {
    int[] windowKmax = new int[roses.length - k + 1];
    fillmax(windowKmax,roses,k);
    int[][] dp = new int[n+1][roses.length + 1];
    int fix = n*k;
    for (int i = 1; i <= n; i++) {
        Arrays.fill(dp[i],Integer.MAX_VALUE);
        int st = i * k;
        for (int j = st; j <= roses.length - fix + st; j++) {
            dp[i][j] = Math.min(dp[i][j - 1], Math.max(dp[i - 1][j - k],windowKmax[j - k]));
        }
    }
    return dp[n][roses.length];
}
```
</details>

* **with a maximum value, cannot pick adjacent elements, find the maximum pick; Knapsack to avoid consecutive elements**
<details>

```python
# Let bool F[i][j] be true if the robot can have j strawberries after going through the first i bushes AND picking i-th bush.
# Let bool G[i][j] be true if the robot can have j strawberries after going through the first i bushes AND NOT picking i-th bush.

# F[i][j] = G[i - 1][j - s[i]];
# G[i][j] = G[i - 1][j] or F[i - 1][j];
def maxStrawberries(A, num):
    n = len(A)
    F = [[False] * (num + 1) for _ in range(n + 1)]
    G = [[False] * (num + 1) for _ in range(n + 1)]
    F[0][0], G[0][0] = True, True
    for i, x in enumerate(A):
        F[i + 1] = [g | (y >= x and G[i][y - x]) for y, g in enumerate(G[i])]
        G[i + 1] = [f | g for f, g in zip(F[i], G[i])]
    return num - min(F[-1][::-1].index(True), G[-1][::-1].index(True))

def maxStrawberries1(A, num):
    n = len(A)
    F = [True] + [False] * num
    G = [True] + [False] * num
    for x in A:
        F_new = G[:x] + [g1 | g2 for g1, g2 in zip(G, G[x:])]
        G_new = [f | g for f, g in zip(F, G)]
        F, G = F_new, G_new
    return num - min(F[::-1].index(True), G[::-1].index(True))

def maxStrawberries2(A, num):
    n = len(A)
    F, G = {0}, {0}
    for x in A:
        F_new = G | {g + x for g in G if g + x <= num}
        G_new = F | G
        F, G = F_new, G_new
    return max(F | G)

```
</details>

* **2 payer dp problem**
<details>

```python
# On each player's turn, that player can take all the stones in the first X remaining piles, where 1 <= X <= 2M.  Then, we set M = max(M, X).
def stoneGameII(piles):
    N = len(piles)
    # sum of stones from end to beggining
    for i in range(N-2, -1, -1):
        piles[i] += piles[i+1]

    memo = [[0 for _ in range(N)] for _ in range(N)]

    def dp(i, m):
        if memo[i][m]:
            return memo[i][m]

        if i + 2*m >= N:
            return piles[i]

        res = float('inf')
        for j in range(1, 2*m+1):
            res = min(res, dp(i+j, max(m,j)))
        memo[i][m] = piles[i]-res

        return memo[i][m]

    return dp(0, 1)
```

```java
\\ a player can choose 1, 2, or 3 cards from the beggining of the array
public int sum(int ar[]) {
    int dp[] = new int[ar.length];
    int sum[] = new int[ar.length];

    for (int i = ar.length - 1; i >= 0; i--)
        sum[i] += (i == ar.length - 1) ? ar[i] : sum[i + 1] + ar[i];

    for (int i = ar.length - 1; i >= 0; i--)
    {
        int one = (i < ar.length - 1) ? (sum[i + 1] - dp[i + 1] + ar[i]) : ar[i]; 
        int two = (i < ar.length - 2) ? (sum[i + 2] - dp[i + 2] + ar[i] + ar[i + 1]) : ((i < ar.length - 1) ? ar[i] + ar[i + 1] : Integer.MIN_VALUE);
        int three = (i < ar.length - 3) ? (sum[i + 3] - dp[i + 3] + ar[i] + ar[i + 1] + ar[i + 2]) : ((i < ar.length - 2) ? ar[i] + ar[i + 1] + ar[i + 2] : Integer.MIN_VALUE);
        
        dp[i] = (int) Math.max(Math.max(one, two), three);
    }
        
    return dp[0];
}
```
</details>

* **Count number of ways to partition a set into k subsets**
<details>

```python
def countP(n, k):  
    # Base cases 
    if (n == 0 or k == 0 or k > n): 
        return 0
    if (k == 1 or k == n): 
        return 1
      
    # S(n+1, k) = k*S(n, k) + S(n, k-1) 
    return (k * countP(n - 1, k) + 
                countP(n - 1, k - 1)) 

# O(nk)

def countP(n, k): 
    dp = [[0 for i in range(k + 1)]  
             for j in range(n + 1)] 
  
    for i in range(n + 1): 
        dp[i][0] = 0
  
    for i in range(k + 1): 
        dp[0][k] = 0
  
    # Fill rest of the entries in  
    # dp[][] in bottom up manner 
    for i in range(1, n + 1): 
        for j in range(1, k + 1): 
            if (j == 1 or i == j): 
                dp[i][j] = 1
            else: 
                dp[i][j] = (j * dp[i - 1][j] +
                                dp[i - 1][j - 1]) 
                  
    return dp[n][k] 
```
</details>

* **min of max of path**
<details>

```java
public static int minMax(int[][] arr) {
        int[][] dp = new int[arr.length][arr[0].length]; 
        
        for(int i = 0; i < dp.length; i++) {
            for(int j = 0; j < dp[i].length; j++) {
                if(i == 0 && j == 0) {
                    dp[i][j] = arr[i][j];
                    continue; 
                } 
                int top =  i > 0 ? dp[i - 1][j] : Integer.MAX_VALUE; 
                int left = j > 0 ? dp[i][j - 1] : Integer.MAX_VALUE; 
                dp[i][j] = Math.min(Math.max(top, arr[i][j]), Math.max(left, arr[i][j])); 
            }
        }
        return dp[dp.length - 1][dp[0].length - 1];
    }
```
</details>

* **Max Sum from left or right, select k numbers with muliplication array**
<details>

```java
int solve(vector<int>& nums, int k, int sum, int i, int j, vector<vector<int>>& memo) {
	if (k == 0) return sum;
	if (i > j) return sum;
	if (memo[i][j] != -1) return memo[i][j];

	int res = max(solve(nums, k - 1, sum + nums[i], i + 1, j, memo), solve(nums, k - 1, sum + nums[j], i, j - 1, memo));

	memo[i][j] = res;

	return memo[i][j];
}

```
</details>

* **Maximum sum of a square no larger than k**

```python
Binary search on the length of the square along with 2D cumulative sum
```
https://leetcode.com/problems/max-sum-of-rectangle-no-larger-than-k/


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

* **Distinct Subsequences**

https://leetcode.com/problems/distinct-subsequences/


* **trapping rain water**

https://leetcode.com/problems/trapping-rain-water/


* **word subsets**

https://leetcode.com/problems/word-subsets/


* **Generalized Abbreviation**

https://leetcode.com/problems/generalized-abbreviation/

* **Group Anagrams**

https://leetcode.com/problems/group-anagrams/

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

* **watering flowers**
<details>

```python
public int waterPlants(int[] plants, int cap1, int cap2) {
    // input validation not needed
    int can1 = 0;
    int can2 = 0;
    int lo = 0;
    int hi = plants.length - 1;
    int numRefills = 0;
  
    while (lo < hi) {
        if (can1 < plants[lo]) {
            can1 = cap1;
            ++numRefills;
        }
        if (can2 < plants[hi]) {
            can2 = cap2;
            ++numRefills;
        }

        can1 -= plants[lo++];
        can2 -= plants[hi--];
    }
    if (lo == hi && (plants[lo] > can1 + can2)) {
        return ++numRefills;
    } else {
        return numRefills;
    }
```
</details>