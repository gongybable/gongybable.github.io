# Numpy

## Why use Numpy
By using built-in numpy functions such as `np.dot` will eliminate the need of a `for loop`. It enables Phyton Pi to take much better advantage of parallelism to do computations on arrays much faster.

## Array creation
<details>

```python
np.arange(start, stop, step)
np.linspace(start, stop, nums)

np.array(42)            # 0D array
np.array([1, 2, 3])     # 1D array
np.ones((2,3))
np.zeros((2,3))
np.identity(4)          # identity array
np.eye(N, M=None, k=0)  # identity array

# Accessing elements in np array
A[1][0]     # inefficient, it creates a tmp array A[1] first, then access tmp[0]
A[1, 0]     # more efficient

# slicing array
# Note - instead of creating a new array, this just creates a new view(reference) of the old array
# i.e. update in the new view will update the original array as well
# np.may_share_memory(A, B) can check if two array share the view or not
A[start:stop:step, start:stop:step]
A[::-1]                 # reverse order of the array
A[::-1,::-1]            # reverse row and column for the 2D array

# copy array
np_arr.copy()

# flatten array
A.flatten()                 # creates a copy of the array
A.ravel()                   # creates a view of the array
A.reshape((2, -1))          # creates a view of new shape
np.concatenate((A, B, C), axis = 0)
B = A[:, np.newaxis]        # creates a view of new shape
np.row_stack((A, B))
np.column_stack((A, B))
np.tile(A, (3,4))           # repeating A 3 times in row and 4 times in column

# Random Numbers
np.random.random((100, 1))
random.randint(1,10)        # random int within 1 - 10
numpy.random.choice(arr)    # pick randomly from arr

# strucutred array
dt = np.dtype([('country', np.unicode, 20), ('population', 'i4')])
population_table = np.array([
    ('Netherlands', 16928800),
    ('Belgium', 11007020)
], dtype=dt)
```
</details>

## Broadcast and Vectorization
<details>

```python
'''
Elementwise Operation
A - matrix shape (m, n)
B - matrix shape (m, n)
'''
A * B
A + B
A == B     # element wise operation, return shape (m, n);
           # np.array_equal(A, B) will return just True/False
           # np.logical_or(A, B), np.logical_and(A, B) is elementwise operation
           # A[A<10] array selection with values <10

'''
Broadcast
A - matrix shape (m, n)
B - matrix shape (m, 1), or (1, n), or just a number a
'''
A * B
A + B
A - B
A / B       # B will broadcast to the same shape as A, and then do element wise operation

'''
Broadcast
A - matrix shape (m, 1)
B - matrix shape (1, n)
'''
A * B       # A will broadcast to (m, n), B will broadcast to (m, n), result in shape (m, n)

'''
A1, B1 - just numbers
A2, B2 - 1D arrays
A3, B3 - numlti-D arrays
'''
np.dot(A1, B1)      # returns a number
np.dot(A2, B2)      # vector multiplication, returns a number
np.dot(A3, B3)      # matrix multiplication, returns a matrix

'''
vectorized outer product
'''
outer = np.outer(x1,x2)

'''
vectorized elementwise multiplication
'''
mul = np.multiply(x1,x2)

'''
matrix multiplication
'''
np.matmul(A, B)
```
</details>