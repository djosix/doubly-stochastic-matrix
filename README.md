## Generating Doubly Stochastic Matrix

Algorithms for randomly generating doubly stochastic matrices. A doubly stochastic matrix is one where each row and each column sums up to 1.


### Algorithm 1

By [Birkhoff–von Neumann theorem](https://en.wikipedia.org/wiki/Doubly_stochastic_matrix), a doubly stochastic matrix is the convex hull of the set of NxN permutation matrices.

```python
import numpy as np

def random_doubly_stochastic_matrix(n, k=1):
    W = np.random.random(k)
    M = np.zeros([n, n])
    
    for w in W:
        M += w * np.random.permutation(np.eye(n))
        
    return M / W.sum()
```

When `n` is small enough, we sum up all the possible permutation matrices:

```python
import math
import itertools
import numpy as np

def random_doubly_stochastic_matrix(n):
    assert n <= 5, 'it runs forever if n is too large'

    weights = np.random.random(math.factorial(n))
    weights = weights / weights.sum()
    
    matrix = np.zeros([n, n])
    perms = itertools.permutations(range(n))
    
    for idxs, weight in zip(perms, weights):
        matrix[range(n), idxs] += weight

    return matrix
```

### Algorithm 2

A recursive algorithm by randomly generating numbers matching constraints.

```python
import numpy as np
import random

def random_doubly_stochastic_matrix(n):
    if n == 1:
        return np.ones([1, 1])

    M = np.zeros([n, n])

    M[-1, -1] = random.uniform(0, 1)

    # Generate submatrix with constraint
    sM = random_doubly_stochastic_matrix(n - 1)
    sM *= (M[-1, -1] + n - 2) / (n - 1)

    M[:-1, :-1] = sM
    M[-1, :-1] = 1 - sM.sum(0)
    M[:-1, -1] = 1 - sM.sum(1)

    # Random permutation since M is symmetric
    M = M @ np.random.permutation(np.eye(n))

    return M
```


### Algorithm 3

Modified from http://people.duke.edu/~ccc14/bios-821-2017/scratch/Python07A.html.

```python
import numpy as np

def random_doubly_stochastic_matrix(n, tol=0):
    x = np.random.random((n, n))
    rsum = None
    csum = None

    while True:                              
        x /= x.sum(0)
        x = x / x.sum(1)[:, np.newaxis]
        rsum = x.sum(1)
        csum = x.sum(0)
        
        if np.any(np.abs(rsum - 1) > tol):
            continue
        if np.any(np.abs(csum - 1) > tol):
            continue
        break

    return x
```


### Algorithm 4

An algorithm by iteratively moving values from an identity matrix.

```python
import numpy as np
import random

def random_doubly_stochastic_matrix(n, k=100):
    A = np.eye(n)
    
    for _ in range(k):
        r0, r1 = random.sample(range(n), 2)

        c0_cand = np.where(A[r0] > 0)[0]
        c0 = c0_cand[random.randrange(c0_cand.size)]

        c1_cand = np.where(A[r1] > 0)[0]
        c1 = c1_cand[random.randrange(c1_cand.size)]

        if A[r0, c1] >= 1:
            continue

        if A[r1, c0] >= 1:
            continue

        num_ub = min(A[r0, c0], A[r1, c1], 1-A[r1, c0])
        num = random.uniform(0, num_ub)

        A[r0, c0] -= num
        A[r1, c1] -= num
        A[r0, c1] += num
        A[r1, c0] += num

    return A
```
