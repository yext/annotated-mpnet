# Cython function to do permutation faster than typical Python

import numpy as np

cimport cython
cimport numpy as np

# Define the main datatype using typical Cython convention
# We're basically establishing the data type for all our numpy arrays and then simultaneously 
# setting the data type for the compile-time arrays (which are those that end in '_t')
DTYPE = np.int64
ctypedef np.int64_t DTYPE_t

# Disable bounds checking and negative indexing to make this function as fast as possible 
# Since we know exactly what the function is doing, there's no need to have these checks in place
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray make_span_perm(np.ndarray perm, list word_begins_idx, int n):
    # Define all the necessary variable according to Cython convention
    cdef np.ndarray spans
    cdef int i, g, j, start, end
    spans = np.zeros(n, dtype=np.int64)

    # The full detail of the below loops are described in data.mpnet_data lines 249 to 278
    g = 0
    for i in range(len(word_begins_idx) - 1):
        start = word_begins_idx[perm[i]]
        end = word_begins_idx[perm[i] + 1]
        for j in range(start, end):
            spans[g] = j
            g = g + 1
    
    return spans
