# %%
import numpy as np
import helper_functions as hf
import numpy.linalg as LA
import scipy.linalg as SLA
import math

machine_epsilon = np.finfo('float').eps
my_epsilon = machine_epsilon * 10

# %%
def identity(n):
    return np.eye(n)

# %%
def get_matrix(n):
    return np.array([[1/(i+j+1) for i in range(n)] for j in range(n)])

# %%
def get_array(n, random_state = 55):
    return np.random.uniform(-100,100,(n,),random_state=random_state)

# %%
def swap_rows(MAT, diagonal_row, pivot_row, from_col = 0, to_col = -1):
    if to_col == -1: 
        to_col = len(MAT)
    MAT[[diagonal_row, pivot_row], from_col:to_col] = MAT[[pivot_row, diagonal_row], from_col:to_col]
    return MAT

# %%
def swap_cols(MAT, diagonal_col, pivot_col, from_row = 0, to_row = -1):
    if to_row == -1: 
        to_row = len(MAT)
    MAT[from_row:to_row, [diagonal_col, pivot_col]] = MAT[from_row:to_row, [pivot_col, diagonal_col]]
    return MAT

# %%
def partial_pivot_index(MAT):
    rook = rook_pivot_index(MAT)
    return 0 if (rook >= 0) else abs(rook)

# %%
def rook_pivot_index(MAT):
    degree = len(MAT)
    potential_pivot_arr = np.append(np.flip(MAT[1:,0]),MAT[0])
    return absolute_max_element_index(potential_pivot_arr)-(degree-1)

# %%
def complete_pivoting(MAT):
    degree = len(MAT)
    flat_matrix = MAT.flatten()
    flat_matrix_max_val_index = absolute_max_element_index(flat_matrix)
    row_index, col_index = flat_matrix_max_val_index // degree, flat_matrix_max_val_index % degree
    return row_index, col_index

#%%
def absolute_max_element_index(arr):
    arr = list(map(abs, arr))
    arr_len, max_el_index  = len(arr), 0
    max_el = arr[max_el_index]

    for index in range(1,arr_len):
        if arr[index] > max_el:
            max_el_index = index
            max_el = arr[max_el_index]
    return max_el_index

# %%
def backward_substitution(mat, arr):
    (n,m) = mat.shape               # n X n square matrix
    C = np.zeros((m,))              # initialization of C
    
    for i in range(n-1,-1,-1):
        substracting_factor = 0 if (i == n-1) else np.dot(mat[i, i+1:], C[i+1:])
        C[i] = (arr[i] - substracting_factor) / mat[i, i]
    return C

# %%
def forward_substitution(mat, arr):
    (n,m) = mat.shape               # n X n square matrix
    C = np.zeros((m,))              # initialization of C

    for i in range(n):
        substracting_factor = 0 if (i == 0) else np.dot(mat[i, :i], C[:i])
        C[i] = (arr[i] - substracting_factor) / mat[i, i]
    return C
