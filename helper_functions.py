# %%
import helper_functions as hf

import numpy as np
import pandas as pd

import numpy.linalg as LA
import scipy.linalg as SLA

from time import perf_counter_ns
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
def get_array(n, state = 55):
    np.random.seed(state)
    return np.random.uniform(-100,100,(n,))

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
    potential_pivot_arr = MAT[:,0]
    return hf.absolute_max_element_index(potential_pivot_arr)

# %%
def rook_pivot_index(MAT):
    degree = len(MAT)
    potential_pivot_arr = np.append(np.flip(MAT[1:,0]),MAT[0])
    max_el_index, max_el = hf.absolute_max_element_index(potential_pivot_arr)
    return max_el_index-(degree-1), max_el

# %%
def complete_pivot_index(MAT):
    degree = len(MAT)
    flat_matrix = MAT.flatten()
    flat_matrix_max_val_index, max_el = hf.absolute_max_element_index(flat_matrix)
    return flat_matrix_max_val_index // degree, flat_matrix_max_val_index % degree, max_el

# %%
def cholesky_pivot_index(MAT):
    potential_pivot_arr = [MAT[k,k] for k in range(len(MAT))]
    return hf.absolute_max_element_index(potential_pivot_arr)

#%%
def absolute_max_element_index(arr):
    arr = list(map(abs, arr))
    arr_len, max_el_index  = len(arr), 0
    max_el = arr[max_el_index]

    for index in range(1,arr_len):
        if arr[index] > max_el:
            max_el_index = index
            max_el = arr[max_el_index]

    return max_el_index, max_el

# %%
def backward_substitution(mat, arr):
    (n,m) = mat.shape
    C = np.zeros((m,))
    
    for i in range(n-1,-1,-1):
        substracting_factor = 0 if (i == n-1) else np.dot(mat[i, i+1:], C[i+1:])
        C[i] = (arr[i] - substracting_factor) / mat[i, i]
    return C

# %%
def forward_substitution(mat, arr):
    (n,m) = mat.shape
    C = np.zeros((m,))

    for i in range(n):
        substracting_factor = 0 if (i == 0) else np.dot(mat[i, :i], C[:i])
        C[i] = (arr[i] - substracting_factor) / mat[i, i]
    return C

# %%
def GEPP(A, b):
    # Initialization
    degree = len(A)
    P = hf.identity(degree)
    L, U = P.copy(), A.copy()

    start = float(perf_counter_ns())

    # decomposition starts
    for row in range(degree-1):
        max_el_index, max_el = hf.partial_pivot_index(A[row:, row:])

        if max_el < my_epsilon:                         # checks if the matrix is sigular
            return {'norm':'Singular', 'time':'Singular'}
        
        pivot_index = row + max_el_index
        if pivot_index != row:
            U = hf.swap_rows(U, row, pivot_index, from_col = row)
            if row > 0: 
                L = hf.swap_rows(L, row, pivot_index, to_col = row)
            P = hf.swap_rows(P, row, pivot_index)

        for row_below in range(row+1, degree):
            pivot_ratio = L[row_below, row] = U[row_below,row] / U[row, row]
            U[row_below,row:] = U[row_below,row:] - (pivot_ratio * U[row,row:])
    # decomposition ends

    C = hf.forward_substitution(L, P@b)         # solves for C, where, LC = Pb
    x = hf.backward_substitution(U, C)          # solves for x, where, Ux = C
    
    solution_time = float(perf_counter_ns()) - start
    return {'norm':LA.norm(A@x - b), 'time':solution_time}

# %%
def GERP(A, b):
    # Initialization
    degree = len(A)
    P = hf.identity(degree)
    Q, L, U = P.copy(), P.copy(), A.copy()

    start = float(perf_counter_ns())

    # decomposition starts
    for row in range(degree-1):
        max_el_index, max_el = hf.rook_pivot_index(A[row:, row:])
        need_row_swap, max_el_index = max_el_index<0, abs(max_el_index)

        if max_el < my_epsilon:                         # checks if the matrix is sigular
            return {'norm':'Singular', 'time':'Singular'}
        
        pivot_index = row + max_el_index

        if pivot_index != row:
            if need_row_swap:
                U = hf.swap_rows(U, row, pivot_index, from_col = row)
                if row > 0: 
                    L = hf.swap_rows(L, row, pivot_index, to_col = row)
                P = hf.swap_rows(P, row, pivot_index)
            else:
                U = hf.swap_cols(U, row, pivot_index)
                Q = hf.swap_cols(Q, row, pivot_index)

        for row_below in range(row+1, degree):
            pivot_ratio = L[row_below, row] = U[row_below,row] / U[row, row]
            U[row_below,row:] = U[row_below,row:] - (pivot_ratio * U[row,row:])
    # decomposition ends

    C = hf.forward_substitution(L, P@b)         # solves for C, where, LC = Pb
    z = hf.backward_substitution(U, C)          # solves for z, where, Uz = C
    x = Q@z
    
    solution_time = float(perf_counter_ns()) - start
    return {'norm':LA.norm(A@x - b), 'time':solution_time}

# %%
def GECP(A, b):
    # Initialization
    degree = len(A)
    P = hf.identity(degree)
    Q, L, U = P.copy(), P.copy(), A.copy()
    
    start = float(perf_counter_ns())

    # decomposition starts
    for row in range(degree-1):
        max_el_row_index, max_el_col_index, max_el = hf.complete_pivot_index(A[row:, row:])
        pivot_row_index, pivot_col_index = row + max_el_row_index, row + max_el_col_index
        # need_row_swap = max_el_row_index > row

        if max_el < my_epsilon:                         # checks if the matrix is sigular
            return {'norm':'Singular', 'time':'Singular'}
        
        if pivot_col_index != row:
            U = hf.swap_cols(U, row, pivot_col_index)
            Q = hf.swap_cols(Q, row, pivot_col_index)

        if pivot_row_index != row:
            U = hf.swap_rows(U, row, pivot_row_index)
            if row > 0: 
                L = hf.swap_rows(L, row, pivot_row_index)
            P = hf.swap_rows(P, row, pivot_row_index)

        for row_below in range(row+1, degree):
            pivot_ratio = L[row_below, row] = U[row_below,row] / U[row, row]
            U[row_below,row:] = U[row_below,row:] - (pivot_ratio * U[row,row:])
    # decomposition ends

    C = hf.forward_substitution(L, P@b)         # solves for C, where, LC = Pb
    z = hf.backward_substitution(U, C)          # solves for z, where, Uz = C
    x = Q@z
    
    solution_time = float(perf_counter_ns()) - start
    return {'norm':LA.norm(A@x - b), 'time':solution_time}

# %%
def CHOP(A, b):
    degree = len(A)
    A_ = A.copy()
    P = hf.identity(degree)
    L = np.zeros(A.shape)

    if min([abs(A[k,k]) for k in range(degree)]) < my_epsilon:
        return {'norm':'Singular', 'time':'Singular'}
    
    start = float(perf_counter_ns())
    for row in range(degree-1):
        max_el_index, max_el = hf.cholesky_pivot_index(A_[row:,row:])
        pivot_index = row + max_el_index
        if pivot_index != row:
            A_ = hf.swap_cols(hf.swap_rows(A_, pivot_index, row),pivot_index, row)
            P = hf.swap_rows(P, pivot_index, row)
        
        diagonal_substracting_factor = 0 if (row==0) else np.dot(L[row,:row], L[row,:row])
        L[row,row] = math.sqrt(A_[row,row] - diagonal_substracting_factor)

        for row_below in range(row+1,degree):
            non_diagonal_substracting_factor = 0 if (row==0) else np.dot(L[row_below,:row], L[row,:row])
            L[row_below, row] = (A_[row_below, row] - non_diagonal_substracting_factor) / L[row, row]
        print(L)
    
    C = hf.forward_substitution(L, P@b)         # solves for C, where, L*C = P*b
    z = hf.backward_substitution(L.T, C)        # solves for z, here, trans(L)*z = C
    x = P.T@z                                   # solves for x, here, x = Q*z = trans(P)*z

    solution_time = float(perf_counter_ns()) - start
    return {'norm':LA.norm(A@x - b), 'time':solution_time}

# %%
def SCIPY(A, b):
    try:
        start = float(perf_counter_ns())
        x = SLA.solve(A, b)
        solution_time = float(perf_counter_ns()) - start
        return {'norm':LA.norm(A@x - b), 'time':solution_time}
    
    except SLA.LinAlgError:
        return {'norm':'Singular', 'time':'Singular'}
    
# %%
def comparision(degree_arr):
    columns = ['GEPP', 'GERP', 'GECP', 'CHOP', 'SCIPY']

    norm_df = pd.DataFrame(columns=columns, index=degree_arr)
    norm_df.index.names = ['Deg']
    time_df = norm_df.copy()

    for degree in degree_arr:
        A, b = hf.get_matrix(degree), hf.get_array(degree)
        norms, times = [], []
        for operation in columns:
            result = eval(operation+'(A,b)')
            norms.append(result['norm'])
            times.append(result['time'])
        norm_df.loc[degree] = norms
        time_df.loc[degree] = times
    return norm_df, time_df
