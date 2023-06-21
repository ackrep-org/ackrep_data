# This file was autogenerated from the template: parameters.py.template (2022-10-10 15:53:25).

import sys
import os
import numpy as np
import sympy as sp

import tabulate as tab


#link to documentation with examples: https://ackrep-doc.readthedocs.io/en/latest/devdoc/contributing_data.html


# set model name
model_name = 'Helicopter control'


# ---------- create symbolic parameters
A = sp.MatrixSymbol('A', 8, 8)
B = sp.MatrixSymbol('B', 8, 4)
B1 = sp.MatrixSymbol('B1', 8, 8)
C1 = sp.MatrixSymbol('C1', 12, 8)
C = sp.MatrixSymbol('C', 6, 8)
D11 = sp.MatrixSymbol('D11', 12, 8)
D12 = sp.MatrixSymbol('D12', 12, 4)
D21 = sp.MatrixSymbol('D21', 6, 8)

pp_symb = [A, B, B1, C1, C, D11, D12, D21]


# ---------- create auxiliary symbolic parameters 

# set numerical values of auxiliary parameters
# trailing "_nv" stands for "numerical value"
A_nv = sp.Matrix(np.array([[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         9.98573780e-01,  5.33842742e-02,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00],
       [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00,
        -3.18221934e-03,  5.95246553e-02,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00],
       [ 0.00000000e+00,  0.00000000e+00, -1.15704956e+01,
        -2.54463768e+00, -6.36026263e-02,  1.06780529e-01,
        -9.49186683e-02,  7.10757449e-03],
       [ 0.00000000e+00,  0.00000000e+00,  4.39356565e-01,
        -1.99818230e+00,  0.00000000e+00,  1.66518837e-02,
         1.84620470e-02, -1.18747074e-03],
       [ 0.00000000e+00,  0.00000000e+00, -2.04089546e+00,
        -4.58999157e-01, -7.35027790e-01,  1.92557573e-02,
        -4.59562242e-03,  2.12036073e-03],
       [-3.21036072e+01,  0.00000000e+00, -5.03355026e-01,
         2.29785919e+00,  0.00000000e+00, -2.12158114e-02,
        -2.11679190e-02,  1.58115923e-02],
       [ 1.02161169e-01,  3.20578308e+01, -2.34721756e+00,
        -5.03611565e-01,  8.34947586e-01,  2.12265700e-02,
        -3.78797352e-02,  3.54003860e-04],
       [-1.91097260e+00,  1.71382904e+00, -4.00543213e-03,
        -5.74111938e-02,  0.00000000e+00,  1.39896348e-02,
        -9.06753354e-04, -2.90513515e-01]]))
B_nv = sp.Matrix(np.array([[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00],
       [ 1.24335051e-01,  8.27858448e-02, -2.75247765e+00,
        -1.78887695e-02],
       [-3.63589227e-02,  4.75095272e-01,  1.42907426e-02,
         0.00000000e+00],
       [ 3.04491520e-01,  1.49580166e-02, -4.96518373e-01,
        -2.06741929e-01],
       [ 2.87735462e-01, -5.44506073e-01, -1.63793564e-02,
         0.00000000e+00],
       [-1.90734863e-02,  1.63674355e-02, -5.44536114e-01,
         2.34842300e-01],
       [-4.82063293e+00, -3.81469727e-04,  0.00000000e+00,
         0.00000000e+00]]))
B1_nv = sp.Matrix(np.array([[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00],
       [ 1.24335051e-01,  8.27858448e-02, -2.75247765e+00,
        -1.78887695e-02],
       [-3.63589227e-02,  4.75095272e-01,  1.42907426e-02,
         0.00000000e+00],
       [ 3.04491520e-01,  1.49580166e-02, -4.96518373e-01,
        -2.06741929e-01],
       [ 2.87735462e-01, -5.44506073e-01, -1.63793564e-02,
         0.00000000e+00],
       [-1.90734863e-02,  1.63674355e-02, -5.44536114e-01,
         2.34842300e-01],
       [-4.82063293e+00, -3.81469727e-04,  0.00000000e+00,
         0.00000000e+00]]))
C1_nv = sp.Matrix(np.array([[1., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0.],
       [0., 0., 1., 0., 0., 0., 0., 0.],
       [0., 0., 0., 1., 0., 0., 0., 0.],
       [0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 0., 0., 0., 0., 1., 0., 0.],
       [0., 0., 0., 0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 0., 0., 0., 1.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.]]))
C_nv = sp.Matrix(np.array([[ 0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.0595 ,
         0.05329, -0.9968 ],
       [ 1.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,
         0.     ,  0.     ],
       [ 0.     ,  1.     ,  0.     ,  0.     ,  0.     ,  0.     ,
         0.     ,  0.     ],
       [ 0.     ,  0.     ,  0.     , -0.05348,  1.     ,  0.     ,
         0.     ,  0.     ],
       [ 0.     ,  0.     ,  1.     ,  0.     ,  0.     ,  0.     ,
         0.     ,  0.     ],
       [ 0.     ,  0.     ,  0.     ,  1.     ,  0.     ,  0.     ,
         0.     ,  0.     ]]))
D11_nv = sp.Matrix(np.array([[0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.]]))
D12_nv = sp.Matrix(np.array([[0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [1., 0., 0., 0.],
       [0., 1., 0., 0.],
       [0., 0., 1., 0.],
       [0., 0., 0., 1.]]))
D21_nv = sp.Matrix(np.array([[0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.]]))


# ---------- create symbolic parameter functions
# parameter values can be constant/fixed values OR set in relation to other parameters (for example: a = 2*b)  


# list of symbolic parameter functions
# tailing "_sf" stands for "symbolic parameter function"
pp_sf = [A_nv, B_nv, B1_nv, C1_nv, C_nv, D11_nv, D12_nv, D21_nv]


#  ---------- list for substitution
# -- entries are tuples like: (independent symbolic parameter, numerical value)
pp_subs_list = []


# OPTONAL: Dictionary which defines how certain variables shall be written
# in the table - key: Symbolic Variable, Value: LaTeX Representation/Code
# useful for example for complex variables: {Z: r"\underline{Z}"}
latex_names = {}


# ---------- Define LaTeX table

# Define table header 
# DON'T CHANGE FOLLOWING ENTRIES: "Symbol", "Value"
tabular_header = ["Symbol", "Value"]

# Define column text alignments
col_alignment = ["center", "left"]


# Define Entries of all columns before the Symbol-Column
# --- Entries need to be latex code
col_1 = [] 

# contains all lists of the columns before the "Symbol" Column
# --- Empty list, if there are no columns before the "Symbol" Column
start_columns_list = []


# Define Entries of the columns after the Value-Column
# --- Entries need to be latex code
col_4 = []

# contains all lists of columns after the FIX ENTRIES
# --- Empty list, if there are no columns after the "Value" column
end_columns_list = []