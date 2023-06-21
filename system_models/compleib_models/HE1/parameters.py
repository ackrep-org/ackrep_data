# This file was autogenerated from the template: parameters.py.template (2022-10-10 15:53:24).

import sys
import os
import numpy as np
import sympy as sp

import tabulate as tab


#link to documentation with examples: https://ackrep-doc.readthedocs.io/en/latest/devdoc/contributing_data.html


# set model name
model_name = 'Longitudinal motion of a VTOL helicopter'


# ---------- create symbolic parameters
A = sp.MatrixSymbol('A', 4, 4)
B = sp.MatrixSymbol('B', 4, 2)
B1 = sp.MatrixSymbol('B1', 4, 2)
C1 = sp.MatrixSymbol('C1', 2, 4)
C = sp.MatrixSymbol('C', 1, 4)
D11 = sp.MatrixSymbol('D11', 2, 2)
D12 = sp.MatrixSymbol('D12', 2, 2)
D21 = sp.MatrixSymbol('D21', 1, 2)

pp_symb = [A, B, B1, C1, C, D11, D12, D21]


# ---------- create auxiliary symbolic parameters 

# set numerical values of auxiliary parameters
# trailing "_nv" stands for "numerical value"
A_nv = sp.Matrix(np.array([[-3.6600e-02,  2.7100e-02,  1.8800e-02, -4.5550e-01],
       [ 4.8200e-02, -1.0100e+00,  2.4000e-03, -4.0208e+00],
       [ 1.0020e-01,  3.6810e-01, -7.0700e-01,  1.4200e+00],
       [ 0.0000e+00,  0.0000e+00,  1.0000e+00,  0.0000e+00]]))
B_nv = sp.Matrix(np.array([[ 0.4422,  0.1761],
       [ 3.5446, -7.5922],
       [-5.52  ,  4.49  ],
       [ 0.    ,  0.    ]]))
B1_nv = sp.Matrix(np.array([[ 0.4422,  0.1761],
       [ 3.5446, -7.5922],
       [-5.52  ,  4.49  ],
       [ 0.    ,  0.    ]]))
C1_nv = sp.Matrix(np.array([[1.41421356, 0.        , 0.        , 0.        ],
       [0.        , 0.70710678, 0.        , 0.        ]]))
C_nv = sp.Matrix(np.array([[0., 1., 0., 0.]]))
D11_nv = sp.Matrix(np.array([[0., 0.],
       [0., 0.]]))
D12_nv = sp.Matrix(np.array([[0.70710678, 0.        ],
       [0.        , 0.70710678]]))
D21_nv = sp.Matrix(np.array([[0., 0.]]))


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