# This file was autogenerated from the template: parameters.py.template (2022-10-10 15:53:28).

import sys
import os
import numpy as np
import sympy as sp

import tabulate as tab


#link to documentation with examples: https://ackrep-doc.readthedocs.io/en/latest/devdoc/contributing_data.html


# set model name
model_name = 'Decentralized system with 2 control stations'


# ---------- create symbolic parameters
A = sp.MatrixSymbol('A', 3, 3)
B = sp.MatrixSymbol('B', 3, 2)
B1 = sp.MatrixSymbol('B1', 3, 3)
C1 = sp.MatrixSymbol('C1', 3, 3)
C = sp.MatrixSymbol('C', 2, 3)
D11 = sp.MatrixSymbol('D11', 3, 3)
D12 = sp.MatrixSymbol('D12', 3, 2)
D21 = sp.MatrixSymbol('D21', 2, 3)

pp_symb = [A, B, B1, C1, C, D11, D12, D21]


# ---------- create auxiliary symbolic parameters 

# set numerical values of auxiliary parameters
# trailing "_nv" stands for "numerical value"
A_nv = sp.Matrix(np.array([[-4.,  2.,  1.],
       [ 3., -2.,  5.],
       [-7.,  0.,  3.]]))
B_nv = sp.Matrix(np.array([[1., 0.],
       [1., 0.],
       [0., 1.]]))
B1_nv = sp.Matrix(np.array([[1., 0.],
       [1., 0.],
       [0., 1.]]))
C1_nv = sp.Matrix(np.array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]]))
C_nv = sp.Matrix(np.array([[0., 1., 0.],
       [0., 0., 1.]]))
D11_nv = sp.Matrix(np.array([[0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.]]))
D12_nv = sp.Matrix(np.array([[0., 0.],
       [1., 0.],
       [0., 1.]]))
D21_nv = sp.Matrix(np.array([[0., 0., 0.],
       [0., 0., 0.]]))


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