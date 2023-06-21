# This file was autogenerated from the template: parameters.py.template (2022-10-10 15:54:06).

import sys
import os
import numpy as np
import sympy as sp

import tabulate as tab


#link to documentation with examples: https://ackrep-doc.readthedocs.io/en/latest/devdoc/contributing_data.html


# set model name
model_name = '2D heat flow example, Distributed control of a perturbed linear heat flow model'


# ---------- create symbolic parameters
A = sp.MatrixSymbol('A', 5, 5)
B = sp.MatrixSymbol('B', 5, 2)
B1 = sp.MatrixSymbol('B1', 5, 5)
C1 = sp.MatrixSymbol('C1', 7, 5)
C = sp.MatrixSymbol('C', 2, 5)
D11 = sp.MatrixSymbol('D11', 7, 5)
D12 = sp.MatrixSymbol('D12', 7, 2)
D21 = sp.MatrixSymbol('D21', 2, 5)

pp_symb = [A, B, B1, C1, C, D11, D12, D21]


# ---------- create auxiliary symbolic parameters 

# set numerical values of auxiliary parameters
# trailing "_nv" stands for "numerical value"
A_nv = sp.Matrix(np.array([[ 2.79156070e-01,  3.53341393e-02, -3.36252593e-02,
        -4.05852270e-02, -3.87993290e-02],
       [ 3.53341393e-02, -6.86688166e-01,  1.81993593e+00,
         2.20137115e+00,  2.14486124e+00],
       [-3.36252593e-02,  1.81993593e+00, -7.84928059e+00,
        -1.22513051e+01, -1.24552902e+01],
       [-4.05852270e-02,  2.20137115e+00, -1.22513051e+01,
        -2.23365878e+01, -2.57559793e+01],
       [-3.87993290e-02,  2.14486124e+00, -1.24552902e+01,
        -2.57559793e+01, -3.54802758e+01]]))
B_nv = sp.Matrix(np.array([[ 5.35849991,  4.99511968],
       [ 4.2215687 , -2.34595033],
       [-0.23358097,  1.5963443 ],
       [ 3.02914089, -1.74518392],
       [-3.14753151,  0.69561472]]))
B1_nv = sp.Matrix(np.array([[ 5.35849991,  4.99511968],
       [ 4.2215687 , -2.34595033],
       [-0.23358097,  1.5963443 ],
       [ 3.02914089, -1.74518392],
       [-3.14753151,  0.69561472]]))
C1_nv = sp.Matrix(np.array([[0.70710678, 0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.70710678, 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.70710678, 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.70710678, 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.70710678],
       [0.        , 0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ]]))
C_nv = sp.Matrix(np.array([[ 0.30567963,  0.62302542,  0.02301636,  0.511443  , -0.58763132],
       [ 5.5803084 , -3.67006331,  0.15921163, -4.22908314,  3.95323   ]]))
D11_nv = sp.Matrix(np.array([[0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.]]))
D12_nv = sp.Matrix(np.array([[0.        , 0.        ],
       [0.        , 0.        ],
       [0.        , 0.        ],
       [0.        , 0.        ],
       [0.        , 0.        ],
       [0.70710678, 0.        ],
       [0.        , 0.70710678]]))
D21_nv = sp.Matrix(np.array([[0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.]]))


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