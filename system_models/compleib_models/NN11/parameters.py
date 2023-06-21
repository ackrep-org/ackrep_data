# This file was autogenerated from the template: parameters.py.template (2022-10-10 15:54:04).

import sys
import os
import numpy as np
import sympy as sp

import tabulate as tab


#link to documentation with examples: https://ackrep-doc.readthedocs.io/en/latest/devdoc/contributing_data.html


# set model name
model_name = 'NN11'


# ---------- create symbolic parameters
A = sp.MatrixSymbol('A', 16, 16)
B = sp.MatrixSymbol('B', 16, 3)
B1 = sp.MatrixSymbol('B1', 16, 3)
C1 = sp.MatrixSymbol('C1', 3, 16)
C = sp.MatrixSymbol('C', 5, 16)
D11 = sp.MatrixSymbol('D11', 3, 3)
D12 = sp.MatrixSymbol('D12', 3, 3)
D21 = sp.MatrixSymbol('D21', 5, 3)

pp_symb = [A, B, B1, C1, C, D11, D12, D21]


# ---------- create auxiliary symbolic parameters 

# set numerical values of auxiliary parameters
# trailing "_nv" stands for "numerical value"
A_nv = sp.Matrix(np.array([[-1.010000e+02, -9.990000e+01,  0.000000e+00,  0.000000e+00,
         0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
         0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
         0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00],
       [ 0.000000e+00, -1.010000e+02,  0.000000e+00,  0.000000e+00,
         0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
         0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
         0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00],
       [ 0.000000e+00,  0.000000e+00, -1.010000e+02, -9.990000e+01,
         0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
         0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
         0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00],
       [ 0.000000e+00,  0.000000e+00,  0.000000e+00, -1.010000e+02,
         0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
         0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
         0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00],
       [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
        -1.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
         1.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
         0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00],
       [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
         0.000000e+00, -1.000000e+00,  0.000000e+00,  0.000000e+00,
         0.000000e+00,  1.000000e+00,  0.000000e+00,  0.000000e+00,
         0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00],
       [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
         0.000000e+00,  0.000000e+00, -1.000000e+00,  0.000000e+00,
         0.000000e+00,  0.000000e+00,  1.000000e+00,  0.000000e+00,
         0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00],
       [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
         0.000000e+00,  0.000000e+00,  0.000000e+00, -1.000000e+00,
         0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00,
         0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00],
       [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
         0.000000e+00,  0.000000e+00,  4.270980e+02, -4.683410e+01,
        -1.000000e+00,  0.000000e+00,  4.271000e-01, -4.680000e-02,
         0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00],
       [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
         0.000000e+00,  0.000000e+00,  2.320719e+02,  1.204649e+02,
         0.000000e+00, -1.000000e+00,  2.321000e-01,  1.205000e-01,
         0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00],
       [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
         0.000000e+00,  0.000000e+00, -7.642456e+02,  8.541540e+01,
         0.000000e+00,  0.000000e+00, -1.764200e+00,  8.540000e-02,
         0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00],
       [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
         0.000000e+00,  0.000000e+00,  1.668270e+02, -2.647739e+02,
         0.000000e+00,  0.000000e+00,  1.668000e-01, -1.264800e+00,
         0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00],
       [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
         3.162000e-01,  0.000000e+00,  0.000000e+00,  0.000000e+00,
         0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
        -1.100000e+00, -7.590000e-02,  0.000000e+00,  0.000000e+00],
       [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
        -1.250000e-01,  0.000000e+00,  0.000000e+00,  0.000000e+00,
         0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
         0.000000e+00, -1.000000e+00,  0.000000e+00,  0.000000e+00],
       [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
         0.000000e+00,  3.162000e-01,  0.000000e+00,  0.000000e+00,
         0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
         0.000000e+00,  0.000000e+00, -1.100000e+00, -7.590000e-02],
       [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
         0.000000e+00, -1.250000e-01,  0.000000e+00,  0.000000e+00,
         0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
         0.000000e+00,  0.000000e+00,  0.000000e+00, -1.000000e+00]]))
B_nv = sp.Matrix(np.array([[ 0.    , -9.995 ,  0.    ],
       [ 0.199 , -9.995 ,  0.    ],
       [ 0.211 ,  0.    , -9.995 ],
       [-0.233 ,  0.    , -9.995 ],
       [ 0.    ,  0.    ,  0.    ],
       [ 0.    ,  0.    ,  0.    ],
       [ 0.    ,  0.    ,  0.    ],
       [ 0.    ,  0.    ,  0.    ],
       [ 0.    ,  2.7173,  1.4274],
       [ 0.    ,  1.4274,  2.8382],
       [ 0.    , -4.7909, -2.6032],
       [ 0.    ,  1.0261, -2.6393],
       [ 0.11  ,  0.    ,  0.    ],
       [ 0.    ,  0.    ,  0.    ],
       [ 0.    ,  0.    ,  0.    ],
       [ 0.01  ,  0.    ,  0.    ]]))
B1_nv = sp.Matrix(np.array([[ 0.    , -9.995 ,  0.    ],
       [ 0.199 , -9.995 ,  0.    ],
       [ 0.211 ,  0.    , -9.995 ],
       [-0.233 ,  0.    , -9.995 ],
       [ 0.    ,  0.    ,  0.    ],
       [ 0.    ,  0.    ,  0.    ],
       [ 0.    ,  0.    ,  0.    ],
       [ 0.    ,  0.    ,  0.    ],
       [ 0.    ,  2.7173,  1.4274],
       [ 0.    ,  1.4274,  2.8382],
       [ 0.    , -4.7909, -2.6032],
       [ 0.    ,  1.0261, -2.6393],
       [ 0.11  ,  0.    ,  0.    ],
       [ 0.    ,  0.    ,  0.    ],
       [ 0.    ,  0.    ,  0.    ],
       [ 0.01  ,  0.    ,  0.    ]]))
C1_nv = sp.Matrix(np.array([[ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  1.5564e+00,  3.4834e+00,  0.0000e+00,  0.0000e+00,
         1.6000e-03,  3.5000e-03,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00],
       [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00, -4.7430e-01,  0.0000e+00,  0.0000e+00,
         0.0000e+00],
       [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00, -3.4790e-01,
         0.0000e+00]]))
C_nv = sp.Matrix(np.array([[ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00, -3.1620e-01,  0.0000e+00,  0.0000e+00,
         0.0000e+00],
       [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00, -3.1620e-01,
         0.0000e+00],
       [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  1.5564e+00,  3.4834e+00,  0.0000e+00,  0.0000e+00,
         1.6000e-03,  3.5000e-03,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00],
       [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00, -4.7430e-01,  0.0000e+00,  0.0000e+00,
         0.0000e+00],
       [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00, -3.4790e-01,
         0.0000e+00]]))
D11_nv = sp.Matrix(np.array([[0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.]]))
D12_nv = sp.Matrix(np.array([[0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.]]))
D21_nv = sp.Matrix(np.array([[0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.],
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