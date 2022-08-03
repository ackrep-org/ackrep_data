
import sys
import os
import numpy as np
import sympy as sp

import tabulate as tab


#link to documentation with examples: https://ackrep-doc.readthedocs.io/en/latest/devdoc/contributing_data.html


# set model name
model_name = ".."


# ---------- create symbolic parameters
pp_symb = [.., .., ..] = sp.symbols(".., .., ..", real=True)


# ---------- create auxiliary symbolic parameters 
# (parameters, which shall not be numerical represented in the parameter table)
.. = sp.Symbol('..')

# set numerical values of auxiliary parameters
# tailing "_nv" stands for "numerical value"
.._nv = ..


# ---------- create symbolic parameter functions
# parameter values can be constant/fixed values OR set in relation to other parameters (for example: a = 2*b)  
.._sf = ..

# list of symbolic parameter functions
# tailing "_sf" stands for "symbolic parameter function"
pp_sf = [.._sf, .._sf, .._sf]


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