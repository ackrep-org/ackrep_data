import sys
import os
import numpy as np
import sympy as sp

import tabulate as tab


# link to documentation with examples: https://ackrep-doc.readthedocs.io/en/latest/devdoc/contributing_data.html


# set model name
model_name = "Hysteresis system"


# ---------- create symbolic parameters
pp_symb = [s1, s2, y1, y2, T_storage] = sp.symbols("s1, s2, y1, y2, T_storage", real=True)


# ---------- create symbolic parameter functions
# parameter values can be constant/fixed values OR set in relation to other parameters (for example: a = 2*b)
s1_sf = 4
s2_sf = 8
y1_sf = 2
y2_sf = 11
T_storage_sf = 1e-4


# list of symbolic parameter functions
# tailing "_sf" stands for "symbolic parameter function"
pp_sf = [s1_sf, s2_sf, y1_sf, y2_sf, T_storage_sf]


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
