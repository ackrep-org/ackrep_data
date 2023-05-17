import sys
import os
import numpy as np
import sympy as sp

import tabulate as tab


# link to documentation with examples: https://ackrep-doc.readthedocs.io/en/latest/devdoc/contributing_data.html


# set model name
model_name = "distillation column"


# ---------- create symbolic parameters
pp_symb = [KR1, TN1, KR2, TN2, T1, K1, K2, K3, K4] = sp.symbols("KR1, TN1, KR2, TN2, T1, K1, K2, K3, K4", real=True)


# ---------- create symbolic parameter functions
# parameter values can be constant/fixed values OR set in relation to other parameters (for example: a = 2*b)
KR1_sf = 1.7
TN1_sf = 1.29

KR2_sf = 0.57
TN2_sf = 1.29

### Plant
T1_sf = 1.0

# equilibrium pojnt 1
K1_sf, K2_sf, K3_sf, K4_sf = 0.4, 1.2, -0.8, -0.2

# equilibrium point 2
# K1_sf, K2_sf, K3_sf, K4_sf = 0.4, 1.2, -1.28, -0.32

# switch of coupling:
# K3_sf, K4_sf = 0,0


# list of symbolic parameter functions
# tailing "_sf" stands for "symbolic parameter function"
pp_sf = [KR1_sf, TN1_sf, KR2_sf, TN2_sf, T1_sf, K1_sf, K2_sf, K3_sf, K4_sf]


#  ---------- list for substitution
# -- entries are tuples like: (independent symbolic parameter, numerical value)
pp_subs_list = []


# OPTONAL: Dictionary which defines how certain variables shall be written
# in the table - key: Symbolic Variable, Value: LaTeX Representation/Code
# useful for example for complex variables: {Z: r"\underline{Z}"}
latex_names = {KR1: r"K_{R1}", TN1: r"T_{N1}", KR2: r"K_{R2}", TN2: r"T_{N2}"}


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
