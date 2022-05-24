
import sys
import os
import numpy as np
import sympy as sp

import tabulate as tab


#link to documentation with examples: 
#


# set model name
model_name = "Overhead Crane"


# ---------- create symbolic parameters
pp_symb = [m, M, l, g] = sp.symbols("m, M, l, g", real=True)




# ---------- create symbolic parameter functions
# parameter values can be constant/fixed values OR set in relation to other parameters (for example: a = 2*b)  
m_sf = 3000
M_sf = 8000
l_sf = 2
g_sf = 9.81

# list of symbolic parameter functions
# trailing "_sf" stands for "symbolic parameter function"
pp_sf = [m_sf, M_sf, l_sf, g_sf]


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
tabular_header = ["Parameter Name", "Symbol", "Value", "Unit"]

# Define column text alignments
col_alignment = ["left", "center", "left", "center"]


# Define Entries of all columns before the Symbol-Column
# --- Entries need to be latex code
col_1 = ["mass of the load",
         "mass of the wagon",
         "rope length", 
         "acceleration due to gravitation"] 

# contains all lists of the columns before the "Symbol" Column
# --- Empty list, if there are no columns before the "Symbol" Column
start_columns_list = [col_1]


# Define Entries of the columns after the Value-Column
# --- Entries need to be latex code
col_4 = ["kg",
         "kg",
         "m",
         r"$\frac{m}{s^2}$"
         ]

# contains all lists of columns after the FIX ENTRIES
# --- Empty list, if there are no columns after the "Value" column
end_columns_list = [col_4]