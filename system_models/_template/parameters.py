
import sys
import os
import numpy as np
import sympy as sp

import tabulate as tab


# Trailing "_nv" stands for "numerical value"

# SET MODEL NAME
model_name = ".."


# ---------- CREATE SYMBOLIC PARAMETERS
pp_symb = [..,..,..] = sp.symbols('..,..,..', real = True)

# symbolic parameter functions
# parameter values can be constant/fixed values OR set in relation to other parameters (for example: a = 2*b)  
.._sf = ..

# List of symbolic parameter functions
pp_sf = [.._sf,.._sf,.._sf]



# ---------- CREATE AUXILIARY SYMBOLIC PARAMETERS 
# (parameters, which shall not be numerical represented in the parameter tabular)
.. = sp.Symbol('..')

# Set numerical values of auxiliary parameters
.._nv = ..



#  ---------- LIST FOR SUBSTITUTION
# -- Entries are tuples like: (independent symbolic parameter, numerical value)
pp_subs_list = []



# OPTONAL: Dictionary which defines how certain variables shall be written
# in the tabular - key: Symbolic Variable, Value: LaTeX Representation/Code
# useful for example for complex variables: {Z: r"\underline{Z}"}
latex_names = {}



# ---------- CREATE BEGIN OF LATEX TABULAR

# Define tabular Header 
# DON'T CHANGE FOLLOWING ENTRIES: "Symbol", "Value"
tabular_header = ["..", "Symbol", "Value", "..."]

# Define column text alignments
col_alignment = ["...", "center", "left", "..."]


# Define Entries of all columns before the Symbol-Column
# --- Entries need to be latex code
col_1 = [".."
         ] 

# contains all lists of the columns before the "Symbol" Column
# --- Empty list, if there are no columns before the "Symbol" Column
start_columns_list = [col_1]


# Define Entries of the columns after the Value-Column
# --- Entries need to be latex code
col_4 = ["..."
         ]

# contains all lists of columns after the FIX ENTRIES
# --- Empty list, if there are no columns after the "Value" column
end_columns_list = [col_4]

