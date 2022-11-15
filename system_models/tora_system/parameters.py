
import sys
import os
import numpy as np
import sympy as sp

import tabulate as tab


#link to documentation with examples: https://ackrep-doc.readthedocs.io/en/latest/devdoc/contributing_data.html


# set model name
model_name = "tora"


# ---------- create symbolic parameters
pp_symb = [m1, m2, l1, J1, a] = sp.symbols("m1, m2, l1, J1, a", real=True)


# ---------- create symbolic parameter functions
# parameter values can be constant/fixed values OR set in relation to other parameters (for example: a = 2*b)  
m1_sf = 5 
m2_sf = 0.8
l1_sf = 0.5 
J1_sf = 0.2
a_sf = 3

# list of symbolic parameter functions
# tailing "_sf" stands for "symbolic parameter function"
pp_sf = [m1_sf, m2_sf, l1_sf, J1_sf, a_sf]


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
col_1 = [
        "mass of the cart",
        "mass of the pendulum",
        "length of the pendulum",
        "moment of inertia of the pendulum",
        "spring constant"] 

# contains all lists of the columns before the "Symbol" Column
# --- Empty list, if there are no columns before the "Symbol" Column
start_columns_list = [col_1]


# Define Entries of the columns after the Value-Column
# --- Entries need to be latex code
col_4 = [
        "kg",
        "kg",
        "m",
        r"$kg \cdot m^2$",
        r"$\frac{N}{m}$"]

# contains all lists of columns after the FIX ENTRIES
# --- Empty list, if there are no columns after the "Value" column
end_columns_list = [col_4]