
import sys
import os
import numpy as np
import sympy as sp

import tabulate as tab


#link to documentation with examples: https://ackrep-doc.readthedocs.io/en/latest/devdoc/contributing_data.html


# set model name
model_name = "Permanent magnet DC motor"


# ---------- create symbolic parameters
pp_symb = [c_phi, J, La, Ra] = sp.symbols("c_phi, J, La, Ra", real=True)


# ---------- create symbolic parameter functions
# parameter values can be constant/fixed values OR set in relation to other parameters (for example: a = 2*b)  
c_phi_sf = 0.169
J_sf = 1.7e-3
La_sf = 0.0256
Ra_sf = 3.2

# list of symbolic parameter functions
# tailing "_sf" stands for "symbolic parameter function"
pp_sf = [c_phi_sf, J_sf, La_sf, Ra_sf]


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
        "motor constant",
        "moment of inertia",
        "armature inductance",
        "armature resistance"] 

# contains all lists of the columns before the "Symbol" Column
# --- Empty list, if there are no columns before the "Symbol" Column
start_columns_list = [col_1]


# Define Entries of the columns after the Value-Column
# --- Entries need to be latex code
col_4 = [ 
        "Vs",
        r"$Ws^3$",
        "H",
        r"$\Omega$"]

# contains all lists of columns after the FIX ENTRIES
# --- Empty list, if there are no columns after the "Value" column
end_columns_list = [col_4]