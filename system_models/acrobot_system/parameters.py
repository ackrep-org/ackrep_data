
import sys
import os
import numpy as np
import sympy as sp

import tabulate as tab


#link to documentation with examples: https://ackrep-doc.readthedocs.io/en/latest/devdoc/contributing_data.html


# set model name
model_name = ".."


# ---------- create symbolic parameters
pp_symb = [s1, s2, m1, m2, J1, J2, l1, g] = sp.symbols("s1, s2, m1, m2, J1, J2, l1, g", real=True)


# ---------- create symbolic parameter functions
# parameter values can be constant/fixed values OR set in relation to other parameters (for example: a = 2*b)  
s1_sf = 0.25
s2_sf = 0.25
m1_sf = 1
m2_sf = 1
J1_sf = 1/150
J2_sf = 1/75
l1_sf = 0.5
g_sf = 9.81


# list of symbolic parameter functions
# tailing "_sf" stands for "symbolic parameter function"
pp_sf = [s1_sf, s2_sf, m1_sf, m2_sf, J1_sf, J2_sf, l1_sf, g_sf]


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
        "center of gravity distance of first bar",
        "center of gravity distance of second bar",
        "mass of first bar",
        "mass of second bar",
        "moment of inertia of first bar",
        "moment of inertia of second bar",
        "length of first bar",
        "acceleration due to gravity"
        ] 

# contains all lists of the columns before the "Symbol" Column
# --- Empty list, if there are no columns before the "Symbol" Column
start_columns_list = [col_1]


# Define Entries of the columns after the Value-Column
# --- Entries need to be latex code
col_4 = [
        "m",
        "m",
        "kg",
        "kg",
        r"$kg \cdot m^2$",
        r"$kg \cdot m^2$",
        "m",
        r"$\frac{m}{s^2}$"
        ]

# contains all lists of columns after the FIX ENTRIES
# --- Empty list, if there are no columns after the "Value" column
end_columns_list = [col_4]