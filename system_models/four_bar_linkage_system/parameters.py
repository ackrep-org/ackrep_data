import sys
import os
import numpy as np
import sympy as sp

import tabulate as tab


# link to documentation with examples: https://ackrep-doc.readthedocs.io/en/latest/devdoc/contributing_data.html


# set model name
model_name = "four-bar linkage"


# ---------- create symbolic parameters
pp_symb = [s1, s2, s3, m1, m2, m3, J1, J2, J3, l1, l2, l3, l4, g] = sp.symbols(
    "s1, s2, s3, m1, m2, m3, J1, J2, J3, l1, l2, l3, l4, g", real=True
)


# ---------- create symbolic parameter functions
# parameter values can be constant/fixed values OR set in relation to other parameters (for example: a = 2*b)
s1_sf = 1 / 2
s2_sf = 1 / 2
s3_sf = 1 / 2
m1_sf = 1
m2_sf = 1
m3_sf = 3
J1_sf = 1 / 12
J2_sf = 1 / 12
J3_sf = 1 / 12
l1_sf = 0.8
l2_sf = 1.5
l3_sf = 1.5
l4_sf = 2
g_sf = 9.81

# list of symbolic parameter functions
# tailing "_sf" stands for "symbolic parameter function"
pp_sf = [s1_sf, s2_sf, s3_sf, m1_sf, m2_sf, m3_sf, J1_sf, J2_sf, J3_sf, l1_sf, l2_sf, l3_sf, l4_sf, g_sf]


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
    "center of gravity distance of third bar",
    "mass of first bar",
    "mass of second bar",
    "mass of third bar",
    "moment of inertia",
    "moment of inertia",
    "moment of inertia",
    "length of first bar",
    "length of second bar",
    "length of third bar",
    "length of fourth bar",
    "acceleration due to gravity",
]

# contains all lists of the columns before the "Symbol" Column
# --- Empty list, if there are no columns before the "Symbol" Column
start_columns_list = [col_1]


# Define Entries of the columns after the Value-Column
# --- Entries need to be latex code
col_4 = [
    "m",
    "m",
    "m",
    "kg",
    "kg",
    "kg",
    r"$\frac{kg}{m^2}$",
    r"$\frac{kg}{m^2}$",
    r"$\frac{kg}{m^2}$",
    "m",
    "m",
    "m",
    "m",
    r"$\frac{m}{s^2}$"
]

# contains all lists of columns after the FIX ENTRIES
# --- Empty list, if there are no columns after the "Value" column
end_columns_list = [col_4]
