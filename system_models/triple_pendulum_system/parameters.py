
import sys
import os
import numpy as np
import sympy as sp

import tabulate as tab


#link to documentation with examples: https://ackrep-doc.readthedocs.io/en/latest/devdoc/contributing_data.html


# set model name
model_name = "triple pendulum"


# ---------- create symbolic parameters
pp_symb = [m0, m1, m2, m3, J1, J2, J3, l1, l2, l3, a1, a2, a3, g] = sp.symbols("m0, m1, m2, m3, J1, J2, J3, l1, l2, l3, a1, a2, a3, g", real=True)


# ---------- create symbolic parameter functions
# parameter values can be constant/fixed values OR set in relation to other parameters (for example: a = 2*b)  
m0_sf = 3.34
m1_sf = 0.8512
m2_sf = 0.8973
m3_sf = 0.5519
J1_sf = 0.01980194
J2_sf = 0.02105375
J3_sf = 0.01818537
l1_sf = 0.32
l2_sf = 0.419
l3_sf = 0.485
a1_sf = 0.20001517
a2_sf = 0.26890449
a3_sf = 0.21666087
g_sf = 9.81

# list of symbolic parameter functions
# tailing "_sf" stands for "symbolic parameter function"
pp_sf = [m0_sf, m1_sf, m2_sf, m3_sf, J1_sf, J2_sf, J3_sf, l1_sf, l2_sf, l3_sf, a1_sf, a2_sf, a3_sf, g_sf]


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
        "mass of link 1",
        "mass of link 2",
        "mass of link 3",
        "moment of inertia of link 1",
        "moment of inertia of link 2",
        "moment of inertia of link 3",
        "length of link 1",
        "length of link 2",
        "length of link 3",
        "distance from the joint to the center of gravity of link 1",
        "distance from the joint to the center of gravity of link 2",
        "distance from the joint to the center of gravity of link 3",
        "acceleration due to gravity"
        ] 

# contains all lists of the columns before the "Symbol" Column
# --- Empty list, if there are no columns before the "Symbol" Column
start_columns_list = [col_1]


# Define Entries of the columns after the Value-Column
# --- Entries need to be latex code
col_4 = [
        "kg",
        "kg",
        "kg",
        "kg",
        r"$kg \cdot m^2$",
        r"$kg \cdot m^2$",
        r"$kg \cdot m^2$",
        "m",
        "m",
        "m",
        "m",
        "m",
        "m",
        r"$\frac{m}{s^2}$"
        ]

# contains all lists of columns after the FIX ENTRIES
# --- Empty list, if there are no columns after the "Value" column
end_columns_list = [col_4]