
import sys
import os
import numpy as np
import sympy as sp

import tabulate as tab


#link to documentation with examples: https://ackrep-doc.readthedocs.io/en/latest/devdoc/contributing_data.html


# set model name
model_name = "overhead crane"


# ---------- create symbolic parameters
pp_symb = [s2, m1, m2, m3, J2, l0, l1, l2, g] = sp.symbols("s2, m1, m2, m3, J2, l0, l1, l2, g", real=True)


# ---------- create symbolic parameter functions
# parameter values can be constant/fixed values OR set in relation to other parameters (for example: a = 2*b)  
s2_sf=0.15 
m1_sf=0.45 
m2_sf=0.557 
m3_sf=0.45
J2_sf=0.000221
l0_sf=0.5
l1_sf=0.4
l2_sf=0.3
g_sf=9.81

# list of symbolic parameter functions
# tailing "_sf" stands for "symbolic parameter function"
pp_sf = [s2_sf, m1_sf, m2_sf, m3_sf, J2_sf, l0_sf, l1_sf, l2_sf, g_sf]


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
col_1 = ["center of gravity distance of the load",
         "mass of trolley 1",
         "mass of load",
         "mass of trolley 2",
         "moment of inertia of the load",
         "initial distance between the trolleys",
         "length of rope 1",
         "length of rope 2",
         "acceleration due to gravity"] 

# contains all lists of the columns before the "Symbol" Column
# --- Empty list, if there are no columns before the "Symbol" Column
start_columns_list = [col_1]


# Define Entries of the columns after the Value-Column
# --- Entries need to be latex code
col_4 = ["m",
    "kg",
    "kg",
    "kg",
    r"$kg \cdot m^2$",
    "m",
    "m",
    "m",
    r"$\frac{m}{s^2}$"]

# contains all lists of columns after the FIX ENTRIES
# --- Empty list, if there are no columns after the "Value" column
end_columns_list = [col_4]