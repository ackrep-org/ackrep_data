
import sys
import os
import numpy as np
import sympy as sp

import tabulate as tab


#link to documentation with examples: 
#


# set model name
model_name = "Two Mass Floating Bodies"


# ---------- create symbolic parameters
pp_symb = [m1, m2, k1, k2, kf, g] = sp.symbols("m1, m2, k1, k2, kf, g", real=True)




# ---------- create symbolic parameter functions
# parameter values can be constant/fixed values OR set in relation to other parameters (for example: a = 2*b)  
m1_sf = 0.05  # mass of the iron ball in kg
m2_sf = 0.04  # mass of the brass ball in kg
k1_sf = 4e-5  # geometry constant
k2_sf = 0.005  # air gap of magnet in m
kf_sf = 10  # Spring constant in N/m
g_sf = 9.8  # acceleration of gravity in m/s^2


# list of symbolic parameter functions
# trailing "_sf" stands for "symbolic parameter function"
pp_sf = [m1_sf, m2_sf, k1_sf, k2_sf, kf_sf, g_sf]


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
col_1 = ["mass of the iron ball",
         "mass of the brass ball",
         "geometry constant", 
         "air gap of magnet",
         "spring constant", 
         "acceleration of gravity"] 

# contains all lists of the columns before the "Symbol" Column
# --- Empty list, if there are no columns before the "Symbol" Column
start_columns_list = [col_1]


# Define Entries of the columns after the Value-Column
# --- Entries need to be latex code
col_4 = ["kg",
         "kg",
         "",
         "m",
         r"$\frac{N}{m}$",
         r"$\frac{m}{s^2}$"
         ]

# contains all lists of columns after the FIX ENTRIES
# --- Empty list, if there are no columns after the "Value" column
end_columns_list = [col_4]