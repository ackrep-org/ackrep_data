import sys
import os
import numpy as np
import sympy as sp

import tabulate as tab


# link to documentation with examples: https://ackrep-doc.readthedocs.io/en/latest/devdoc/contributing_data.html


# set model name
model_name = "Ball in tube"


# ---------- create symbolic parameters
pp_symb = [A_B, A_SP, m, g, T_M, k_M, k_V, k_L, n_0] = sp.symbols("A_B, A_SP, m, g, T_M, k_M, k_V, k_L, n_0", real=True)


# ---------- create symbolic parameter functions
# parameter values can be constant/fixed values OR set in relation to other parameters (for example: a = 2*b)
A_B_sf = 2.8274e-3
A_SP_sf = 0.4299e-3
m_sf = 2.8e-3
g_sf = 9.81
T_M_sf = 369e-3
k_M_sf = 0.273
k_V_sf = 12e-5  # 0.0001
k_L_sf = 2.823e-4
n_0_sf = 456

# list of symbolic parameter functions
# trailing "_sf" stands for "symbolic parameter function"
pp_sf = [A_B_sf, A_SP_sf, m_sf, g_sf, T_M_sf, k_M_sf, k_V_sf, k_L_sf, n_0_sf]


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
    "ball cross-sectional area",
    "air gap cross-sectional area",
    "mass of the ball",
    "acceleration due to gravitation",
    "time constant",
    "amplification",
    "proportional factor",
    "parameter",
    "basic rotation speed",
]

# contains all lists of the columns before the "Symbol" Column
# --- Empty list, if there are no columns before the "Symbol" Column
start_columns_list = [col_1]


# Define Entries of the columns after the Value-Column
# --- Entries need to be latex code
col_4 = [
    r"$m^2$",
    r"$m^2$",
    "kg",
    r"$\frac{m}{s^2}$",
    "s",
    r"$s^{-1}$",
    r"$m^3$",
    r"$\frac{kg}{m}$",
    r"$\frac{U}{min}$",
]

# contains all lists of columns after the FIX ENTRIES
# --- Empty list, if there are no columns after the "Value" column
end_columns_list = [col_4]
