import sys
import os
import numpy as np
import sympy as sp

import tabulate as tab


# tailing "_nv" stands for "numerical value"

# SET MODEL NAME
model_name = "PVTOL with 2 forces"


# CREATE SYMBOLIC PARAMETERS
pp_symb = [g, l, m, J] = sp.symbols("g, l, m, J", real=True)


# SYMBOLIC PARAMETER FUNCTIONS
# parameter values can be constant/fixed values OR set in relation to other parameters (for example: a = 2*b)
g_sf = 9.81
l_sf = 0.1
m_sf = 0.25
J_sf = 0.00076

# List of symbolic parameter functions
pp_sf = [g_sf, l_sf, m_sf, J_sf]

pp_subs_list = []


# OPTONAL: Dictionary which defines how certain variables shall be written
# in the tabular - key: Symbolic Variable, Value: LaTeX Representation/Code
# useful for example for complex variables: {Z: r"\underline{Z}"}
latex_names = {}


# ---------- CREATE BEGIN OF LATEX TABULAR

# Define tabular Header
# DON'T CHANGE FOLLOWING ENTRIES: "Symbol", "Value"
tabular_header = ["Parametername", "Symbol", "Value", "Unit"]

# Define column text alignments
col_alignment = ["left", "center", "left", "center"]


# Define Entries of all columns before the Symbol-Column
# --- Entries need to be latex code
col_1 = ["acceleration due to gravity", "distance of forces to mass center", "mass", "moment of inertia"]

# contains all lists of the columns before the "Symbol" Column
# --- Empty list, if there are no columns before the "Symbol" Column
start_columns_list = [col_1]


# Define Entries of the columns after the Value-Column
# --- Entries need to be latex code
col_4 = [r"$\frac{\mathrm{m}}{\mathrm{s}^2}$", "m", "kg", r"$\mathrm{kg} \cdot \mathrm{m}^2$"]

# contains all lists of columns after the FIX ENTRIES
# --- Empty list, if there are no columns after the "Value" column
end_columns_list = [col_4]
