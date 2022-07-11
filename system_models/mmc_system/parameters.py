
import sys
import os
import numpy as np
import sympy as sp

import tabulate as tab


# Trailing "_nv" stands for "numerical value"

model_name = "MMC"

# --------- CREATE SYMBOLIC PARAMETERS
pp_symb = [vdc, vg, omega, Lz, Mz, R, L] \
        = sp.symbols('v_DC, v_g, omega, L_z, M_z, R, L', real=True)

# -------- CREATE AUXILIARY SYMBOLIC PARAMETERS 
# (parameters, which shall not numerical represented in the parameter tabular)

# --------- SYMBOLIC PARAMETER FUNCTIONS
# ------------ parameter values can be constant/fixed values OR 
# ------------ set in relation to other parameters (for example: a = 2*b)
# ------------ useful for a clean looking parameter table in the Documentation     

# Due to performance of the simulation the parameters Lz, Mz and L are choosen to be scaled with 1/10

vdc_sf = 300
vg_sf = 235
omega_sf = 2*sp.pi*5
Lz_sf = 1.5/10
Mz_sf = 0.94/10
R_sf = 26
L_sf = 3/10
# List of symbolic parameter functions
pp_sf = [vdc_sf, vg_sf, omega_sf, Lz_sf, Mz_sf, R_sf, L_sf]

# Set numerical values of auxiliary parameters

# List for Substitution 
# -- Entries are tuples like: (independent symbolic parameter, numerical value)
pp_subs_list = []

# OPTONAL: Dictionary which defines how certain variables shall be written
# in the tabular - key: Symbolic Variable, Value: LaTeX Representation/Code
# useful for example for complex variables: {Z: r"\underline{Z}"}
latex_names = {}

# ---------- CREATE BEGIN OF LATEX TABULAR
# Define tabular Header 

# DON'T CHANGE FOLLOWING ENTRIES: "Symbol", "Value"
tabular_header = ["Parameter Name", "Symbol", "Value", "Unit"]

# Define column text alignments
col_alignment = ["left", "center", "left", "center"]

# Define Entries of all columns before the Symbol-Column
# --- Entries need to be latex code

col_1 = ["DC voltage", 
         "grid voltage",
         "angular speed",
         "arm inductance",
         "mutual inductance",
         "load resistance",
         "load inductance"
         ] 
# contains all lists of the columns before the "Symbol" Column
# --- Empty list, if there are no columns before the "Symbol" Column
start_columns_list = [col_1]

# Define Entries of the columns after the Value-Column
# --- Entries need to be latex code
col_4 = ["V", 
         "V",
         "Hz",
         "mH",
         "mH",
         r"$\Omega$",
         "mH"
         ]
# contains all lists of columns after the FIX ENTRIES
# --- Empty list, if there are no columns after the "Value" column
end_columns_list = [col_4]
