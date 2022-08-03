# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 13:51:06 2021

@author: Jonathan Rockstroh
"""
import sys
import os
import numpy as np
import sympy as sp

import tabulate as tab


model_name = "Stable_PT_n_Element"

# --------- CREATE SYMBOLIC PARAMETERS
pp_symb = [K, T1, T2] = sp.symbols("K, T1, T2", real=True)

# -------- CREATE AUXILIARY SYMBOLIC PARAMETERS
# (parameters, which shall not numerical represented in the parameter tabular)


# --------- SYMBOLIC PARAMETER FUNCTIONS
# ------------ parameter values can be constant/fixed values OR
# ------------ set in relation to other parameters (for example: a = 2*b)
# ------------ useful for a clean looking parameter table in the Documentation
K_sf = 3
T1_sf = 5
T2_sf = 0.5
# List of symbolic parameter functions
pp_sf = [K_sf, T1_sf, T2_sf]

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

col_1 = ["Proportional Factor", "Time Constant 1", "Time Constant 2"]
# contains all lists of the columns before the "Symbol" Column
# --- Empty list, if there are no columns before the "Symbol" Column
start_columns_list = [col_1]

# Define Entries of the columns after the Value-Column
# --- Entries need to be latex code
col_4 = ["", "s", "s"]
# contains all lists of columns after the FIX ENTRIES
# --- Empty list, if there are no columns after the "Value" column
end_columns_list = [col_4]
