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


# Trailing "_nv" stands for "numerical value"


model_name = "Lorenz_Attractor"

# CREATE SYMBOLIC PARAMETERS
pp_symb = [r, b, sigma] = sp.symbols('r, b, sigma', real = True)



# SYMBOLIC PARAMETER FUNCTIONS  
r_sf = 35
b_sf = 2
sigma_sf = 20


# List of symbolic parameter functions
pp_sf = [r_sf, b_sf, sigma_sf]


# List for Substitution 
pp_subs_list = []

# OPTONAL: Dictionary which defines how certain variables shall be written
# in the tabular - key: Symbolic Variable, Value: LaTeX Representation/Code
# useful for example for complex variables: {Z: r"\underline{Z}"}
latex_names = {}




# ---------- CREATE BEGIN OF LATEX TABULAR
# Define tabular Header 

# DON'T CHANGE FOLLOWING ENTRIES: "Symbol", "Value"
tabular_header = ["Parameter Name", "Symbol", "Value"]

# Define column text alignments
col_alignment = ["left", "center", "left"]

# Define Entries of all columns before the Symbol-Column
# --- Entries need to be latex code
col_1 = ["Prandtl Number", 
         "Raileight coeff",
         "Parameter"
         ] 

# contains all lists of the columns before the "Symbol" Column
# --- Empty list, if there are no columns before the "Symbol" Column
start_columns_list = [col_1]


# contains all lists of columns after the FIX ENTRIES
# --- Empty list, if there are no columns after the "Value" column
end_columns_list = []

