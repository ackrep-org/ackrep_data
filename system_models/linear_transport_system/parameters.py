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


# tailing "_nv" stands for "numerical value"


model_name = "Transport_System"

# CREATE SYMBOLIC PARAMETERS
pp_symb = [v, l, T] = sp.symbols('v, l, T', real = True)



# SYMBOLIC PARAMETER FUNCTIONS  
v_sf = 4
l_sf = 5
T_sf = 5


# List of symbolic parameter functions
pp_sf = [v_sf, l_sf, T_sf]


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
col_1 = ["transport velocity", 
         "simulated space",
         "simulated time"
         ] 

# contains all lists of the columns before the "Symbol" Column
# --- Empty list, if there are no columns before the "Symbol" Column
start_columns_list = [col_1]


# contains all lists of columns after the FIX ENTRIES
# --- Empty list, if there are no columns after the "Value" column
end_columns_list = []

