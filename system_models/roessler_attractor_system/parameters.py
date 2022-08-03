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


model_name = "Roessler_Atractor_1979_1"

# CREATE SYMBOLIC PARAMETERS
pp_symb = [a, b, c] = sp.symbols("a, b, c", real=True)


# SYMBOLIC PARAMETER FUNCTIONS
a_sf = 0.38
b_sf = 0.3
c_sf = 4.84

# List of symbolic parameter functions
pp_sf = [a_sf, b_sf, c_sf]


# List for Substitution
pp_subs_list = []


# OPTONAL: Dictionary which defines how certain variables shall be written
# in the tabular - key: Symbolic Variable, Value: LaTeX Representation/Code
# useful for example for complex variables: {Z: r"\underline{Z}"}
latex_names = {}

# ---------- CREATE BEGIN OF LATEX TABULAR

# Define tabular Header
tabular_header = ["Symbol", "Value"]

# Define column text alignments
col_alignment = ["center", "left"]

col_1 = []

# contains all lists of the columns before the "Symbol" Column
start_columns_list = []

# contains all lists of columns after the FIX ENTRIES
end_columns_list = []
