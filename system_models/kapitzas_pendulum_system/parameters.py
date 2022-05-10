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


model_name = "Kapitzas_Pendulum"

# --------- CREATE SYMBOLIC PARAMETERS
pp_symb = [l, g, a, omega, gamma] = sp.symbols('l, g, a, omega, gamma', real = True)

# -------- CREATE AUXILIARY SYMBOLIC PARAMETERS 
# (parameters, which shall not be numerical represented in the parameter tabular)
omega_0 = sp.Symbol('omega_0')

# --------- SYMBOLIC PARAMETER FUNCTIONS
# ------------ parameter values can be constant/fixed values OR 
# ------------ set in relation to other parameters (for example: a = 2*b)  
l_sf = 30/100
g_sf = 9.81
a_sf = 1/5 * l
omega_sf = 16*omega_0
gamma_sf = 0.1*omega_0

# List of symbolic parameter functions
pp_sf = [l_sf, g_sf, a_sf, omega_sf, gamma_sf]

# Set numerical values of auxiliary parameters
omega_0_nv = np.sqrt(g_sf/l_sf)

# List for Substitution 
# -- Entries are tuples like: (independent symbolic parameter, numerical value)
pp_subs_list = [(l, l_sf), (omega_0, omega_0_nv)]

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

col_1 = ["Pendulum length", 
         "acceleration due to gravitation", 
         "Amplitude of Oscillation", 
         "Frequency of Oscillation", 
         "Dampening Factor"
         ] 
# contains all lists of the columns before the "Symbol" Column
# --- Empty list, if there are no columns before the "Symbol" Column
start_columns_list = [col_1]

# Define Entries of the columns after the Value-Column
# --- Entries need to be latex code
col_4 = ["cm", 
         r"$\frac{m}{s^2}$", 
         "cm", 
         "Hz", 
         "Hz"
         ]
# contains all lists of columns after the FIX ENTRIES
# --- Empty list, if there are no columns after the "Value" column
end_columns_list = [col_4]

