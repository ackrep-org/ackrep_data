
import sys
import os
import numpy as np
import sympy as sp

import tabulate as tab


#link to documentation with examples: https://ackrep-doc.readthedocs.io/en/latest/devdoc/contributing_data.html


# set model name
model_name = "Lotka_Volterra"


# ---------- create symbolic parameters
pp_symb = [a, b, c , d] = sp.symbols("a, b, c, d", real=True)


# set numerical values of auxiliary parameters
# tailing "_nv" stands for "numerical value"
a_nv = 1.3
b_nv = 0.9
c_nv = 0.8
d_nv = 1.8

# list of symbolic parameter functions
# tailing "_sf" stands for "symbolic parameter function"
pp_sf = [a_nv, b_nv, c_nv, d_nv]

#OPTIONAL
# range of parameters
a_range = (0, np.inf)
b_range = (0, np.inf)
c_range = (0, np.inf)
d_range = (0, np.inf)

#OPTIONAL
# list of ranges
pp_range_list = [a_range, b_range, c_range, d_range]

#  ---------- list for substitution
# -- entries are tuples like: (independent symbolic parameter, numerical value)
pp_subs_list = []


# OPTONAL: Dictionary which defines how certain variables shall be written
# in the table - key: Symbolic Variable, Value: LaTeX Representation/Code
# useful for example for complex variables: {Z: r"\underline{Z}"}
latex_names = {a: r"\alpha", b: r"\beta", c: r"\gamma", d: r"\delta"}


# ---------- Define LaTeX table

# ---------- CREATE BEGIN OF LATEX TABULAR
# Define tabular Header

# DON'T CHANGE FOLLOWING ENTRIES: "Symbol", "Value"
tabular_header = ["Parameter Name", "Symbol", "Value"]

# Define column text alignments
col_alignment = ["left", "center", "left"]

# Define Entries of all columns before the Symbol-Column
# --- Entries need to be latex code
col_1 = [
    "reproduction rate of prey alone", 
    "mortality rate of prey per predator", 
    "mortality rate of predators",
    "reproduction rate of predators per prey"
    ]

# contains all lists of the columns before the "Symbol" Column
# --- Empty list, if there are no columns before the "Symbol" Column
start_columns_list = [col_1]


# contains all lists of columns after the FIX ENTRIES
# --- Empty list, if there are no columns after the "Value" column
end_columns_list = []