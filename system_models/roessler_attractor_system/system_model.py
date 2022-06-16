# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 13:33:34 2021

@author: Jonathan Rockstroh
"""

import sympy as sp
import symbtools as st
import importlib
import sys, os
from ipydex import IPS, activate_ips_on_exception  # for debugging only

from ackrep_core.system_model_management import GenericModel, import_parameters
from ackrep_core.core import get_metadata_from_file

# Import parameter_file
yml_path = os.path.join(os.path.dirname(__file__), "metadata.yml")
md = get_metadata_from_file(yml_path)
params = import_parameters(md["key"])



class Model(GenericModel): 
    
    def initialize(self):
        """
        this function is called by the constructor of GenericModel

        :return: None
        """
        
        # Define number of inputs -- MODEL DEPENDENT
        self.u_dim = 0
        # Set "sys_dim" to constant value, if system dimension is constant 
        # else set "sys_dim" to x_dim -- MODEL DEPENDENT
        self.sys_dim = 3
       
        # check existance of params file -> if not: System is defined to hasn't 
        # parameters
        self.has_params = True
        self.params = params


    # ----------- SYMBOLIC RHS FUNCTION ---------- # 
    # --------------- MODEL DEPENDENT  
    
    def get_rhs_symbolic(self):
        """
        :return:(matrix) symbolic rhs-functions
        """
        if self.dxx_dt_symb is not None:
            return self.dxx_dt_symb
        x, y, z = self.xx_symb
        a, b, c = self.pp_symb 

        # create symbolic rhs functions
        dx_dt = - y - z
        dy_dt = x + a*y
        dz_dt = b*x - c*z + x*z
        
        # put rhs functions into a vector
        self.dxx_dt_symb = sp.Matrix([dx_dt, dy_dt, dz_dt])
        
        return self.dxx_dt_symb               
