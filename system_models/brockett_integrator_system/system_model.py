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
params = None

class Model(GenericModel): 
    
    def initialize(self):
        """
        this function is called by the constructor of GenericModel

        :return: None
        """
        
        # Define number of inputs -- MODEL DEPENDENT
        self.u_dim = 2

        # Set "sys_dim" to constant value, if system dimension is constant 
        # else set "sys_dim" to x_dim -- MODEL DEPENDENT
        self.sys_dim = 3
        
        # check existance of params file -> if not: System is defined to hasn't 
        # parameters
        self.has_params = True
        self.params = params
        

    # ----------- SET DEFAULT INPUT FUNCTION ---------- # 
    # --------------- Only for non-autonomous Systems
    # --------------- MODEL DEPENDENT
    
    def uu_default_func(self):
        """
        :param t:(scalar or vector) Time
        :param xx_nv: (vector or array of vectors) state vector with 
                                                    numerical values at time t      
        :return:(function with 2 args - t, xx_nv) default input function 
        """         
        def uu_rhs(t, xx_nv):
            u1 = 0
            u2 = 0
            if t > 0:
                u1 = sp.sin(4*sp.pi*t)
                u2 = sp.cos(4*sp.pi*t)  
            return [u1, u2]
        
        return uu_rhs

         
    # ----------- SYMBOLIC RHS FUNCTION ---------- # 
    # --------------- MODEL DEPENDENT  
    
    def get_rhs_symbolic(self):
        """
        :return:(matrix) symbolic rhs-functions
        """
        if self.dxx_dt_symb is not None:
            return self.dxx_dt_symb
        x1, x2, x3 = self.xx_symb
        # u0 = input force     
        u1, u2 = self.uu_symb
        # create symbolic rhs functions
        dx1_dt = u1
        dx2_dt = u2
        dx3_dt = x2*u1 - x1*u2 
        
        # put rhs functions into a vector
        self.dxx_dt_symb = sp.Matrix([dx1_dt, dx2_dt, dx3_dt])
        
        return self.dxx_dt_symb    