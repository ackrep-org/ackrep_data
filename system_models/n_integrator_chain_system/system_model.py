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
        self.u_dim = 1

        # only needed for n extendable systems -- MODEL DEPENDENT
        self.default_param_sys_dim = 3
     
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
        T = 10
        y1 = 0
        # create symbolic polnomial function
        f1 = sp.sin(2*sp.pi*self.t_symb/T)
        # create symbolic piecewise defined symbolic transition function
        transition = st.piece_wise((0, self.t_symb < 0), (f1, self.t_symb < T),
                                   (-f1, self.t_symb < 2*T), (y1, True))
        # transform symbolic to numeric function 
        transition_func = st.expr_to_func(self.t_symb, transition) 
        
        # Wrapper function to unify function arguments
        def uu_rhs(t, xx_nv):
            """
            :param t:(vector) time
            :param xx_nv:(vector or array of vectors) numeric state vector
            :return:(vector) numeric inputs 
            """
            res = transition_func(t)
            return res
        
        return uu_rhs

         
    # ----------- SYMBOLIC RHS FUNCTION ---------- # 
    # --------------- MODEL DEPENDENT  
    
    def get_rhs_symbolic(self):
        """
        :return:(scalar or array) symbolic rhs-functions
        """
        if self.dxx_dt_symb is not None:
            return self.dxx_dt_symb
        
        dxx_dt_symb = self.xx_symb*1
        dxx_dt_symb[:-1] = self.xx_symb[1:]
        dxx_dt_symb[-1] = self.uu_symb[0]
        
        self.dxx_dt_symb = dxx_dt_symb
        
        return self.dxx_dt_symb


