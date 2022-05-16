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
        
        # Define number of inputs
        self.u_dim = 1

        # Set "sys_dim" to constant value, if system dimension is constant 
        self.sys_dim = 2
        
        # check existance of params file -> if not: System is defined to hasn't 
        # parameters
        self.has_params = True
        self.params = params
        
                                


    # ----------- SET DEFAULT INPUT FUNCTION ---------- # 
    # --------------- Only for non-autonomous Systems
    # --------------- MODEL DEPENDENT
    
    def uu_default_func(self):
        """
        :return:(function with 2 args - t, xx_nv) default input function 
        """ 
        a, omega = self.pp_symb[2], self.pp_symb[3]
        u_sp = self.pp_dict[a]*sp.sin(self.pp_dict[omega]*self.t_symb-sp.pi/2)
        du_dtt_sp = u_sp.diff(self.t_symb, 2)
        du_dtt_sp = du_dtt_sp.subs(self.pp_subs_list)
        du_dtt_func = st.expr_to_func(self.t_symb , du_dtt_sp)
        
        def uu_rhs(t, xx_nv):
            du_dtt_nv = du_dtt_func(t)
            return [du_dtt_nv]
        
        return uu_rhs
         
    # ----------- SYMBOLIC RHS FUNCTION ---------- # 
    # --------------- MODEL DEPENDENT  
    
    def get_rhs_symbolic(self):
        """
        :return:(scalar or array) symbolic rhs-functions
        """
        if self.dxx_dt_symb is not None:
            return self.dxx_dt_symb
        x1, x2 = self.xx_symb
        l, g, a, omega, gamma = self.pp_symb 
        # u0 = input force     
        u0 = self.uu_symb[0]
        # create symbolic rhs functions
        dx1_dt = x2
        dx2_dt = -2*gamma*x2 - (g/l + 1/l * u0) *sp.sin(x1)
        
        # put rhs functions into a vector
        self.dxx_dt_symb = [dx1_dt, dx2_dt]
        
        return self.dxx_dt_symb