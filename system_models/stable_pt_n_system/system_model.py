# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 13:33:34 2021

@author: Jonathan Rockstroh
"""

import sympy as sp
import symbtools as st
import importlib
import sys, os
from itertools import combinations as comb
import numpy as np
from ipydex import IPS, activate_ips_on_exception  # for debugging only

from ackrep_core.system_model_management import GenericModel, import_parameters
from ackrep_core.core import get_metadata_from_file

# Import parameter_file
yml_path = os.path.join(os.path.dirname(__file__), "metadata.yml")
md = get_metadata_from_file(yml_path)
params = import_parameters(md["key"])

    
class Model(GenericModel): 
    ## NOTE:
        # x_dim usw vllt als keywordargs definieren - Vermeidung von effektlosen, optionelen parametern  
    has_params = True    
    def __init__(self, x_dim=None, u_func=None, pp=None):
        """
        :param x_dim:(int, positive) dimension of the state vector 
                                - has no effect for non-extendible systems
        :param u_func:(callable) input function, args: time, state vector
                        return: list of numerical input values 
                        - has no effect for autonomous systems    
        :param pp:(vector or dict-type with floats>0) parameter values
        :return:
        """
        
  
        # Define number of inputs -- MODEL DEPENDENT
        self.u_dim = 1
        # Set "sys_dim" to constant value, if system dimension is constant 
        # else set "sys_dim" to x_dim -- MODEL DEPENDENT
        self.sys_dim = x_dim
        # Adjust sys_dim to dimension fitting to default parameters
        # only needed for n extendable systems -- MODEL DEPENDENT
        self.default_param_sys_dim = 2
        
        # check existance of params file -> if not: System is defined to hasn't 
        # parameters
        self.has_params = True
        self.params = params

        # Initialize     
        super().__init__(x_dim=None, u_func=None, pp=None)
        

    # ----------- _CREATE_N_DIM_SYMB_PARAMETERS ---------- #
    # ------------- MODEL DEPENDENT, IF: Model has parameters
    # -------------                      AND is n extendible
    # ------------- If Model isn't n extendible set return to None

    def _create_n_dim_symb_parameters(self):
        """Creates the symbolic parameter list for n extendible systems
        
        :return: 
            pp_symb: list of sympy.symbol type entries
                contains the symbolic parameters
            None: 
                if system has a constant dimension
        """
        # Create T_i are the time constants and K is the proportional factor
        pp_symb = [sp.Symbol('T' + str(i)) for i in range(0, self.n)]
        pp_symb = [sp.Symbol('K')] + pp_symb
        return pp_symb

                                

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
            u = 10
            c = t > 0
            try: 
                uu = list(c*u)
            except TypeError:
                uu = [c*u]
            uu = [np.sin(0.387298334621*t)]
            return uu
        
        return uu_rhs

         
    # ----------- SYMBOLIC RHS FUNCTION ---------- # 
    # --------------- MODEL DEPENDENT  
    
    def get_rhs_symbolic(self):
        """
        :return:(scalar or array) symbolic rhs-functions
        """
        if self.dxx_dt_symb is not None:
            return self.dxx_dt_symb
        xx_symb = self.xx_symb  
        u_symb= self.uu_symb[0]
        K_symb = self.pp_symb[0]
        TT_symb = self.pp_symb[1:]
        # create symbolic rhs functions
        # define entry/derivation 1 to n-1
        dxx_dt = [entry for entry in xx_symb]
        dxx_dt = dxx_dt[1:]
        # define entry/derivation n
        sum_vec = []
        # write all summands of the expanded inverse laplace into a vector
        for k in range(self.n + 1):
            next_elem = self._create_factor(TT_symb, k)        
            sum_vec = sum_vec + [next_elem]
        # calculate expanded inverse laplace
        
        inv_laplace = 0
        for i in range(len(sum_vec) - 1):
            inv_laplace = inv_laplace + sum_vec[i]*xx_symb[i]
        
        
        dxn_dt = (K_symb*u_symb - inv_laplace) * 1/sum_vec[-1]
        dxx_dt = dxx_dt + [dxn_dt]
        # put rhs functions into a vector
        self.dxx_dt_symb = dxx_dt
        
        return self.dxx_dt_symb
    
    

    # ----------- VALIDATE PARAMETER VALUES ---------- #
    # -------------- MODEL DEPENDENT 
    
    def _validate_p_values(self, p_value_list):
        """ raises exception if values in list aren't valid 
        :param p_value_list:(float) list of parameter values
        """
        # Check for convertability to float
        try: [float(i) for i in p_value_list]
        except ValueError:
                raise Exception(":param pp: Values are not valid. \
                                (aren't convertible to float)")
                                 
        #--- MODEL DEPENDENT 
        # possible to include, not necessary                         
        # Check if values are in required range                          
        assert not any(flag <= 0 for flag in p_value_list), \
                        ":param pp: does have values <= 0"