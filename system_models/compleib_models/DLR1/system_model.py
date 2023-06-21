# This file was autogenerated from the template: system_model.py.template (2022-10-10 15:57:36).

import sympy as sp
import numpy as np
import symbtools as st
import importlib
import sys, os
#from ipydex import IPS, activate_ips_on_exception  

from ackrep_core.system_model_management import GenericModel, import_parameters

# Import parameter_file
params = import_parameters()


#link to documentation with examples: https://ackrep-doc.readthedocs.io/en/latest/devdoc/contributing_data.html


class Model(GenericModel): 

    def initialize(self):
        """
        this function is called by the constructor of GenericModel

        :return: None
        """

        # Define number of inputs -- MODEL DEPENDENT
        self.u_dim = 2

        # Set "sys_dim" to constant value, if system dimension is constant 
        self.sys_dim = 10

        # check existence of params file
        self.has_params = True
        self.params = params
        

    # ----------- SET DEFAULT INPUT FUNCTION ---------- # 
    def uu_default_func(self):
        """
        define input function
    
        :return:(function with 2 args - t, xx_nv) default input function 
        """ 
        
        def uu_rhs(t, xx_nv):
            """
            sequence of numerical input values

            :param t:(scalar or vector) time
            :param xx_nv:(vector or array of vectors) numeric state vector
            :return:(list) numeric inputs 
            """ 
            u = np.zeros(self.u_dim)
            return u

        return uu_rhs


    # ----------- SYMBOLIC RHS FUNCTION ---------- # 

    def get_rhs_symbolic(self):
        """
        define symbolic rhs function

        :return: matrix of symbolic rhs-functions
        """
        if self.dxx_dt_symb is not None:
            return self.dxx_dt_symb

        x = self.xx_symb  
        A, B, B1, C1, C, D11, D12, D21 = self.pp_symb   # parameters
        w = np.zeros(2) # noise
        u = self.uu_symb   # inputs

        # define symbolic rhs functions
        self.dxx_dt_symb = np.matmul(A,x) + np.matmul(B1,w) + np.matmul(B,u)
        



        return self.dxx_dt_symb
    