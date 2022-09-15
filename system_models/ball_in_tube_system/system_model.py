
import sympy as sp
import symbtools as st
import importlib
import sys, os
#from ipydex import IPS, activate_ips_on_exception  
from random import randrange

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
        
        # ---------start of edit section--------------------------------------
        # Define number of inputs -- MODEL DEPENDENT
        self.u_dim = 1

        # Set "sys_dim" to constant value, if system dimension is constant 
        self.sys_dim = 3

        # ---------end of edit section----------------------------------------

        # check existence of params file
        self.has_params = True
        self.params = params
        

    # ----------- SET DEFAULT INPUT FUNCTION ---------- # 
    # --------------- Only for non-autonomous Systems
    def uu_default_func(self):
        """
        define input function
    
        :return:(function with 2 args - t, xx_nv) default input function 
        """ 
        # ---------start of edit section--------------------------------------
        def uu_rhs(t, xx_nv):
            """
            sequence of numerical input values

            :param t:(scalar or vector) time
            :param xx_nv:(vector or array of vectors) numeric state vector
            :return:(list) numeric inputs 
            """ 
            u = 120
            return [u]
        # ---------end of edit section----------------------------------------

        return uu_rhs


    # ----------- SYMBOLIC RHS FUNCTION ---------- # 

    def get_rhs_symbolic(self):
        """
        define symbolic rhs function

        :return: matrix of symbolic rhs-functions
        """
        if self.dxx_dt_symb is not None:
            return self.dxx_dt_symb


        # ---------start of edit section--------------------------------------
        x1, x2, x3 = self.xx_symb   #state components
        A_B, A_SP, m, g, T_M, k_M, k_V, k_L, n_0 = self.pp_symb   #parameters
    
        u1 = self.uu_symb[0]   # inputs

        # define symbolic rhs functions
        dx1_dt = -60/T_M * x1 + k_M/T_M * u1 *60**2
        dx2_dt = x3
        dx3_dt = k_L/m * ((k_V*(x1 + n_0)/60 - A_B*x3)/A_SP)**2 - g

        # rhs functions matrix
        self.dxx_dt_symb = sp.Matrix([dx1_dt, dx2_dt, dx3_dt])
        # ---------end of edit section----------------------------------------


        return self.dxx_dt_symb
    