
import sympy as sp
import symbtools as st
import importlib
import sys, os
#from ipydex import IPS, activate_ips_on_exception  

from ackrep_core.system_model_management import GenericModel, import_parameters

# Import parameter_file
params = None


#link to documentation with examples: https://ackrep-doc.readthedocs.io/en/latest/devdoc/contributing_data.html


class Model(GenericModel): 

    def initialize(self):
        """
        this function is called by the constructor of GenericModel

        :return: None
        """
        
        # ---------start of edit section--------------------------------------
        # Define number of inputs -- MODEL DEPENDENT
        self.u_dim = 0

        # Set "sys_dim" to constant value, if system dimension is constant 
        self.sys_dim = 2

        # ---------end of edit section----------------------------------------

        # check existence of params file
        self.has_params = False
        self.params = params
        

    # ----------- SYMBOLIC RHS FUNCTION ---------- # 

    def get_rhs_symbolic(self):
        """
        define symbolic rhs function

        :return: matrix of symbolic rhs-functions
        """
        if self.dxx_dt_symb is not None:
            return self.dxx_dt_symb


        # ---------start of edit section--------------------------------------
        x1, x2 = self.xx_symb   #state components


        # define symbolic rhs functions
        dx1_dt = 2*x2
        dx2_dt = (x1 + x2)*(-(x1 -x2)**2 + 1)

        # rhs functions matrix
        self.dxx_dt_symb = sp.Matrix([dx1_dt, dx2_dt])
        # ---------end of edit section----------------------------------------


        return self.dxx_dt_symb
    