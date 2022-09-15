
import sympy as sp
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
        
        # ---------start of edit section--------------------------------------
        # Define number of inputs -- MODEL DEPENDENT
        self.u_dim = 2

        # Set "sys_dim" to constant value, if system dimension is constant 
        self.sys_dim = 2

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
            if t<2.5:
                u_func_nv = 2
            else: 
                u_func_nv = 6
                
            if u_func_nv==0:
                e_func_nv = 0
            else: 
                e_func_nv = 5e-2
            return [u_func_nv, e_func_nv]
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
        x1, x2 = self.xx_symb   #state components
        c_phi, J, La, Ra = self.pp_symb   #parameters
    
        u, e = self.uu_symb   # inputs

        # define symbolic rhs functions
        dx1_dt = c_phi/J*x2 - e/J
        dx2_dt =-Ra/La*x2 - c_phi/La*x1 + u/La

        # rhs functions matrix
        self.dxx_dt_symb = sp.Matrix([dx1_dt, dx2_dt])
        # ---------end of edit section----------------------------------------


        return self.dxx_dt_symb
    