
import sympy as sp
from sympy import sin, cos, pi
import symbtools as st
import importlib
import sys, os
#from ipydex import IPS, activate_ips_on_exception 

import symbtools.modeltools as mt
 

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
        self.u_dim = 3

        # Set "sys_dim" to constant value, if system dimension is constant 
        self.sys_dim = 8


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
        T = 10
        f1 = 1*sp.sin(2*sp.pi*self.t_symb/T)
        u_num_func = st.expr_to_func(self.t_symb, f1)
        # ---------start of edit section--------------------------------------
        def uu_rhs(t, xx_nv):
            """
            sequence of numerical input values

            :param t:(scalar or vector) time
            :param xx_nv:(vector or array of vectors) numeric state vector
            :return:(list) numeric inputs 
            """ 
            
            u = u_num_func(t)
            return [u]
        # ---------end of edit section----------------------------------------

        return uu_rhs


    # ----------- SYMBOLIC RHS FUNCTION ---------- # 

    def get_rhs_symbolic(self):
        """
        define symbolic model
        return: object of class SymbolicModel from symbtools
        """
        _mod = getattr(self, "_mod", None)
        if _mod is not None:
            return _mod

        x1, x2, x3, xdot1, xdot2, xdot3,lambda_1, lambda_2= sp.symbols('x1, x2, x3, xdot1, xdot2, xdot3, lambda_1, lambda_2') 
        xddot1, xddot2, xddot3 = sp.symbols('xddot1, xddot2, xddot3')

        #s1, s2, s3, m1, m2, m3, J1, J2, J3, l1, l2, l3, l4, g = self.pp_symb  
        params = sp.symbols('s1, s2, s3, m1, m2, m3, J1, J2, J3, l1, l2, l3, l4, g')
        s1, s2, s3, m1, m2, m3, J1, J2, J3, l1, l2, l3, l4, g = params

        u1 = self.uu_symb[0]   # inputs

        mod = mt.SymbolicModel()
        mod.constraints = sp.Matrix([[l1*cos(x3) + l2*cos(x1 + x3) - l3*cos(x2) - l4], [l1*sin(x3) + l2*sin(x1 + x3) - l3*sin(x2)]])

        eqns1 = [J2*xddot1 + J2*xddot3 + g*m2*s2*cos(x1 + x3) + l1*m2*xddot3*s2*cos(x1) + l1*m2*xdot3**2*s2*sin(x1) 
                + l2*lambda_1*sin(x1 + x3) - l2*lambda_2*cos(x1 + x3) + m2*xddot1*s2**2 + m2*xddot3*s2**2]
        eqns2 = [J3*xddot2 + g*m3*s3*cos(x2) - l3*lambda_1*sin(x2) + l3*lambda_2*cos(x2) + m3*xddot2*s3**2]
        eqns3 = [J1*xddot3 + J2*xddot1 + J2*xddot3 + g*l1*m2*cos(x3) + g*m1*s1*cos(x3) + g*m2*s2*cos(x1 + x3) 
                + l1**2*m2*xddot3 + l1*lambda_1*sin(x3) - l1*lambda_2*cos(x3) + l1*m2*xddot1*s2*cos(x1) 
                - l1*m2*xdot1**2*s2*sin(x1) - 2*l1*m2*xdot1*xdot3*s2*sin(x1) + 2*l1*m2*xddot3*s2*cos(x1) 
                + l2*lambda_1*sin(x1 + x3) - l2*lambda_2*cos(x1 + x3) + m1*xddot3*s1**2 + m2*xddot1*s2**2 
                + m2*xddot3*s2**2 - u1]
        mod.eqns = sp.Matrix([eqns1, eqns2, eqns3])
        mod.llmd = sp.Matrix([[lambda_1], [lambda_2]])
        mod.tt = sp.Matrix([[x1], [x2], [x3]])
        mod.ttd = sp.Matrix([[xdot1], [xdot2], [xdot3]])
        mod.ttdd = sp.Matrix([[xddot1], [xddot2], [xddot3]])
        mod.tau = sp.Matrix([[u1]])
        
        # prevent unneccessary recalculation of the model
        self._mod = mod
        return mod
    
    def get_rhs_func(self):
        msg = "This DAE model has no rhs func like ODE models."
        raise NotImplementedError(msg)
        
    def get_dae_model_func(self):
        """
        generate dae system out of the symbolic model
        return: function
        """
        parameter_values = list(self.pp_str_dict.items())
        mod = self.get_rhs_symbolic()
        dae = mod.calc_dae_eq(parameter_values)

        dae.generate_eqns_funcs()
        
        dae_mod_func = dae.model_func
        
        return dae_mod_func

    