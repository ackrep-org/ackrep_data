
import sympy as sp
import symbtools as st
from symbtools import modeltools as mt
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
        self.sys_dim = 4

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
            if 1 < t < 2: 
                u1 = 0.5
            else: 
                u1 = 0
            u2 = 0

            return [u1, u2]
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

        x1, x2, x3, x4 = self.xx_symb   #state components
        x = sp.Matrix([[x1], [x2]])
        xdot1, xdot2 = sp.symbols('xdot1, xdot2')

        s1, s2, m1, m2, J1, J2, l1 = self.pp_symb   #parameters
    
        u1, u2 = self.uu_symb   # inputs

        ex = sp.Matrix([1,0])

        S1 = mt.Rz(x[0])*ex*s1
        G1 = mt.Rz(x[0])*ex*l1

        S2 = G1 + mt.Rz(x[0]+x[1])*ex*s2

        Sd1 = st.time_deriv(S1, x)
        Sd2 = st.time_deriv(S2, x)
        
        T_trans = (m1*Sd1.T*Sd1  +  m2*Sd2.T*Sd2) /2 
        T_rot = (J1*xdot1**2 + J2*(xdot1+xdot2)**2)/2

        T = T_trans[0] + T_rot #kinetic energy
        V = 0 #potential energy

        external_forces = sp.Matrix([[u1, 0]])
        mod = mt.generate_symbolic_model(T, V, x, external_forces)

        mod.calc_state_eq(simplify=False)

        state_eq = mod.state_eq.subs([(xdot1, x3), (xdot2, x4)])

        return state_eq
    