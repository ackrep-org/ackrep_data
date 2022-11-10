
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
            u1 = 0
            if t < 1: 
                u2 = 0
            elif 1 < t < 2:
                u2 = 2
            elif 2 < t < 3:
                u2 = -2
            else: 
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
        m1, m2, l1, s1, J1, J2, g = self.pp_symb   #parameters
    
        u1, u2 = self.uu_symb   # inputs

        q = sp.Matrix([[x1], [x2]])
        xdot1, xdot2 = sp.symbols('xdot1, xdot2')

        # positions
        S1 = sp.Matrix([sp.sin(q[0]),sp.cos(q[0])])*s1
        S2 = sp.Matrix([sp.sin(q[0]),sp.cos(q[0])])*l1

        # velocities
        Sd1 = st.time_deriv(S1, q)
        Sd2 = st.time_deriv(S2, q)

        # kinetic energy
        T_trans = (m1*Sd1.T*Sd1  +  m2*Sd2.T*Sd2) /2 
        T_rot = (J1*xdot1**2 + J2*(xdot1+xdot2)**2)/2 
        T = T_trans[0] + T_rot 

        #potential energy
        V = (m1*s1+m2*l1)*g*(sp.cos(q[0])-1) 

        external_forces = sp.Matrix([[0, u2]])
        mod = mt.generate_symbolic_model(T, V, q, external_forces)

        mod.calc_state_eq(simplify=False)

        state_eq = mod.state_eq.subs([(xdot1, x3), (xdot2, x4)])

        return state_eq
    