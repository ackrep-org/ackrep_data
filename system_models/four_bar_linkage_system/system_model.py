
import sympy as sp
from sympy import sin, cos
import symbtools as st
import importlib
import sys, os
#from ipydex import IPS, activate_ips_on_exception  

from ackrep_core.system_model_management import GenericModel, import_parameters
from ackrep_core.core import get_metadata_from_file

# Import parameter_file
yml_path = os.path.join(os.path.dirname(__file__), "metadata.yml")
md = get_metadata_from_file(yml_path)
params = import_parameters(md["key"])


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

        #Set model type for calculation 
        self.state_space_model = False

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
        define symbolic rhs function

        :return: matrix of symbolic rhs-functions
        """
        if self.dxx_dt_symb is not None:
            return self.dxx_dt_symb


        # ---------start of edit section--------------------------------------

        p1, p2, q1, pdot1, pdot2, qdot1, lambda_1, lambda_2 = self.xx_symb   
        s1, s2, s3, m1, m2, m3, J1, J2, J3, l1, l2, l3, l4, kappa1, kappa2, g = self.pp_symb   #parameters
    
        tau1 = self.uu_symb[0]   # inputs

        pddot1, pddot2, qddot1 = sp.symbols('pddot1, pddot2, qddot1')


        constraints = sp.Matrix([[l1*cos(q1) + l2*cos(p1 + q1) - l3*cos(p2) - l4], [l1*sin(q1) + l2*sin(p1 + q1) - l3*sin(p2)]])

        ode1 = [J2*pddot1 + J2*qddot1 + g*m2*s2*cos(p1 + q1) + l1*m2*qddot1*s2*cos(p1) + l1*m2*qdot1**2*s2*sin(p1) 
                + l2*lambda_1*sin(p1 + q1) - l2*lambda_2*cos(p1 + q1) + m2*pddot1*s2**2 + m2*qddot1*s2**2]
        ode2 = [J3*pddot2 + g*m3*s3*cos(p2) - l3*lambda_1*sin(p2) + l3*lambda_2*cos(p2) + m3*pddot2*s3**2]
        ode3 = [J1*qddot1 + J2*pddot1 + J2*qddot1 + g*l1*m2*cos(q1) + g*m1*s1*cos(q1) + g*m2*s2*cos(p1 + q1) 
                + l1**2*m2*qddot1 + l1*lambda_1*sin(q1) - l1*lambda_2*cos(q1) + l1*m2*pddot1*s2*cos(p1) 
                - l1*m2*pdot1**2*s2*sin(p1) - 2*l1*m2*pdot1*qdot1*s2*sin(p1) + 2*l1*m2*qddot1*s2*cos(p1) 
                + l2*lambda_1*sin(p1 + q1) - l2*lambda_2*cos(p1 + q1) + m1*qddot1*s1**2 + m2*pddot1*s2**2 
                + m2*qddot1*s2**2 - tau1]
        odes = sp.Matrix([ode1, ode2, ode3])

        
        # rhs functions matrix
        self.eqns = odes
        self.constraints = constraints

        # ---------end of edit section----------------------------------------


        return [self.eqns, self.constraints]
    