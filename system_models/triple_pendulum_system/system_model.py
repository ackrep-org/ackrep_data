
import sympy as sp
import symbtools as st
import importlib
import sys, os
import pickle
from symbtools import modeltools as mt
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
        self.u_dim = 4

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
        
        # ---------start of edit section--------------------------------------
        def uu_rhs(t, xx_nv):
            """
            sequence of numerical input values

            :param t:(scalar or vector) time
            :param xx_nv:(vector or array of vectors) numeric state vector
            :return:(list) numeric inputs 
            """ 
            u1 = 0
            u2 = 0
            u3 = 0
            u4 = 1
            
            return [u1, u2, u3, u4]
        # ---------end of edit section----------------------------------------

        return uu_rhs


    # ----------- SYMBOLIC RHS FUNCTION ---------- # 

    def get_rhs_symbolic(self):
        """
        define symbolic equations of the model

        :return: model 
        """

        x1, x2, x3, x4, x5, x6, x7, x8 = self.xx_symb   #state components
        m0, m1, m2, m3, J1, J2, J3, l1, l2, l3, a1, a2, a3, g = self.pp_symb   #parameters
    
        u1, u2, u3, u4 = self.uu_symb   # inputs

        ttheta = sp.Matrix([[x1], [x2], [x3], [x4]])
        xdot1, xdot2, xdot3, xdot4 = sp.symbols('xdot1, xdot2, xdot3, xdot4')

        ex = sp.Matrix([1, 0])
        ey = sp.Matrix([0, 1])

        Rz = mt.Rz

        S0 = G0 = ex*x4

        G1 = G0 + Rz(x1)*ey*l1
        S1 = G0 + Rz(x1)*ey*a1

        G2 = G1 + Rz(x2)*ey*l2
        S2 = G1 + Rz(x2)*ey*a2 

        G3 = G2 + Rz(x3)*ey*l3 
        S3 = G2 + Rz(x3)*ey*a2

        # Time derivatives of coordinates of the centers of masses
        S0dt, S1dt, S2dt, S3dt = st.col_split(st.time_deriv(st.col_stack(S0, S1, S2, S3), ttheta)) 

        # kinetic energy of the cart
        T0 = 0.5 * m0 * xdot4**2
        # kinetic energy of pendulum1
        T1 = 0.5 * m1 * (S1dt.T * S1dt)[0] + 0.5 * J1 * xdot1**2
        # kinetic energy of pendulum2
        T2 = 0.5 * m2 * (S2dt.T * S2dt)[0] + 0.5 * J2 * xdot2**2
        # kinetic energy of pendulum3
        T3 = 0.5 * m3 * (S3dt.T * S3dt)[0] + 0.5 * J3 * xdot3**2

        # total kinetic energy
        T = T0 + T1 + T2 + T3

        # total potential energy
        V = g * (m1 * S1[1] + m2 * S2[1] + m3 * S3[1])

        external_forces = [0, 0, 0, u4]

        dir_of_this_file = os.path.dirname(os.path.abspath(sys.modules.get(__name__).__file__))
        fpath = os.path.join(dir_of_this_file, "pendulum.pcl")

        if not os.path.isfile(fpath):
            # Calculate the model based on lagrange equation (about 30s)
            mod = mt.generate_symbolic_model(T, V, ttheta, external_forces)
            
            # perform patial linearization such that system input is acceleration and not force (about 9min)
            mod.calc_coll_part_lin_state_eq()
            
            # write the model to disk to save time in the next run of the notebook
            with open("pendulum.pcl", "wb") as pfile:
                pickle.dump(mod, pfile)
        else:
            with open("pendulum.pcl", "rb") as pfile:
                mod = pickle.load(pfile)

        mod.eqns = mod.eqns.subs([(xdot1, x5), (xdot2, x6), (xdot3, x7), (xdot4, x8)])

        mod.ff = mod.ff.subs([(xdot1, x5), (xdot2, x6), (xdot3, x7), (xdot4, x8)])
        mod.xx = mod.xx.subs([(xdot1, x5), (xdot2, x6), (xdot3, x7), (xdot4, x8)])
        mod.gg = mod.gg.subs([(xdot1, x5), (xdot2, x6), (xdot3, x7), (xdot4, x8)])

        return mod         

    def get_rhs_odeint_fnc(self):
        """
        Creates an executable function of the model for the odeint solver

        :return: rhs function
        """

        mod = self.get_rhs_symbolic()

        #parameter_values = list(self.pp_str_dict.items())
        self._create_subs_list()
        parameter_values = self.pp_subs_list

        ff = mod.ff.subs(parameter_values)
        xx = mod.xx.subs(parameter_values)
        gg = mod.gg.subs(parameter_values)
        simmod = st.SimulationModel(ff, gg, xx)
        rhs = simmod.create_simfunction()

        return rhs



    
    