import sympy as sp
import symbtools as st
import importlib
import sys, os

# from ipydex import IPS, activate_ips_on_exception

from ackrep_core.system_model_management import GenericModel, import_parameters

# Import parameter_file
params = import_parameters()


# link to documentation with examples:
#


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

        # T = 10
        # f1 = 0.2 * sp.sin(self.t_symb) + 0.74
        # u_num_func = st.expr_to_func(self.t_symb, f1)

        # ---------start of edit section--------------------------------------
        def uu_rhs(t, xx_nv):
            """
            sequence of numerical input values

            :param t:(scalar or vector) time
            :param xx_nv:(vector or array of vectors) numeric state vector
            :return:(list) numeric inputs
            """
            #u = u_num_func(t)
            #u = 0.745

            if t < 0.25:
                u = 8.877
            else: 
                u = 0.7

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
        x1, x2, x3, x4 = self.xx_symb  # state components
        m1, m2, k1, k2, kf, g = self.pp_symb  # parameters

        u1 = self.uu_symb[0]  # inputs

        # define symbolic rhs functions
        dx1_dt = x3
        dx2_dt = x4
        dx3_dt = (g * m1 - u1 * k1 / (k2 + x1) ** 2 - kf * (x1 - x2)) / m1
        dx4_dt = (g * m2 + kf * (x1 - x2)) / m2

        # rhs functions list
        self.dxx_dt_symb = sp.Matrix([dx1_dt, dx2_dt, dx3_dt, dx4_dt])
        # ---------end of edit section----------------------------------------

        return self.dxx_dt_symb
