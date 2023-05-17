import sympy as sp
import symbtools as st
import importlib
import sys, os

# from ipydex import IPS, activate_ips_on_exception

from ackrep_core.system_model_management import GenericModel, import_parameters

# Import parameter_file
params = import_parameters()


# link to documentation with examples: https://ackrep-doc.readthedocs.io/en/latest/devdoc/contributing_data.html


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
            D = 0.6  # duty cycle
            return [D]

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
        x1, x2 = self.xx_symb  # state components
        L, C, R, U_E, k = self.pp_symb  # parameters

        u = self.uu_symb[0]  # inputs

        # define symbolic rhs functions
        dx1_dt = -k * x2 / L + (U_E + k * x2) * u / L
        dx2_dt = (1 - u) * k / C * x1 - x2 / R / C

        # rhs functions matrix
        self.dxx_dt_symb = sp.Matrix([dx1_dt, dx2_dt])
        # ---------end of edit section----------------------------------------

        return self.dxx_dt_symb
