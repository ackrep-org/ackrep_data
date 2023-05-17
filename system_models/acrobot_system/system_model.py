import sympy as sp
import symbtools as st
import importlib
import sys, os
import symbtools.modeltools as mt

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

            if t < 5:
                u1 = 0.5
            else:
                u1 = 0

            return [u1]

        # ---------end of edit section----------------------------------------

        return uu_rhs

    def get_rhs_symbolic(self):
        """
        define symbolic rhs function

        :return: matrix of symbolic rhs-functions
        """

        np = 1
        nq = 1
        n = np + nq

        x1, x2, x3, x4 = self.xx_symb
        ttheta = st.row_stack(x1, x2)

        xdot1, xdot2 = sp.symbols("xdot1, xdot2")

        s1, s2, m1, m2, J1, J2, l1, g = self.pp_symb

        u1 = self.uu_symb[0]

        mt.Rz(x2)

        # unuit vectors
        ex = sp.Matrix([1, 0])
        ey = sp.Matrix([0, 1])

        # coordinates of the centers of gravity and joints
        S1 = mt.Rz(x1) * (-ey) * s1
        G1 = mt.Rz(x1) * (-ey) * l1  # "elbow joint"
        S2 = G1 + mt.Rz(x2 + x1) * (-ey) * s2

        # time derivatives of the center of gravity coordinates
        Sd1, Sd2 = st.col_split(st.time_deriv(st.col_stack(S1, S2), ttheta))

        # kinetic energy
        T_rot = (J1 * x3**2) / 2 + (J2 * (x4 + x3) ** 2) / 2
        T_trans = (m1 * Sd1.T * Sd1 + m2 * Sd2.T * Sd2) / 2

        T = T_rot + T_trans[0]

        # potential energy
        V = m1 * g * S1[1] + m2 * g * S2[1]

        external_forces = [0, u1]
        assert not any(external_forces[:np])
        mod = mt.generate_symbolic_model(T, V, ttheta, external_forces)

        mod.calc_state_eq(simplify=False)

        state_eq = mod.state_eq.subs([(xdot1, x3), (xdot2, x4)])

        return state_eq
