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
        self.u_dim = 4

        # Set "sys_dim" to constant value, if system dimension is constant
        self.sys_dim = 10

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
            # if 1 < t < 1.5:
            #     u3 = 0.5
            # else:
            u3 = 0
            # if 2 < t < 3:
            #     u4 = 1
            # else:
            u4 = 0

            return [u1, u2, u3, u4]

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

        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = self.xx_symb  # state components
        xdot1, xdot2, xdot3, xdot4, xdot5 = sp.symbols("xdot1, xdot2, xdot3, xdot4, xdot5")
        xx = sp.Matrix([[x1], [x2], [x3], [x4], [x5]])

        s2, m1, m2, m3, J2, l0, l1, l2, g = self.pp_symb  # parameters

        u1, u2, u3, u4 = self.uu_symb  # inputs

        # unit vectors
        ex = sp.Matrix([1, 0])
        ey = sp.Matrix([0, 1])

        # basis 1 and 2 (cart positions)
        S1 = G1 = B1 = sp.Matrix([x4, 0])
        S3 = G6 = B2 = sp.Matrix([l0 + x5, 0])

        # center of gravity of load
        S2 = sp.Matrix([x1, x2])

        # suspension points of load
        G3 = S2 - mt.Rz(x3) * ex * s2
        G4 = S2 + mt.Rz(x3) * ex * s2

        # Time derivatives of centers of masses
        Sd1, Sd2, Sd3 = st.col_split(st.time_deriv(st.col_stack(S1, S2, S3), xx))

        # kinetic energy
        T1 = (m1 / 2 * Sd1.T * Sd1)[0]
        T2 = (m2 / 2 * Sd2.T * Sd2)[0] + J2 / 2 * (xdot3) ** 2
        T3 = (m3 / 2 * Sd3.T * Sd3)[0]

        T = T1 + T2 + T3

        # potential energy
        V = m2 * g * S2[1]

        Q1, Q2, Q3, Q4, Q5 = sp.symbols("Q1, Q2, Q3, Q4, Q5")
        QQ = sp.Matrix([[Q1], [Q2], [Q3], [Q4], [Q5]])
        mod = mt.generate_symbolic_model(T, V, xx, QQ)

        F1 = sp.Matrix([u1, 0])
        F2 = sp.Matrix([u2, 0])

        # unit vectors for ropes to split forces according to angles
        rope1 = G3 - S1
        rope2 = G4 - S3
        uv_rope1 = rope1 / sp.sqrt((rope1.T * rope1)[0])
        uv_rope2 = rope2 / sp.sqrt((rope2.T * rope2)[0])

        # simplify expressions by using l1, l2 as shortcuts
        uv_rope1 = rope1 / l1
        uv_rope2 = rope2 / l2

        F3 = uv_rope1 * u3
        F4 = uv_rope2 * u4

        ddelta_theta = st.symb_vector(f"\\delta\\theta_1:{6}")

        delta_S1 = S1 * 0
        delta_S3 = S3 * 0

        delta_G3 = G3 * 0
        delta_G4 = G4 * 0

        for theta, delta_theta in zip(xx, ddelta_theta):

            delta_S1 += S1.diff(theta) * delta_theta
            delta_S3 += S3.diff(theta) * delta_theta

            delta_G3 += G3.diff(theta) * delta_theta
            delta_G4 += G4.diff(theta) * delta_theta

        # simple part (carts)
        delta_W = delta_S1.T * F1 + delta_S3.T * F2

        # rope1 (F3 > 0 means rope is pushing from S1 towards G3)
        delta_W = delta_W + delta_G3.T * F3 - delta_S1.T * F3

        # rope2 (F4 > 0 means rope is pushing from S3 towards G4)
        delta_W = delta_W + delta_G4.T * F4 - delta_S3.T * F4

        QQ_expr = delta_W.jacobian(ddelta_theta).T

        mod.eqns = mod.eqns.subs(
            [(QQ[0], QQ_expr[0]), (QQ[1], QQ_expr[1]), (QQ[2], QQ_expr[2]), (QQ[3], QQ_expr[3]), (QQ[4], QQ_expr[4])]
        )

        mod.calc_state_eq(simplify=False)

        self.dxx_dt_symb = mod.state_eq.subs([(xdot1, x6), (xdot2, x7), (xdot3, x8), (xdot4, x9), (xdot5, x10)])

        return self.dxx_dt_symb
