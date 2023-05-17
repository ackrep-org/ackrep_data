import sympy as sp
import symbtools as st
import importlib
import sys, os
from pyblocksim import *

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
            u = sp.sin(4 * sp.pi * t)
            return [u]

        # ---------end of edit section----------------------------------------

        return uu_rhs

    def get_rhs_func(self):
        msg = "This DAE model has no rhs func like ODE models."
        raise NotImplementedError(msg)

    def get_rhs_symbolic(self):
        """This model is not represented by the standard rhs equations."""
        return False

    def get_blockfnc(self):
        """
        Calculate resulting blockfunction

        :return: final block, input
        """

        # ---------start of edit section--------------------------------------

        s1, s2, y1, y2, T_storage = self.pp_symb  # parameters
        s1 = self.pp_dict[s1]
        s2 = self.pp_dict[s2]
        y1 = self.pp_dict[y1]
        y2 = self.pp_dict[y2]
        T_storage = self.pp_dict[T_storage]

        _tanh_factor = 1e3

        def step_factory(y0, y1, x_step):
            """
            Factory to create continously approximated step functions
            """
            # tanh maps R to (-1, 1)

            # first map R to (0, 1)
            # then map (0, 1) -> (y0, y1)

            dy = y1 - y0

            def fnc(x, module=sp):
                return (module.tanh(_tanh_factor * (x - x_step)) + 1) / 2 * dy + y0

            fnc.__doc__ = "approximated step function %f, %f, %f" % (y0, y1, x_step)

            return fnc

        #  togehter these two are applied to the input
        step1 = step_factory(-1, 0, s1)
        step2 = step_factory(0, 1, s2)

        # the third step-function limits the input of the PT1 between between 0 and 1
        # the exact value for 63 % Percent
        time_const_value = 1 - np.exp(-1)
        step3 = step_factory(0, 1, time_const_value)

        hyst_in, fb = inputs("hyst_in, fb")  # overall input and internal feedback

        # the sum of the two step-functions is basically a three-point-controller
        SUM1 = Blockfnc(step1(hyst_in) + step2(hyst_in) + fb)

        LIMITER = Blockfnc(step3(SUM1.Y))

        PT1_storage = TFBlock(1 / (T_storage * s + 1), LIMITER.Y)

        loop(PT1_storage.Y, fb)

        # gain and offset for the output
        Result_block = Blockfnc(PT1_storage.Y * (y2 - y1) + y1)

        return [Result_block, hyst_in]

    def input_ramps(self, t):
        """
        input signal

        :return: numerical value
        """

        T1 = 10
        T2 = 20
        k1 = 1
        k2 = 1

        if t < 0:
            return 0
        elif t < T1:
            return k1 * t
        elif t < T2:
            return k1 * T1 - k2 * (t - T1)
        else:
            return 0
