
import sympy as sp
import symbtools as st
import importlib
import sys, os
import numpy as np
from pyblocksim import *
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
        self.u_dim = 7

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
            u1 = 1
            u2 = 1
            u3 = 1
            u4 = 1
            u5 = 1
            u6 = 1
            u7 = 1
            return [u1, u2, u3, u4, u5, u6, u7]
        # ---------end of edit section----------------------------------------

        return uu_rhs


    def get_rhs_func(self):
        msg = "This DAE model has no rhs func like ODE models."
        raise NotImplementedError(msg)
   

    def get_Blockfnc(self):
        """
        generate blockfunctions

        :return: (list) two blockfunctions and input
        """

        x1, x2 = self.xx_symb   #state components
        KR1, TN1, KR2, TN2, T1, K1, K2, K3, K4 = self.pp_symb  #parameters
        KR1 = self.pp_dict[KR1]
        TN1 = self.pp_dict[TN1]
        KR2 = self.pp_dict[KR2]
        TN2 = self.pp_dict[TN2]
        T1 = self.pp_dict[T1]
        K1 = self.pp_dict[K1]
        K2 = self.pp_dict[K2]
        K3 = self.pp_dict[K3]
        K4 = self.pp_dict[K4] 

        #u1, u2, u3, u4, u5, u6, u7 = self.uu_symb   # inputs 
        u1, u2, u3, u4, u5, u6, u7 = inputs('u1, u2, u3, u4, u5, u6, u7')

        DIF1 = Blockfnc(u3 - u1)
        DIF2 = Blockfnc(- u2)

        PI1 = TFBlock(KR1*(1+1/(s*TN1)), DIF1.Y)
        PI2 = TFBlock(KR2*(1+1/(s*TN2)), DIF2.Y)

        SUM11 = Blockfnc(PI1.Y + u4)
        SUM21 = Blockfnc(PI1.Y + u5)

        SUM12 = Blockfnc(PI2.Y + u6)
        SUM22 = Blockfnc(PI2.Y + u7)

        P11 = TFBlock( K1/s         , SUM11.Y)
        P21 = TFBlock( K4/(1+s*T1)  , SUM21.Y)
        P12 = TFBlock( K3/s         , SUM12.Y)
        P22 = TFBlock( K2/s         , SUM22.Y)

        SUM1 = Blockfnc(P11.Y + P12.Y)
        SUM2 = Blockfnc(P22.Y + P21.Y)

        loop(SUM1.Y, u1)
        loop(SUM2.Y, u2)

        return [SUM1, SUM2, u3]
    