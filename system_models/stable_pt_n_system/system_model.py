# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 13:33:34 2021

@author: Jonathan Rockstroh
"""

import sympy as sp
import symbtools as st
import importlib
import sys, os
from itertools import combinations as comb
import numpy as np
from ipydex import IPS, activate_ips_on_exception  # for debugging only

from ackrep_core.system_model_management import GenericModel, import_parameters

# Import parameter_file
params = import_parameters()

    
class Model(GenericModel): 
   
    def initialize(self):
        """
        this function is called by the constructor of GenericModel

        :return: None
        """
    
        # Define number of inputs
        self.u_dim = 1
       
        # Adjust sys_dim to dimension fitting to default parameters
        self.default_param_sys_dim = 2
        
        # check existance of params file -> if not: System is defined to hasn't 
        # parameters
        self.has_params = True
        self.params = params

        

    # ----------- _CREATE_N_DIM_SYMB_PARAMETERS ---------- #
    # ------------- MODEL DEPENDENT, IF: Model has parameters
    # -------------                      AND is n extendible
    # ------------- If Model isn't n extendible set return to None

    def _create_n_dim_symb_parameters(self):
        """Creates the symbolic parameter list for n extendible systems
        
        :return: 
            pp_symb: list of sympy.symbol type entries
                contains the symbolic parameters
            None: 
                if system has a constant dimension
        """
        # Create T_i are the time constants and K is the proportional factor
        pp_symb = [sp.Symbol('T' + str(i)) for i in range(0, self.n)]
        pp_symb = [sp.Symbol('K')] + pp_symb
        return pp_symb

                                

    # ----------- SET DEFAULT INPUT FUNCTION ---------- # 
    # --------------- Only for non-autonomous Systems
    # --------------- MODEL DEPENDENT
    
    def uu_default_func(self):
        """
        :param t:(scalar or vector) Time
        :param xx_nv: (vector or array of vectors) state vector with 
                                                    numerical values at time t      
        :return:(function with 2 args - t, xx_nv) default input function 
        """ 
        def uu_rhs(t, xx_nv):          
            u = 10
            c = t > 0
            try: 
                uu = list(c*u)
            except TypeError:
                uu = [c*u]
            uu = [np.sin(0.387298334621*t)]
            return uu
        
        return uu_rhs

         
    # ----------- SYMBOLIC RHS FUNCTION ---------- # 
    # --------------- MODEL DEPENDENT  
    
    def get_rhs_symbolic(self):
        """
        :return:(matrix) symbolic rhs-functions
        """
        if self.dxx_dt_symb is not None:
            return self.dxx_dt_symb
        xx_symb = self.xx_symb  
        u_symb= self.uu_symb[0]
        K_symb = self.pp_symb[0]
        TT_symb = self.pp_symb[1:]
        # create symbolic rhs functions
        # define entry/derivation 1 to n-1
        dxx_dt = [entry for entry in xx_symb]
        dxx_dt = dxx_dt[1:]
        # define entry/derivation n
        sum_vec = []
        # write all summands of the expanded inverse laplace into a vector
        for k in range(self.n + 1):
            next_elem = self._create_factor(TT_symb, k)        
            sum_vec = sum_vec + [next_elem]
        # calculate expanded inverse laplace
        
        inv_laplace = 0
        for i in range(len(sum_vec) - 1):
            inv_laplace = inv_laplace + sum_vec[i]*xx_symb[i]
        
        
        dxn_dt = (K_symb*u_symb - inv_laplace) * 1/sum_vec[-1]
        dxx_dt = dxx_dt + [dxn_dt]
        # put rhs functions into a vector
        self.dxx_dt_symb = dxx_dt
        
        return self.dxx_dt_symb
        
    
    # ----------- CREATE_FACTOR ---------- # 
    # --------------- Exclusivly made for this model
    
    def _create_factor(self, pp_symb, deriv_nr):
        '''Auxiliary Function to create the symb function of the pt_n element
        returns the factor infront of the state, which represents the 
        deriv_nr-th derivation of y. Take a look at product in the equation 
        for dx_n_dt in the model documentation.
        
        :param pp_symb: list of sympy variables
            symbolic parameter vectorm, which only contains the time coefficients
        :param deriv_nr: int >= 0
            number of the state of the dxn_dt solution for which the leading factor
            shall be calculated
            
        :return summand: sympy function
            returns the summand of
    
        '''
        # assure, that deriv_nr is a proper value
        assert deriv_nr >= 0, "deriv_nr needs to be positive or zero"
    
        assert deriv_nr <= len(pp_symb), "deriv_nr can't be greater than the \
                                            length of the pp_symb list"
        # initialize summand
        factor = 0
        # Solve Special case of deriv_nr = 0
        if deriv_nr == 0:
            factor = 1
        # create factor for deriv_nr > 0
        else:
            # create list of all deriv_nr-element combinations 
            subsummand_vec = list(comb(pp_symb, deriv_nr))
            # save length of the sublists, to improve performance
            sublist_len = len(subsummand_vec[0])
            # iterate over the combinations and create summand = sum of subsummands
            for i in range(len(subsummand_vec)):
                subsummand = 1
                # create one summand of the factor
                for j in range(sublist_len):
                    subsummand = subsummand * subsummand_vec[i][j]
                # add subsummand to factor    
                factor = factor + subsummand
                
        return factor