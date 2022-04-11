# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 13:33:34 2021

@author: Jonathan Rockstroh
"""

import sympy as sp
import symbtools as st
import importlib
import sys
from ipydex import IPS, activate_ips_on_exception  # for debugging only


from ackrep_core.system_model_management import GenericModel

# Import parameter_file
# Name of the parameter file without ending -- MODEL DEPENDENT
# parameter_file_name = 'lorenz_parameter'


try:
    import parameters
except ImportError:
    from . import parameters
params = parameters

# try to find ModuleSpecs
# param_module = importlib.util.find_spec(parameter_file_name)
# # if ModuleSpecs are found, paramters_package can be loaded
# if param_module is not None:
#     params = importlib.import_module(parameter_file_name)
#     print("module specs found successfully!")
# else:
#     print("no module specs found!")

class Model(GenericModel): 

    def __init__(self, x_dim=None, u_func=None, pp=None):
        """
        :param x_dim:(int, positive) dimension of the state vector 
                                - has no effect for non-extendible systems
        :param u_func:(callable) input function, args: time, state vector
                        return: list of numerical input values 
                        - has no effect for autonomous systems    
        :param pp:(vector or dict-type with floats>0) parameter values
        :return:
        """
        # Initialize all Parameters of the Model-Object with None    
        super().__init__()
        # check existance of params file -> if not: System is defined to hasn't 
        # parameters
        self.has_params = True
        try:
            params.get_default_parameters()
        except NameError:
            self.has_params = False     
        # Define number of inputs -- MODEL DEPENDENT
        self.u_dim = 0
        # Set "sys_dim" to constant value, if system dimension is constant 
        # else set "sys_dim" to x_dim -- MODEL DEPENDENT
        sys_dim = 3
        # Adjust sys_dim to dimension fitting to default parameters
        # only needed for n extendable systems -- MODEL DEPENDENT
        default_param_sys_dim = None
        if x_dim is None and sys_dim is None:
            sys_dim = default_param_sys_dim
        # check existance of params file -> if not: System is defined to hasn't 
        # parameters
        self.has_params = True
        try:
            params.get_default_parameters()
        except NameError:
            self.has_params = False  
        # Set self.n
        self._set_dimension(sys_dim)        
        # Create symbolic input vector
        self._create_symb_uu(self.u_dim)
        # Create symbolic xx and xxuu
        self._create_symb_xx_xxuu()
        # Create parameter dict, subs_list and symbolic parameter vector
        self.set_parameters(pp)
        # Create Symbolic parameter vector and subs list
        self._create_symb_pp()
        # Create Substitution list
        self._create_subs_list()
        # choose input function
        self.set_input_func(self.uu_default_func())
        if u_func is not None:
            self.set_input_func(u_func)


    # ----------- VALIDATE PARAMETER VALUES ---------- #
    # -------------- MODEL DEPENDENT 
    
    def _validate_p_values(self, p_value_list):
        """ raises exception if values in list aren't valid 
        :param p_value_list:(float) list of parameter values
        """
        # Check for convertability to float
        try: float(p_value_list)
        except ValueError:
                raise Exception(":param pp: Values are not valid. \
                                (aren't convertible to float)")
                                 
        # Check if values are in required range --- MODEL DEPENDENT                         
        assert not any(flag <= 0 for flag in p_value_list), \
                        ":param pp: does have values <= 0"
                                


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
            return []
        
        return uu_rhs

         
    # ----------- SYMBOLIC RHS FUNCTION ---------- # 
    # --------------- MODEL DEPENDENT  
    
    def get_rhs_symbolic(self):
        """
        :return:(scalar or array) symbolic rhs-functions
        """
        if self.dxx_dt_symb is not None:
            return self.dxx_dt_symb
        x, y, z = self.xx_symb
        r, b, sigma = self.pp_symb
        # create symbolic rhs function vector
        dx1_dt = - sigma*x + sigma*y
        dx2_dt = -x*z + r*x - y
        dx3_dt = x*y - b*z
        
        self.dxx_dt_symb = [dx1_dt, dx2_dt, dx3_dt]
        
        return self.dxx_dt_symb
 
    
    # ----------- SET_PARAMETERS ---------- #
    # ------------- MODEL INDEPENDENT
    # ------------- Reason for not being in Class GenericModel: 
    # -------------                         Uses imported params module
 
    def set_parameters(self, pp):
        """
        :param pp:(vector or dict-type with floats>0) parameter values
        :param x_dim:(positive int)
        """       

        # Case: System doesn't have parameters
        if not self.has_params:
            return  
        
        # Case: No symbolic parameters created
        if self.pp_symb is None: 
            try:
                symb_pp = self._create_n_dim_symb_parameters()
            except AttributeError: # To ensure compatibility with old classes
                symb_pp = None
            # Check if system has constant dimension
            if  symb_pp is None:
                symb_pp = list(params.get_default_parameters().keys())
            self._create_symb_pp(symb_pp)

        # Case: Use Default Parameters
        if pp is None:
            pp_dict = params.get_default_parameters()
            # Check if a possibly given system dimension fits to the default
            # parameter length
            assert len(self.pp_symb) == len(pp_dict), \
                "Expected :param pp: to be given, because the amount of \
                    parameters needed (" + str(len(self.pp_symb)) +") \
                    for the system of given dimension (" + str(self.n) + ") \
                    doesn't fit to the number of default parameters (" \
                        + str(len(pp_dict)) + ")."
            self.pp_dict = pp_dict
            return
        
        # Check if pp is list or dict type
        assert isinstance(pp, dict) or isinstance(pp, list),\
                            ":param pp: must be a dict or list type object"
                            
        # Case: Individual parameter (list or dict type) variable is given
        if pp is not None:
            # Check if pp has correct length                    
            assert len(self.pp_symb) == len(pp), \
                    ":param pp: Given Dimension: " + str(len(pp)) \
                    + ", Expected Dimension: " + str(len(self.pp_symb))
            # Case: parameter dict ist given -> individual parameter symbols 
            # and values
            if isinstance(pp, dict):
                self._create_individual_p_dict(pp)
                return
            # Case: Use individual parameter values
            else:                     
                self._create_individual_p_dict(pp, self.pp_symb)
                return
        
        # Case: Should never happen.
        raise Exception("Critical Error: Check Source Code of set_parameters.")  