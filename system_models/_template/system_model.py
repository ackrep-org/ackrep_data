
import sympy as sp
import symbtools as st
import importlib
import sys, os
from ipydex import IPS, activate_ips_on_exception  # for debugging only

from ackrep_core.system_model_management import GenericModel, import_parameters
from ackrep_core.core import get_metadata_from_file

# Import parameter_file
yml_path = os.path.join(os.path.dirname(__file__), "metadata.yml")
md = get_metadata_from_file(yml_path)
params = import_parameters(md["key"])



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
        

        # Define number of inputs -- MODEL DEPENDENT
        self.u_dim = ...

        # Set "sys_dim" to constant value, if system dimension is constant 
        # else set "sys_dim" to x_dim -- MODEL DEPENDENT
        self.sys_dim = ...

        # Adjust sys_dim to dimension fitting to default parameters
        # only needed for n extendable systems -- MODEL DEPENDENT
        self.default_param_sys_dim = ...
       

        # check existance of params file
        self.has_params = True
        self.params = params
        
        # Initialize     
        super().__init__(x_dim=x_dim, u_func=u_func, pp=pp)
                                


    # ----------- SET DEFAULT INPUT FUNCTION ---------- # 
    # --------------- Only for non-autonomous Systems
    # --------------- MODEL DEPENDENT
    
    def uu_default_func(self):
        """
        :param t:(scalar or vector) Time
        :param xx_nv: (vector or array of vectors) state vector with numerical values at time t      
        :return:(function with 2 args - t, xx_nv) default input function 
        """ 
        
        
        def uu_rhs(t, xx_nv):
            
            return [..]
        
        return uu_rhs
         
    # ----------- SYMBOLIC RHS FUNCTION ---------- # 
    # --------------- MODEL DEPENDENT  
    
    def get_rhs_symbolic(self):
        """
        :return:(scalar or array) symbolic rhs-functions
        """
        if self.dxx_dt_symb is not None:
            return self.dxx_dt_symb

        .. = self.xx_symb
        .. = self.pp_symb 

        # input     
        .. = self.uu_symb

        # create symbolic rhs functions
        dx_dt = ..
        
        # put rhs functions into a vector
        self.dxx_dt_symb = [dx_dt,..]
        
        return self.dxx_dt_symb
    