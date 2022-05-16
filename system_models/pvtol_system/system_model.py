
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
    
    
    def initialize(self):
        """
        this function is called by the constructor of GenericModel

        :return: None
        """

        # Define number of inputs -- MODEL DEPENDENT
        self.u_dim = 2

        # Set "sys_dim" to constant value, if system dimension is constant 
        # else set "sys_dim" to x_dim -- MODEL DEPENDENT
        self.sys_dim = 6

        # check existance of params file
        self.has_params = True
        self.params = params
        
                                
    # ----------- SET DEFAULT INPUT FUNCTION ---------- # 
    # --------------- Only for non-autonomous Systems
    # --------------- MODEL DEPENDENT
    
    def uu_default_func(self):
        """
        :param t:(scalar or vector) Time
        :param xx_nv: (vector or array of vectors) state vector with numerical values at time t      
        :return:(function with 2 args - t, xx_nv) default input function 
        """ 
        m = self.pp_dict[self.pp_symb[2]]
        T_raise = 2
        T_left = T_raise + 2 + 2
        T_right = T_left + 4
        T_straight = T_right + 2
        T_land = T_straight + 3
        force = 0.75*9.81*m
        force_lr = 0.7*9.81*m
        g_nv = 0.5*self.pp_dict[self.pp_symb[0]]*m
        # create symbolic polnomial functions for raise and land
        poly1 = st.condition_poly(self.t_symb, (0, 0, 0, 0), 
                                  (T_raise, force, 0, 0))
        
        poly_land = st.condition_poly(self.t_symb, (T_straight, g_nv, 0, 0), 
                                      (T_land, 0, 0, 0))
        
        # create symbolic piecewise defined symbolic transition functions
        transition_u1 = st.piece_wise((0, self.t_symb < 0), 
                                      (poly1, self.t_symb < T_raise), 
                                      (force, self.t_symb < T_raise + 2), 
                                      (g_nv, self.t_symb < T_left),
                                      (force_lr, self.t_symb < T_right),
                                      (g_nv, self.t_symb < T_straight),
                                      (poly_land, self.t_symb < T_land),
                                      (0, True))
        
        transition_u2 = st.piece_wise((0, self.t_symb < 0), 
                                      (poly1, self.t_symb < T_raise), 
                                      (force, self.t_symb < T_raise + 2), 
                                      (force_lr, self.t_symb < T_left),
                                      (g_nv, self.t_symb < T_right),
                                      (force_lr, self.t_symb < T_straight),
                                      (poly_land, self.t_symb < T_land),
                                      (0, True))
        
        # transform symbolic to numeric function 
        transition_u1_func = st.expr_to_func(self.t_symb, transition_u1)
        transition_u2_func = st.expr_to_func(self.t_symb, transition_u2)
        
        
        def uu_rhs(t, xx_nv):
            
            u1 = transition_u1_func(t)
            u2 = transition_u2_func(t)

            return [u1, u2]
        
        return uu_rhs
         
    # ----------- SYMBOLIC RHS FUNCTION ---------- # 
    # --------------- MODEL DEPENDENT  
    
    def get_rhs_symbolic(self):
        """
        :return:(scalar or array) symbolic rhs-functions
        """
        if self.dxx_dt_symb is not None:
            return self.dxx_dt_symb
            
        x1, x2, x3, x4, x5, x6 = self.xx_symb
        g, l, m, J = self.pp_symb 

        # input     
        u1, u2 = self.uu_symb

        # create symbolic rhs functions
        dx1_dt = x2
        dx2_dt = -sp.sin(x5)/m * (u1 + u2)
        dx3_dt = x4
        dx4_dt = sp.cos(x5)/m * (u1 + u2) - g
        dx5_dt = x6 *2*sp.pi/360
        dx6_dt = l/J * (u2 - u1) *2*sp.pi/360
        
        # put rhs functions into a vector
        self.dxx_dt_symb = [dx1_dt, dx2_dt, dx3_dt, dx4_dt, dx5_dt, dx6_dt]
        
        return self.dxx_dt_symb
    