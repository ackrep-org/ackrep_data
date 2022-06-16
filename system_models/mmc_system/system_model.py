
import sympy as sp
import symbtools as st
import importlib
from sympy import I
import numpy as np
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
        self.u_dim = 4

        # Set "sys_dim" to constant value, if system dimension is constant 
        # else set "sys_dim" to x_dim -- MODEL DEPENDENT
        self.sys_dim = 8
       
        # check existance of params file
        self.has_params = True
        self.params = params
        
    
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

        vdc, vg, omega, Lz, Mz, R, L = list(self.pp_dict.values())
        Ind_sum = Mz + Lz
        def uu_rhs(t, xx_nv):
            	
            es0, ed0, es, ed, iss, iss0, i, theta = xx_nv
                   
            # Define Input Functions
            Kp = 5
            # define reference trajectory for i
            T_dur = 1.5
            tau = t/T_dur
            i_max = 10
        
            i_ref = 10
            dt_i_ref = 0            
            
            if tau < 1:
                i_ref = 4 + (i_max - 4) *np.sin(0.5*tau *np.pi) *np.sin(0.5*tau *np.pi)
                dt_i_ref = np.pi/T_dur * np.sin(0.5*tau *np.pi) *np.cos(0.5*tau *np.pi)
            if t < 1:
                i_ref = 4
                dt_i_ref = 0  
            
            vy = vg + (R+1j*omega*L) *i + 1*(i_ref - i) + L *dt_i_ref
            vy = complex(vy)
            
            vy0 = -1/6 *np.absolute(vy) *np.real(
                    np.exp(3 *1j*(theta + np.angle(vy))) )
            
            vx = 1j *omega*Ind_sum*iss - Kp*iss # p- controller to get iss
            
                # define reference trajectory of es0
            T_dur = 0.5 #duration of es0 ref_traj
            tau = t/T_dur
            #es0_max = 68
            #es0_ref = sp.Piecewise((0, tau < 0), (es0_max *sp.sin( tau*sp.pi/2)  \
            #                        *sp.sin( tau*sp.pi/2), tau < 1), (es0_max, True) )
            es0_ref = 56
                # derive iss0 ref trajectory
            iss0_ref = 1/vdc *np.real(i *np.conjugate(vy)) + Kp *(es0_ref - es0)
            
            # p-controller for vx0
            vx0 = Kp*(iss0_ref - iss0) 


            return [vy, vy0, vx, vx0]
        
        return uu_rhs

         
    # ----------- SYMBOLIC RHS FUNCTION ---------- # 
    # --------------- MODEL DEPENDENT  
    
    def get_rhs_symbolic(self):
        """
        :return:(matrix) symbolic rhs-functions
        """
        if self.dxx_dt_symb is not None:
            return self.dxx_dt_symb
        
        es0, ed0, es, ed, iss, iss0, i, theta = self.xx_symb
        vdc, vg, omega, Lz, Mz, R, L = self.pp_symb 
        # u0 = input force     
        vy0, vy, vx0, vx = self.uu_symb
        
        # Auxiliary variables
        Ind_sum = Lz + Mz
        vydelta = vy - Mz *(I *omega*i + 1/L *(vy - (R + I *omega*L) *i - vg))
        
        # create symbolic rhs functions
        des0_dt = vdc*iss0 - sp.re(i*sp.conjugate(vy) )
        
        ded0_dt = -2 *vy0*iss0 - sp.re(sp.conjugate(iss)*vydelta )
        
        des_dt = vdc*iss - sp.exp(- 3*I*theta) *sp.conjugate(vy) \
                *sp.conjugate(i) - 2 *i*vy0 - I *omega*es
                
        ded_dt = vdc*i - sp.exp(- 3*I*theta) *sp.conjugate(iss) \
                *sp.conjugate(vydelta) - 2*iss*vy0 \
                - 2*iss0*vydelta - I *omega*ed
                
        diss_dt = 1/Ind_sum * (vx -  I*omega*Ind_sum*iss)
        
        diss0_dt = vx0/Ind_sum
        
        di_dt = 1/L * (vy - (R + I *omega*L) *i - vg) 
        
        dtheta_dt = 2*sp.pi/360*omega
        
        # put rhs functions into a vector
        self.dxx_dt_symb = sp.Matrix([des0_dt, ded0_dt, des_dt, ded_dt, diss_dt, 
                            diss0_dt, di_dt, dtheta_dt])
        
        return self.dxx_dt_symb
    
    # ----------- NUMERIC RHS FUNCTION ---------- # 
    # -------------- written sepeeratly cause it seems like that lambdify can't
    # -------------- handle complex values in a proper way
     
    def get_rhs_func(self):
        """
        Creates an executable function of the model which uses:
            - the current parameter values
            - the current input function
        
        To evaluate the effect of a changed parameter set a new rhs_func needs 
        to be created with this method.
        
        :return:(function) rhs function for numerical solver like 
                            scipy.solve_ivp
        """
     
        # create rhs function
        def rhs(t, xx_nv):
            """
            :param t:(tuple or list) Time
            :param xx_nv:(self.n-dim vector) numerical state vector
            :return:(self.n-dim vector) first time derivative of state vector
            """
            uu_nv = self.uu_func(t, xx_nv)
            vy, vy0, vx, vx0 = uu_nv
            es0, ed0, es, ed, iss, iss0, i, theta = xx_nv
            vdc, vg, omega, Lz, Mz, R, L = list(self.pp_dict.values())
            Ind_sum = Mz + Lz 
            # = vy-Mz(j*omega*i-dt_i)
            vydelta = vy - Mz *(1j *omega*i + 1/L 
                                *(vy - (R + 1j *omega*L) *i - vg) )
            
            dt_es0 = vdc*iss0 - np.real(i*np.conjugate( vy) )
            dt_ed0 = -2 *vy0*iss0 - np.real(np.conjugate( iss)*vydelta )
            
            dt_es = vdc*iss - np.exp( -3*1j*theta) *np.conjugate( vy) \
                    *np.conjugate( i) - 2 *i*vy0 - 1j *omega*es
                    
            dt_ed = vdc*i - np.exp( -3*1j*theta) *np.conjugate( iss) \
                    *np.conjugate( vydelta)- 2 *iss*vy0 - 2 *iss0*vydelta \
                    - 1j *omega*ed        
                    
            dt_iss = 1/Ind_sum * (vx -  1j*omega*Ind_sum*iss)
            dt_iss0 = vx0/Ind_sum      
            dt_i = 1/L * (vy - (R+1j *omega*L) *i - vg)
            dt_theta = omega
            
            dxx_dt_nv = [dt_es0, dt_ed0, dt_es, dt_ed, dt_iss, 
                         dt_iss0, dt_i, dt_theta
                         ]

            return dxx_dt_nv
            
        return rhs
    
    