 
"""
system description: brokett-integrator: second order
polynomial, three state components, two affine inputs, no drift

problem specification for control problem: brokett-integrator does not admit a continuously differentiable control
law
"""
import numpy as np
import sympy as sp
from sympy import cos, sin, symbols
from math import pi
from ackrep_core import ResultContainer
from system_models.brockett_integrator_system.system_model import Model
import symbtools as st

from ipydex import IPS

class ProblemSpecification(object):
    # system symbols for setting up the equation of motion
    model = Model()
    x1, x2, x3 = model.xx_symb
    xx = sp.Matrix(model.xx_symb)  # states of system
    u = model.uu_symb # input of system

    zz = st.symb_vector("z1:4")
    vv = st.symb_vector("v1:3")


    @classmethod
    def rhs(cls):
        """ 
        :return: symbolic rhs-functions
        """
        return sp.Matrix(cls.model.get_rhs_symbolic())



def evaluate_solution(solution_data):
    """
    :param solution_data: solution data of problem of solution
    :return: ResultContainer
    """
    expected_final_state = [0.009998614859126023, 0.009957360996513998, 0.009967762399005667, 0.009978835190223072, 0.009952036758485753,
        0.009919577816086893, 0.009980123216488885, 0.009949576326506008, 0.009914913316709522, 0.009953621346320544,
        0.009988298247260814, 0.009967790153907679, 0.009991916470644626, 0.00999680390155051, 0.009916822160298005,
        0.009918459738891074, 0.00989244893289128, 0.009845376127559528, 0.009974183969899204, 0.009887464353824928]

    rc = ResultContainer(score=1.0)
    simulated_final_state = []
    for i in range(len(solution_data)):
        simulated_final_state.append(solution_data[i][-1][-1])
    rc.final_state_errors = [simulated_final_state[i] - expected_final_state[i] for i in np.arange(0, len(simulated_final_state))]
    rc.success = np.allclose(expected_final_state, simulated_final_state, rtol=0, atol=1e-2)

    return rc
