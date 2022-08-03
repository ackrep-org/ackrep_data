import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.special import i1

import pyinduct as pi
from pyinduct.core import integrate_function

from simulation import ModalApproximation, FEMApproximation


class AnalyticBacksteppingController(pi.SimulationInput):
    """
    Implementation of the backstepping controller from example 4.1 in
    [KristicEtAl08].
    """

    def __init__(self, spat_dom, orig_params, sym_system):
        super().__init__("AnalyticBacksteppingControl")
        self.dom = spat_dom
        _, _, self.lamda, _, _ = orig_params
        self.sym_sys = sym_system
        self.k_sim = None
        self._build_feedback()

    def _bst_kernel(self, x, y):
        arg = np.sqrt(self.lamda * (x**2 - y**2))
        if np.isclose(arg, 0):
            return -self.lamda * y / 2
        k = -self.lamda * x * i1(arg) / arg
        return k

    def _build_feedback(self):
        """Approximate the analytic kernel with the simulation basis"""

        def _kernel_factory(frac):
            def _kernel_func(y):
                return self._bst_kernel(1, y) * frac(y)

            return _kernel_func

        sys_base = pi.get_base(self.sym_sys.base_lbl)
        areas = [self.dom.bounds]
        k_sys = np.atleast_2d([integrate_function(_kernel_factory(f), areas)[0] for f in sys_base])
        if isinstance(self.sym_sys, ModalApproximation):
            self.k_sim = k_sys
        elif isinstance(self.sym_sys, FEMApproximation):
            # manually compute feedback vector
            self.k_sim = self.sym_sys.transform_feedback(k_sys, sys_base)
        else:
            raise NotImplementedError

    def _calc_output(self, **kwargs):
        sim_weights = kwargs["weights"]
        u = self.k_sim @ sim_weights
        return dict(output=u)


class ApproximatedBacksteppingController(pi.SimulationInput):

    idx = 0

    def __init__(self, orig_params, tar_params, n_modal, spatial_domain, sym_system, be_verbose=False):
        super().__init__("ApproximatedBacksteppingController")
        ApproximatedBacksteppingController.idx += 1
        self.orig_base_lbl = "orig_eigen_vectors_{}".format(self.idx)
        self.tar_base_lbl = "tar_eigen_vectors_{}".format(self.idx)
        self.sym_sys = sym_system
        self.verbose = be_verbose
        self.k_sim = None

        self._build_bases(n_modal, orig_params, tar_params, spatial_domain)
        self._build_feedback()

    def _build_feedback(self):
        # weak form of the controller
        orig_x = pi.FieldVariable(self.orig_base_lbl)
        tar_x = pi.FieldVariable(self.tar_base_lbl, weight_label=self.orig_base_lbl)
        cont_weak_form = pi.WeakFormulation(
            [pi.ScalarTerm(orig_x(1)), pi.ScalarTerm(tar_x(1), scale=-1)], name="feedback_law"
        )
        # create implementation that fits the simulated system
        if isinstance(self.sym_sys, ModalApproximation):
            # simply use pyinduct feedback framework
            self.approx_cont = pi.StateFeedback(cont_weak_form)
        elif isinstance(self.sym_sys, FEMApproximation):
            # manually compute feedback vector
            ce = pi.parse_weak_formulation(cont_weak_form, finalize=False)
            k_mod = ce.dynamic_forms[self.orig_base_lbl].matrices["E"][0][1]
            modal_base = pi.get_base(self.orig_base_lbl)
            self.k_sim = self.sym_sys.transform_feedback(k_mod, modal_base)
        else:
            raise NotImplementedError

    def _build_bases(self, n_modal, orig_params, tar_params, spatial_domain):
        """
        Compute approximation bases for original and target system
        """
        # eigenvalues and -vectors of the original system
        orig_eig_values, orig_eig_vectors = pi.SecondOrderDirichletEigenfunction.cure_interval(
            spatial_domain, param=orig_params, n=n_modal
        )
        norm_orig_eig_vectors = pi.normalize_base(orig_eig_vectors)
        pi.register_base(self.orig_base_lbl, norm_orig_eig_vectors)

        # eigenvectors of the target system (slightly different)
        target_eig_freq = pi.SecondOrderDirichletEigenfunction.eigval_tf_eigfreq(tar_params, eig_val=orig_eig_values)
        orig_scale = np.array([vec(0) for vec in norm_orig_eig_vectors.derive(1)])
        tar_scale = orig_scale / target_eig_freq
        _, tar_eig_vectors = pi.SecondOrderDirichletEigenfunction.cure_interval(
            spatial_domain, param=tar_params, eig_val=orig_eig_values, scale=tar_scale
        )
        pi.register_base(self.tar_base_lbl, tar_eig_vectors)
        if self.verbose:
            pi.visualize_functions(orig_eig_vectors)
            pi.visualize_functions(norm_orig_eig_vectors)
            pi.visualize_functions(tar_eig_vectors)

    def _calc_output(self, **kwargs):
        if isinstance(self.sym_sys, ModalApproximation):
            u = self.approx_cont(**kwargs)
        elif isinstance(self.sym_sys, FEMApproximation):
            sim_weights = kwargs["weights"]
            u = self.k_sim @ sim_weights
        else:
            raise NotImplementedError
        return dict(output=u)

    def __del__(self):
        pi.deregister_base(self.orig_base_lbl)
        pi.deregister_base(self.tar_base_lbl)
        ApproximatedBacksteppingController.idx -= 1
