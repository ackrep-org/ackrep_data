from abc import abstractmethod
import numpy as np

import pyinduct as pi


class ApproximatedSystem:
    @abstractmethod
    def get_system(self, u):
        pass

    @abstractmethod
    def get_initial_state(self, initial_profile, u):
        pass

    @abstractmethod
    def get_results(self, weights, u, temp_dom, spat_dom, name=None):
        pass


class ModalApproximation(ApproximatedSystem):
    """
    Build a simulation model using modal transformation
    """
    def __init__(self, params, n_modal, spat_dom):
        a2 = params[0]
        z_start, z_end = spat_dom.bounds
        self.base_lbl = "eigen_vectors"

        # eigenvalues and -vectors of the system system
        eig_values, eig_vectors = \
            pi.SecondOrderDirichletEigenfunction.cure_interval(spat_dom,
                                                               param=params,
                                                               n=n_modal)
        # pi.visualize_functions(orig_eig_vectors)
        norm_eig_vectors = pi.normalize_base(eig_vectors)
        # pi.visualize_functions(normalized_eig_vectors)
        pi.register_base(self.base_lbl, norm_eig_vectors)

        self.a_mat = np.diag(np.real_if_close(eig_values))
        b_mat = -a2 * np.array([eig_vect.derive()(z_end)
                                for eig_vect in norm_eig_vectors])
        self.b_mat = np.reshape(b_mat, (b_mat.size, 1))

    def get_system(self, u):
        sys = pi.StateSpace(self.a_mat,
                            self.b_mat,
                            base_lbl=self.base_lbl,
                            input_handle=u)
        return sys

    def get_initial_state(self, initial_profile, u):
        eig_vectors = pi.get_base(self.base_lbl)
        initial_weights = pi.project_on_base(initial_profile, eig_vectors)
        return initial_weights

    def get_results(self, weights, u, temp_dom, spat_dom, name=None):
        ed = pi.evaluate_approximation(self.base_lbl,
                                       weights,
                                       temp_dom,
                                       spat_dom,
                                       name="x(z,t)" + name)
        return ed

    def __del__(self):
        pi.deregister_base(self.base_lbl)


class FEMApproximation:
    def __init__(self, params, n_fem, spat_bounds):
        self.params = params
        self.approx_cnt = n_fem
        self.bounds = spat_bounds

        self.base_lbl = "fem_base"
        self.a_bar = None
        self.a_tilde = None
        self.a_tilde_inv = None
        self.b_bar = None
        self.b1 = None
        self._build_system()

    def _build_system(self):
        # initial and test functions
        nodes = pi.Domain(self.bounds, num=self.approx_cnt)
        full_fem_base = pi.LagrangeFirstOrder.cure_interval(nodes)
        act_fem_base = pi.Base(full_fem_base[-1])
        not_act_fem_base = pi.Base(full_fem_base[1:-1])
        pi.register_base("act_base", act_fem_base)
        pi.register_base("sim_base", not_act_fem_base)
        pi.register_base(self.base_lbl, full_fem_base)

        a2, a1, a0, _, _ = self.params

        # weak form
        x = pi.FieldVariable("sim_base")
        x_dt = x.derive(temp_order=1)
        x_dz = x.derive(spat_order=1)
        phi = pi.TestFunction("sim_base")
        phi_dz = phi.derive(1)
        act_phi = pi.ScalarFunction("act_base")
        act_phi_dz = act_phi.derive(1)
        u = pi.ConstantTrajectory(0)  # dummy input

        weak_form = pi.WeakFormulation([
            # ... of the homogeneous part of the system
            pi.IntegralTerm(pi.Product(x_dt, phi),
                            limits=self.bounds),
            pi.IntegralTerm(pi.Product(x_dz, phi_dz),
                            limits=self.bounds,
                            scale=a2),
            pi.IntegralTerm(pi.Product(x_dz, phi),
                            limits=self.bounds,
                            scale=-a1),
            pi.IntegralTerm(pi.Product(x, phi),
                            limits=self.bounds,
                            scale=-a0),

            # ... of the inhomogeneous part of the system
            pi.IntegralTerm(pi.Product(pi.Product(act_phi, phi),
                                       pi.Input(u, order=1)),
                            limits=self.bounds),
            pi.IntegralTerm(pi.Product(pi.Product(act_phi_dz, phi_dz),
                                       pi.Input(u)),
                            limits=self.bounds,
                            scale=a2),
            pi.IntegralTerm(pi.Product(pi.Product(act_phi_dz, phi),
                                       pi.Input(u)),
                            limits=self.bounds,
                            scale=-a1),
            pi.IntegralTerm(pi.Product(pi.Product(act_phi, phi),
                                       pi.Input(u)),
                            limits=self.bounds,
                            scale=-a0)],
            name="main_system")

        # system matrices \dot x = A x + b0 u + b1 \dot u
        cf = pi.parse_weak_formulation(weak_form)
        ss = pi.create_state_space(cf)

        a_mat = ss.A[1]
        b0 = ss.B[0][1]
        self.b1 = ss.B[1][1]

        # Idea: \bar x = \tilde A x + b1 u
        self.a_tilde = np.diag(np.ones(a_mat.shape[0]), 0)
        self.a_tilde_inv = np.linalg.inv(self.a_tilde)

        # Yields: \dot \bar x = \bar A \bar x + \bar b u
        self.a_bar = (self.a_tilde @ a_mat) @ self.a_tilde_inv
        self.b_bar = self.a_tilde @ (a_mat @ self.b1) + b0

    def get_system(self, u):
        ss = pi.StateSpace(self.a_bar, self.b_bar,
                           base_lbl="transformed_base", input_handle=u)
        return ss

    def get_initial_state(self, initial_profile, u):
        full_initial_state = pi.project_on_base(initial_profile,
                                                pi.get_base(self.base_lbl))
        hom_initial_state = full_initial_state[1:-1]
        u0 = u(time=0, weights=hom_initial_state, weight_lbl=self.base_lbl)
        bar_initial_state = (self.a_tilde @ hom_initial_state
                             - (self.b1 * u0).flatten())
        return bar_initial_state

    def transform_feedback(self, k_src, src_base):
        """ Transform the given feedback to work with the simulated system """
        fem_base = pi.get_base(self.base_lbl)
        t_fem_mod = pi.calculate_base_transformation_matrix(fem_base,
                                                            src_base)
        # in the following, we set \tilde A = I
        k_fem = k_src @ t_fem_mod
        k_inhom_0 = k_fem[:, 0]
        k_hom = k_fem[:, 1:-1]
        k_inhom_1 = k_fem[:, -1]
        k_sim = - (k_hom + k_inhom_0 * 0) / (k_hom @ self.b1
                                             + k_inhom_1 - 1)
        return k_sim

    def get_results(self, weights, u, temp_dom, spat_dom, name=None):
        # back transformation
        u_vec = u.get_results(temp_dom)
        # if u_vec.dim == 1:
        #     u_vec = u_vec[:, None]
        orig_weights = weights @ self.a_tilde_inv + u_vec @ self.b1.T

        # add missing values from dirichlet bc
        all_weights = np.hstack((np.zeros_like(u_vec), orig_weights, u_vec))

        # evaluate
        ed = pi.evaluate_approximation("fem_base",
                                       all_weights,
                                       temp_dom,
                                       spat_dom,
                                       name="x(z,t)" + name)
        return ed