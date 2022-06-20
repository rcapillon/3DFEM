import numpy as np
import copy
from scipy.sparse.linalg import spsolve
from solvers.solver import Solver


class NonlinearStaticsArcLengthSolver(Solver):
    def __init__(self, structure, neumann_BC):
        super(NonlinearStaticsArcLengthSolver, self).__init__(structure, neumann_BC=neumann_BC)

        self.y_axis = None

        self.mat_U = None

        self.mat_U_observed = None

    def run(self, Delta_L=1e-2, psi=1, n_arclengths=10, n_iter_max=10, tol=1e-3,
            corrections="cylindrical", corrector_root_selection="automatic", attenuation=0.5, n_selection=3,
            n_restart=10, n_grow=10, n_switch=4, verbose=True):
        # corrections              : can be set to either "cylindrical" or "spherical"
        # corrector_root_selection : can be set to "automatic", "forward" or "default". Anything else is interpreted
        #                            as "default"

        self.y_axis = np.zeros((n_arclengths + 1,))

        self.mat_U = np.zeros((self.n_total_dofs, n_arclengths + 1))

        if verbose:
            print("Computing forces...")

        self.neumann_BC.compute_F0(self.n_total_dofs)
        vec_F_f = self.neumann_BC.vec_F0[self.free_dofs]

        if verbose:
            print("Starting arc-length resolution...")

        vec_U = np.zeros((self.n_total_dofs,))
        old_vec_Delta_U = np.zeros((self.n_free_dofs,))
        lambda_factor = 0
        old_Delta_L = Delta_L

        counter_arclength = 0
        S = 1
        counter_signswitch = 0
        counter_cut = 0
        counter_grow = 0

        if corrector_root_selection == "automatic" or corrector_root_selection == "forward":
            forward_centering = True
        else:
            forward_centering = False

        counter_selection = 0

        self.structure.compute_factorized_KT_Ktot_vectors(vec_U)
        self.structure.compute_factorized_KT_Fint(vec_U)
        self.structure.apply_dirichlet_KT_Fint()

        while counter_arclength < n_arclengths:
            if verbose:
                print("\nArc-length n°", counter_arclength + 1, " , Delta_L =", Delta_L)

            if corrector_root_selection == "automatic":
                if counter_selection >= n_selection:
                    forward_centering = False
                    # root selection: lowest error
                else:
                    forward_centering = True
                    # root selection: forward centering

            k0 = psi ** 2 * np.linalg.norm(vec_F_f) ** 2

            vec_delta_Uf = spsolve(self.structure.mat_KTff, vec_F_f)

            Delta_lambda = Delta_L / np.sqrt(np.dot(vec_delta_Uf, vec_delta_Uf) + k0)

            a0 = np.dot(old_vec_Delta_U, vec_delta_Uf) + k0 * Delta_lambda
            if a0 < 0:
                Delta_lambda = -Delta_lambda
                if S == 1:
                    counter_signswitch += 1
                else:
                    counter_signswitch = 0
                S = -1
            else:
                if S == -1:
                    counter_signswitch += 1
                else:
                    counter_signswitch = 0
                S = 1

            vec_Delta_U = Delta_lambda * vec_delta_Uf

            vec_U[self.free_dofs] += vec_Delta_U
            lambda_factor += Delta_lambda

            if forward_centering:
                Delta_L_corrections = 0.5 * Delta_L
            else:
                Delta_L_corrections = Delta_L

            if corrections == "cylindrical":
                k1 = 0
                Delta_L_corrections = Delta_L_corrections * np.sqrt(
                    np.dot(vec_delta_Uf, vec_delta_Uf) / (np.dot(vec_delta_Uf, vec_delta_Uf) + k0))
            else:
                k1 = k0
                Delta_L_corrections = Delta_L

            counter_iter = 0

            if counter_signswitch >= n_switch:
                error = tol + 1
                counter_iter = n_iter_max
                counter_signswitch = 0
            else:
                self.structure.compute_factorized_KT_Ktot_vectors(vec_U)
                self.structure.compute_factorized_KT_Fint(vec_U)
                self.structure.apply_dirichlet_KT_Fint()
                error = np.linalg.norm(lambda_factor * vec_F_f - self.structure.vec_Fint_f) \
                        / np.linalg.norm(lambda_factor * vec_F_f)

            if verbose:
                print("Predictor error:", error)

            if np.isnan(error):
                counter_iter = n_iter_max

            error = tol + 1

            while counter_iter < n_iter_max and error > tol:
                if verbose:
                    print("Iteration", counter_iter + 1)

                vec_delta_Uf = spsolve(self.structure.mat_KTff, vec_F_f)
                vec_delta_Un = spsolve(self.structure.mat_KTff, lambda_factor * vec_F_f - self.structure.vec_Fint_f)

                if forward_centering:
                    a_iter = np.dot(vec_delta_Uf, vec_delta_Uf) + k1
                    b_iter = 2 * np.dot(vec_delta_Uf, 0.5 * vec_Delta_U + vec_delta_Un) + Delta_lambda * k1
                    c_iter = np.dot(0.5 * vec_Delta_U + vec_delta_Un, 0.5 * vec_Delta_U + vec_delta_Un) \
                             + (0.5 * Delta_lambda) ** 2 * k1 - Delta_L_corrections ** 2

                else:
                    a_iter = np.dot(vec_delta_Uf, vec_delta_Uf) + k1
                    b_iter = 2 * np.dot(vec_delta_Uf, vec_Delta_U + vec_delta_Un) + 2 * Delta_lambda * k1
                    c_iter = np.dot(vec_Delta_U + vec_delta_Un, vec_Delta_U + vec_delta_Un) \
                             + Delta_lambda ** 2 * k1 - Delta_L_corrections ** 2

                d_iter = np.dot(vec_Delta_U, vec_delta_Uf)
                discriminant = b_iter ** 2 - 4 * a_iter * c_iter

                if discriminant > 0:
                    delta_lambda_root1 = -(b_iter + np.sqrt(discriminant)) / (2 * a_iter)
                    delta_lambda_root2 = -(b_iter - np.sqrt(discriminant)) / (2 * a_iter)

                    if forward_centering:
                        vec_Delta_U_root1 = vec_Delta_U + vec_delta_Un + delta_lambda_root1 * vec_delta_Uf
                        vec_Delta_U_root2 = vec_Delta_U + vec_delta_Un + delta_lambda_root2 * vec_delta_Uf
                        Delta_lambda_root1 = Delta_lambda + delta_lambda_root1
                        Delta_lambda_root2 = Delta_lambda + delta_lambda_root2

                        arc_length_root1 = np.dot(vec_Delta_U_root1, vec_Delta_U_root1) + k0 * Delta_lambda_root1 ** 2
                        arc_length_root2 = np.dot(vec_Delta_U_root2, vec_Delta_U_root2) + k0 * Delta_lambda_root2 ** 2

                        if arc_length_root1 >= arc_length_root2:
                            delta_lambda = delta_lambda_root1
                        else:
                            delta_lambda = delta_lambda_root2

                    else:
                        delta_lambda = delta_lambda_root1
                        if (d_iter * delta_lambda_root2) > (d_iter * delta_lambda_root1):
                            delta_lambda = delta_lambda_root2

                    vec_Delta_U += vec_delta_Un + delta_lambda * vec_delta_Uf
                    vec_U[self.free_dofs] += vec_delta_Un + delta_lambda * vec_delta_Uf
                    Delta_lambda += delta_lambda
                    lambda_factor += delta_lambda

                    self.structure.compute_factorized_KT_Ktot_vectors(vec_U)
                    self.structure.compute_factorized_KT_Fint(vec_U)
                    self.structure.apply_dirichlet_KT_Fint()

                    error = np.linalg.norm(lambda_factor * vec_F_f - self.structure.vec_Fint_f) \
                            / np.linalg.norm(lambda_factor * vec_F_f)

                    if verbose:
                        print("Corrector error:", error)

                elif discriminant == 0:
                    delta_lambda = -b_iter / (2 * a_iter)

                    vec_Delta_U += vec_delta_Un + delta_lambda * vec_delta_Uf
                    vec_U[self.free_dofs] += vec_delta_Un + delta_lambda * vec_delta_Uf
                    Delta_lambda += delta_lambda
                    lambda_factor += delta_lambda

                    self.structure.compute_factorized_KT_Ktot_vectors(vec_U)
                    self.structure.compute_factorized_KT_Fint(vec_U)
                    self.structure.apply_dirichlet_KT_Fint()

                    error = np.linalg.norm(lambda_factor * vec_F_f - self.structure.vec_Fint_f) \
                            / np.linalg.norm(lambda_factor * vec_F_f)

                    if verbose:
                        print("Corrector error:", error)

                else:
                    a0_partial = a_iter
                    b0_partial = 2 * np.dot(vec_delta_Uf, vec_Delta_U) + 2 * Delta_lambda * k1
                    b1_partial = 2 * np.dot(vec_delta_Uf, vec_delta_Un)
                    c0_partial = np.dot(vec_Delta_U, vec_Delta_U) + Delta_lambda ** 2 * k1 - Delta_L_corrections ** 2
                    c1_partial = 2 * np.dot(vec_delta_Un, vec_Delta_U)
                    c2_partial = np.dot(vec_delta_Un, vec_delta_Un)

                    a_s = b1_partial ** 2 - 4 * a0_partial * c2_partial
                    b_s = 2 * b0_partial * b1_partial - 4 * a0_partial * c1_partial
                    c_s = b0_partial ** 2 - 4 * a0_partial * c0_partial

                    discriminant_s = b_s ** 2 - 4 * a_s * c_s

                    delta_s_max = (-b_s - np.sqrt(discriminant_s)) / (2 * a_s)

                    if delta_s_max <= 0 or delta_s_max > 1:
                        counter_iter = n_iter_max
                        print("Complex load increment cannot be resolved, restarting with lower arc-length.")
                    else:
                        delta_s = delta_s_max

                        a_iter_s = a0_partial
                        b_iter_s = b0_partial + b1_partial * delta_s
                        c_iter_s = c0_partial + c1_partial * delta_s + c2_partial * delta_s ** 2

                        discriminant_iter_s = b_iter_s ** 2 - 4 * a_iter_s * c_iter_s

                        if discriminant_iter_s < 0:
                            print("Complex load increment cannot be resolved, restarting with lower arc-length.")
                            error = tol + 1
                            counter_iter = n_iter_max
                        else:
                            if discriminant_iter_s < 1e-10:
                                delta_lambda = -b_iter_s / (2 * a_iter_s)
                            else:
                                delta_lambda_root1 = -(b_iter_s + np.sqrt(discriminant_iter_s)) / (2 * a_iter_s)
                                delta_lambda_root2 = -(b_iter_s - np.sqrt(discriminant_iter_s)) / (2 * a_iter_s)

                                delta_lambda = delta_lambda_root1

                                if (d_iter * delta_lambda_root2) > (d_iter * delta_lambda_root1):
                                    delta_lambda = delta_lambda_root2

                            vec_Delta_U += delta_s * vec_delta_Un + delta_lambda * vec_delta_Uf
                            vec_U[self.free_dofs] += delta_s * vec_delta_Un + delta_lambda * vec_delta_Uf
                            Delta_lambda += delta_lambda
                            lambda_factor += delta_lambda

                            self.structure.compute_factorized_KT_Ktot_vectors(vec_U)
                            self.structure.compute_factorized_KT_Fint(vec_U)
                            self.structure.apply_dirichlet_KT_Fint()

                            error = np.linalg.norm(lambda_factor * vec_F_f - self.structure.vec_Fint_f) \
                                    / np.linalg.norm(lambda_factor * vec_F_f)

                            if verbose:
                                print("Corrector error:", error)

                if error != error:
                    counter_iter = n_iter_max
                    error = tol + 1

                counter_iter += 1

            if error <= tol:
                if verbose:
                    print("Increment converged.")
                old_vec_Delta_U = vec_Delta_U
                counter_grow += 1
                if counter_grow >= n_grow:
                    counter_grow = 0
                    if counter_cut > 0:
                        counter_cut -= 1
                    if counter_selection > 0:
                        counter_selection -= 1
                    if Delta_L < old_Delta_L:
                        Delta_L /= attenuation
                    elif Delta_L > old_Delta_L:
                        Delta_L *= attenuation
                counter_arclength += 1
                self.mat_U[self.free_dofs, counter_arclength] = vec_U[self.free_dofs]
                self.y_axis[counter_arclength] = lambda_factor
            else:
                vec_U = copy.deepcopy(self.mat_U[:, counter_arclength])
                lambda_factor = copy.deepcopy(self.y_axis[counter_arclength])
                Delta_L *= attenuation
                if counter_cut >= n_restart:
                    print("Arc-length method failed to progress at arc-length n°", counter_arclength + 1, ".")
                    self.mat_U[self.free_dofs, (counter_arclength + 1):] = np.tile(
                                                                    np.array([vec_U[self.free_dofs]]).transpose(),
                                                                    (1, n_arclengths - counter_arclength))
                    self.y_axis[(counter_arclength + 1):] = lambda_factor
                    counter_arclength = n_arclengths
                counter_grow = 0
                counter_cut += 1
                counter_selection += 1
                self.structure.compute_factorized_KT_Ktot_vectors(vec_U)
                self.structure.compute_factorized_KT_Fint(vec_U)
                self.structure.apply_dirichlet_KT_Fint()

        self.mat_U_observed = self.mat_U[self.observed_dofs, :]
