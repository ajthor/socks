from functools import partial

from gym_socks.algorithms.algorithm import AlgorithmInterface

import gym_socks.kernel.metrics as kernel

from gym_socks.utils import normalize, indicator_fn, generate_batches

import numpy as np


class KernelSR_RFF(AlgorithmInterface):
    """
    Stochastic reachability using kernel distribution embeddings (RFF).

    References
    ----------
    .. [1] `Approximate Stochastic Reachability for High Dimensional Systems, 2021
           Adam J. Thorpe, Vignesh Sivaramakrishnan, Meeko M. K. Oishi
           IEEE American Control Conference,
           <https://arxiv.org/abs/1910.10818>`_
    """

    def __init__(self, kernel_fn=None, l=None, *args, **kwargs):
        """
        Initialize the algorithm.
        """
        super().__init__(*args, **kwargs)

        if kernel_fn is None:
            kernel_fn = partial(kernel.rbf_kernel, sigma=0.1)

        if l is None:
            l = 1

        self.kernel_fn = kernel_fn
        self.l = l

    def _validate_inputs(
        self,
        system=None,
        S: "State sample." = None,
        T: "Test points." = None,
        constraint_tube=None,
        target_tube=None,
        problem: "Stochastic reachability problem." = "THT",
    ):

        if system is None:
            raise ValueError("Must supply a system.")

        if S is None:
            raise ValueError("Must supply a sample.")

        if T is None:
            raise ValueError("Must supply test points.")

        if constraint_tube is None:
            raise ValueError("Must supply constraint tube.")

        if target_tube is None:
            raise ValueError("Must supply target tube.")

        if problem != "THT" and problem != "FHT":
            raise ValueError("problem is not in {'THT', 'FHT'}")

    def run(
        self,
        system=None,
        S: "State sample." = None,
        T: "Test points." = None,
        constraint_tube=None,
        target_tube=None,
        problem: "Stochastic reachability problem." = "THT",
    ):
        """
        Run the algorithm.
        """

        self._validate_inputs(
            system=system,
            S=S,
            T=T,
            constraint_tube=constraint_tube,
            target_tube=target_tube,
            problem=problem,
        )

        kernel_fn = self.kernel_fn
        l = self.l

        T = np.array(T)
        num_time_steps = system.num_time_steps - 1

        S = np.array(S)
        X = S[:, 0, :]
        Y = S[:, 1, :]

        W = kernel.regularized_inverse(X, kernel_fn=kernel_fn, l=l)

        CXY = kernel_fn(X, Y)

        betaXY = normalize(np.einsum("ii,ij->ij", W, CXY))

        # set up empty array to hold value functions
        Vt = np.zeros((num_time_steps + 1, len(X)))

        Vt[num_time_steps, :] = indicator_fn(Y, constraint_tube[num_time_steps])

        # run backwards in time and compute the safety probabilities
        for t in range(num_time_steps - 1, -1, -1):

            # print(f"Computing for k={t}")

            VX = np.einsum("i,ij->j", Vt[t + 1, :], betaXY)

            Y_in_safe_set = indicator_fn(Y, constraint_tube[t])

            if problem == "THT":

                Vt[t, :] = Y_in_safe_set * VX

            elif problem == "FHT":

                Y_in_target_set = indicator_fn(Y, target_tube[t])

                Vt[t, :] = Y_in_target_set + (Y_in_safe_set & ~Y_in_target_set) * VX

        # set up empty array to hold safety probabilities
        Pr = np.zeros((num_time_steps + 1, len(T)))

        CXT = kernel_fn(X, T)
        betaXT = normalize(np.einsum("ii,ij->ij", W, CXT))

        Pr[num_time_steps, :] = indicator_fn(T, constraint_tube[num_time_steps])

        # run backwards in time and compute the safety probabilities
        for t in range(num_time_steps - 1, -1, -1):

            # print(f"Computing for k={t}")

            VT = np.einsum("i,ij->j", Vt[t + 1, :], betaXT)

            T_in_safe_set = indicator_fn(T, constraint_tube[t])

            if problem == "THT":

                Pr[t, :] = T_in_safe_set * VT

            elif problem == "FHT":

                T_in_target_set = indicator_fn(T, target_tube[t])

                Pr[t, :] = T_in_target_set + (T_in_safe_set & ~T_in_target_set) * VT

        return Pr, Vt

    def run_batch(
        self,
        system=None,
        S: "State sample." = None,
        T: "Test points." = None,
        constraint_tube=None,
        target_tube=None,
        problem: "Stochastic reachability problem." = "THT",
        batch_size: "Batch size." = 100,
    ):
        """
        Run the algorithm (batch mode).
        """

        self._validate_inputs(
            system=system,
            S=S,
            T=T,
            constraint_tube=constraint_tube,
            target_tube=target_tube,
            problem=problem,
        )

        kernel_fn = self.kernel_fn
        l = self.l

        T = np.array(T)
        num_time_steps = system.num_time_steps - 1

        S = np.array(S)
        X = S[:, 0, :]
        Y = S[:, 1, :]

        W = kernel.regularized_inverse(X, kernel_fn=kernel_fn, l=l)

        # set up empty array to hold value functions
        Vt = np.empty((num_time_steps + 1, len(X)))

        # run backwards in time and compute the safety probabilities
        for t in range(num_time_steps - 1, -1, -1):

            # print(f"Computing for k={t}")

            for batch in generate_batches(num_elements=len(X), batch_size=batch_size):

                CXY = kernel_fn(X, Y[batch])
                betaXY = normalize(np.einsum("ii,ij->ij", W, CXY))

                Vt[num_time_steps, batch] = indicator_fn(
                    Y[batch], constraint_tube[num_time_steps]
                )

                VX = np.einsum("i,ij->j", Vt[t + 1, :], betaXY)

                Y_in_safe_set = indicator_fn(Y[batch], constraint_tube[t])

                if problem == "THT":

                    Vt[t, batch] = Y_in_safe_set * VX

                elif problem == "FHT":

                    Y_in_target_set = indicator_fn(Y[batch], target_tube[t])

                    Vt[t, batch] = (
                        Y_in_target_set + (Y_in_safe_set & ~Y_in_target_set) * VX
                    )

        # set up empty array to hold safety probabilities
        Pr = np.empty((num_time_steps + 1, len(T)))

        # run backwards in time and compute the safety probabilities
        for t in range(num_time_steps - 1, -1, -1):

            # print(f"Computing for k={t}")

            for batch in generate_batches(num_elements=len(T), batch_size=batch_size):

                CXT = kernel_fn(X, T[batch])
                betaXT = normalize(np.einsum("ii,ij->ij", W, CXT))

                Pr[num_time_steps, batch] = indicator_fn(
                    T[batch], constraint_tube[num_time_steps]
                )

                VT = np.einsum("i,ij->j", Vt[t + 1, :], betaXT)

                T_in_safe_set = indicator_fn(T[batch], constraint_tube[t])

                if problem == "THT":

                    Pr[t, batch] = T_in_safe_set * VT

                elif problem == "FHT":

                    T_in_target_set = indicator_fn(T[batch], target_tube[t])

                    Pr[t, batch] = (
                        T_in_target_set + (T_in_safe_set & ~T_in_target_set) * VT
                    )

        return Pr, Vt
