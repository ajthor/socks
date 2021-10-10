from functools import partial

import gym_socks
from gym_socks.algorithms.algorithm import AlgorithmInterface

import gym_socks.kernel.metrics as kernel

from gym_socks.utils import normalize, indicator_fn, generate_batches
from gym_socks.utils.logging import ms_tqdm, _progress_fmt

import numpy as np


class KernelSR(AlgorithmInterface):
    """
    Stochastic reachability using kernel distribution embeddings.

    References
    ----------
    .. [1] `Model-Free Stochastic Reachability
            Using Kernel Distribution Embeddings, 2019
           Adam J. Thorpe, Meeko M. K. Oishi
           IEEE Control Systems Letters,
           <https://arxiv.org/abs/1908.00697>`_
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

        pbar = ms_tqdm(total=num_time_steps * 2 + 3, bar_format=_progress_fmt)

        X, U, Y = gym_socks.envs.sample.transpose_sample(S)
        X = np.array(X)
        U = np.array(U)
        Y = np.array(Y)

        W = kernel.regularized_inverse(X, kernel_fn=kernel_fn, l=l)
        pbar.update()

        CXY = kernel_fn(X, Y)
        betaXY = normalize(np.einsum("ii,ij->ij", W, CXY))
        pbar.update()

        # set up empty array to hold value functions
        Vt = np.zeros((num_time_steps + 1, len(X)))

        Vt[num_time_steps, :] = indicator_fn(Y, target_tube[num_time_steps])

        # run backwards in time and compute the safety probabilities
        for t in range(num_time_steps - 1, -1, -1):

            VX = np.einsum("i,ij->j", Vt[t + 1, :], betaXY)

            Y_in_safe_set = indicator_fn(Y, constraint_tube[t])

            if problem == "THT":

                Vt[t, :] = Y_in_safe_set * VX

            elif problem == "FHT":

                Y_in_target_set = indicator_fn(Y, target_tube[t])

                Vt[t, :] = Y_in_target_set + (Y_in_safe_set & ~Y_in_target_set) * VX

            pbar.update()

        # set up empty array to hold safety probabilities
        Pr = np.zeros((num_time_steps + 1, len(T)))

        CXT = kernel_fn(X, T)
        betaXT = normalize(np.einsum("ii,ij->ij", W, CXT))

        pbar.update()

        Pr[num_time_steps, :] = indicator_fn(T, target_tube[num_time_steps])

        # run backwards in time and compute the safety probabilities
        for t in range(num_time_steps - 1, -1, -1):

            VT = np.einsum("i,ij->j", Vt[t + 1, :], betaXT)

            T_in_safe_set = indicator_fn(T, constraint_tube[t])

            if problem == "THT":

                Pr[t, :] = T_in_safe_set * VT

            elif problem == "FHT":

                T_in_target_set = indicator_fn(T, target_tube[t])

                Pr[t, :] = T_in_target_set + (T_in_safe_set & ~T_in_target_set) * VT

            pbar.update()

        pbar.close()

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

        X, U, Y = gym_socks.envs.sample.transpose_sample(S)
        X = np.array(X)
        U = np.array(U)
        Y = np.array(Y)

        W = kernel.regularized_inverse(X, kernel_fn=kernel_fn, l=l)

        CXY = kernel_fn(X, Y)

        betaXY = normalize(np.einsum("ii,ij->ij", W, CXY))

        # set up empty array to hold value functions
        Vt = np.empty((num_time_steps + 1, len(X)))

        # run backwards in time and compute the safety probabilities
        for t in range(num_time_steps - 1, -1, -1):

            # print(f"Computing for k={t}")

            for batch in generate_batches(num_elements=len(X), batch_size=batch_size):

                CXY = kernel_fn(X, Y[batch])
                betaXY = normalize(np.einsum("ii,ij->ij", W, CXY))

                Vt[num_time_steps, batch] = indicator_fn(
                    Y[batch], target_tube[num_time_steps]
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
                    T[batch], target_tube[num_time_steps]
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


class KernelMaximalSR(AlgorithmInterface):
    """
    Stochastic reachability using kernel distribution embeddings.

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
        A: "Admissible action sample." = None,
        T: "Test points." = None,
        constraint_tube=None,
        target_tube=None,
        problem: "Stochastic reachability problem." = "THT",
    ):

        if system is None:
            raise ValueError("Must supply a system.")

        if S is None:
            raise ValueError("Must supply a sample.")

        if A is None:
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
        U: "Action sample." = None,
        A: "Admissible action sample." = None,
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
            A=A,
            T=T,
            constraint_tube=constraint_tube,
            target_tube=target_tube,
            problem=problem,
        )

        kernel_fn = self.kernel_fn
        l = self.l

        T = np.array(T)
        num_time_steps = system.num_time_steps - 1

        pbar = ms_tqdm(total=num_time_steps * 2 + 3, bar_format=_progress_fmt)

        X, U, Y = gym_socks.envs.sample.transpose_sample(S)
        X = np.array(X)
        U = np.array(U)
        Y = np.array(Y)

        A = np.array(A)

        W = kernel.regularized_inverse(X, U=U, kernel_fn=kernel_fn, l=l)
        pbar.update()

        CXY = kernel_fn(X, Y)

        CUA = kernel_fn(U, A)

        CXT = kernel_fn(X, T)

        betaXY = normalize(
            np.einsum("ii,ij,ik->ijk", W, CXY, CUA, optimize=["einsum_path", (0, 1, 2)])
        )
        pbar.update()

        betaXT = normalize(
            np.einsum("ii,ij,ik->ijk", W, CXT, CUA, optimize=["einsum_path", (0, 1, 2)])
        )
        pbar.update()

        # set up empty array to hold value functions
        Vt = np.zeros((num_time_steps + 1, len(X)), dtype=np.float32)

        Vt[num_time_steps, :] = indicator_fn(Y, target_tube[num_time_steps])

        # run backwards in time and compute the safety probabilities
        for t in range(num_time_steps - 1, -1, -1):

            # print(f"Computing for k={t}")

            # use optimized multiplications
            wX = np.einsum(
                "i,ijk->jk", Vt[t + 1, :], betaXY, optimize=["einsum_path", (0, 1)]
            )

            VX = np.max(wX, axis=1)

            Y_in_safe_set = indicator_fn(Y, constraint_tube[t])

            if problem == "THT":

                Vt[t, :] = Y_in_safe_set * VX

            elif problem == "FHT":

                Y_in_target_set = indicator_fn(Y, target_tube[t])

                Vt[t, :] = Y_in_target_set + (Y_in_safe_set & ~Y_in_target_set) * VX

            pbar.update()

        # set up empty array to hold safety probabilities
        Pr = np.zeros((num_time_steps + 1, len(T)), dtype=np.float32)

        Pr[num_time_steps, :] = indicator_fn(T, target_tube[num_time_steps])

        # run backwards in time and compute the safety probabilities
        for t in range(num_time_steps - 1, -1, -1):

            # use optimized multiplications
            wT = np.einsum(
                "i,ijk->jk", Vt[t + 1, :], betaXT, optimize=["einsum_path", (0, 1)]
            )

            VT = np.max(wT, axis=1)

            T_in_safe_set = indicator_fn(T, constraint_tube[t])

            if problem == "THT":

                Pr[t, :] = T_in_safe_set * VT

            elif problem == "FHT":

                T_in_target_set = indicator_fn(T, target_tube[t])

                Pr[t, :] = T_in_target_set + (T_in_safe_set & ~T_in_target_set) * VT

            pbar.update()

        pbar.close()

        return Pr, Vt

    def run_batch(
        self,
        system=None,
        S: "State sample." = None,
        U: "Action sample." = None,
        A: "Admissible action sample." = None,
        T: "Test points." = None,
        constraint_tube=None,
        target_tube=None,
        problem: "Stochastic reachability problem." = "THT",
        batch_size: "Batch size." = 5,
    ):
        """
        Run the algorithm.
        """

        self._validate_inputs(
            system=system,
            S=S,
            A=A,
            T=T,
            constraint_tube=constraint_tube,
            target_tube=target_tube,
            problem=problem,
        )

        kernel_fn = self.kernel_fn
        l = self.l

        T = np.array(T)
        num_time_steps = system.num_time_steps - 1

        X, U, Y = gym_socks.envs.sample.transpose_sample(S)
        X = np.array(X)
        U = np.array(U)
        Y = np.array(Y)

        A = np.array(A)

        W = kernel.regularized_inverse(X, U=U, kernel_fn=kernel_fn, l=l)

        CUA = kernel_fn(U, A)

        # set up empty array to hold value functions
        Vt = np.empty((num_time_steps + 1, len(X)))

        Vt[num_time_steps, :] = indicator_fn(Y, target_tube[num_time_steps])

        # run backwards in time and compute the safety probabilities
        for t in range(num_time_steps - 1, -1, -1):

            for batch in generate_batches(num_elements=len(X), batch_size=batch_size):

                CXY = kernel_fn(X, Y[batch])
                betaXY = normalize(np.einsum("ii,ij,ik->ijk", W, CXY, CUA))

                wX = np.einsum(
                    "i,ijk->jk", Vt[t + 1, :], betaXY, optimize=["einsum_path", (0, 1)]
                )

                VX = np.max(wX, axis=1)

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

        Pr[num_time_steps, :] = indicator_fn(T, target_tube[num_time_steps])

        # run backwards in time and compute the safety probabilities
        for t in range(num_time_steps - 1, -1, -1):

            for batch in generate_batches(num_elements=len(T), batch_size=batch_size):

                CXT = kernel_fn(X, T[batch])
                betaXT = normalize(np.einsum("ii,ij,ik->ijk", W, CXT, CUA))

                # use optimized multiplications
                wT = np.einsum(
                    "i,ijk->jk", Vt[t + 1, :], betaXT, optimize=["einsum_path", (0, 1)]
                )

                VT = np.max(wT, axis=1)

                T_in_safe_set = indicator_fn(T[batch], constraint_tube[t])

                if problem == "THT":

                    Pr[t, batch] = T_in_safe_set * VT

                elif problem == "FHT":

                    T_in_target_set = indicator_fn(T[batch], target_tube[t])

                    Pr[t, batch] = (
                        T_in_target_set + (T_in_safe_set & ~T_in_target_set) * VT
                    )

        return Pr, Vt
