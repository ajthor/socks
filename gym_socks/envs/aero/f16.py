"""F-16 aircraft dynamics.

The F-16 aircraft is a single-engine, supersonic, multirole fighter aircraft. It is
typically modeled as a rigid body, and the dynamics are governed by Newton's second law.


Based on the book "Aircraft Control and Simulation" by Stevens and Lewis.

Converted from available fortran code. 

.. [1] Stevens, Brian L., Frank L. Lewis, and Eric N. Johnson. Aircraft control and
    simulation: dynamics, controls design, and autonomous systems. John Wiley & Sons,
    2015.

The Morelli model is based on:

.. [2] E. A. Morelli, "Global nonlinear parametric modelling with application to F-16
    aerodynamics," Proceedings of the 1998 American Control Conference. ACC (IEEE
    Cat. No.98CH36207), Philadelphia, PA, USA, 1998, pp. 997-1001 vol.2, doi:
    10.1109/ACC.1998.703559.

"""

import numpy as np

from scipy.integrate import solve_ivp

from gym_socks.envs.spaces import Box
from gym_socks.envs.spaces import Space
from gym_socks.envs.dynamical_system import DynamicalSystem

from gym_socks.policies.policy import BasePolicy


class _BaseF16Env(DynamicalSystem):
    """Base F-16 system.

    Bases: :py:class:`gym_socks.envs.dynamical_system.DynamicalSystem`

    The F-16 aircraft is a single-engine, supersonic, multirole fighter aircraft. It is
    typically modeled as a rigid body, and the dynamics are governed by Newton's second
    law.

    The state is defined as:
        x1: [ft/s] air speed, VT
        x2: [rad] angle of attack, alpha
        x3: [rad] sideslip angle, beta
        x4: [rad] roll angle, phi
        x5: [rad] pitch angle, theta
        x6: [rad] yaw angle, psi
        x7: [rad/s] roll rate, p
        x8: [rad/s] pitch rate, q
        x9: [rad/s] yaw rate, r
        x10: [ft] north position, pn
        x11: [ft] east position, pe
        x12: [ft] altitude, alt
        x13: [-] engine thrust, T

    The action is defined as:
        u1: [-] throttle, thtl (0.0 <= thtl <= 1.0)
        u2: [deg] elevator deflection, el (el >= 0.0)
        u3: [deg] aileron deflection, ail (ail >= 0.0)
        u4: [deg] rudder deflection, rdr (rdr >= 0.0)

    """

    _euler = True

    _sampling_time = 0.01

    # Turn off black formatting for this section.
    # fmt: off
    _damp_a = [
        [ -0.267, -0.110,  0.308,  1.340,  2.080,  2.910,  2.760,  2.050,  1.500,  1.490,  1.830,  1.210],
        [  0.882,  0.852,  0.876,  0.958,  0.962,  0.974,  0.819,  0.483,  0.590,  1.210, -0.493, -1.040],
        [ -0.108, -0.108, -0.188,  0.110,  0.258,  0.226,  0.344,  0.362,  0.611,  0.529,  0.298, -2.270],
        [ -8.800,-25.800,-28.900,-31.400,-31.200,-30.700,-27.700,-28.200,-29.000,-29.800,-38.300,-35.300],
        [ -0.126, -0.026,  0.063,  0.113,  0.208,  0.230,  0.319,  0.437,  0.680,  0.100,  0.447, -0.330],
        [ -0.360, -0.359, -0.443, -0.420, -0.383, -0.375, -0.329, -0.294, -0.230, -0.210, -0.120, -0.100],
        [ -7.210, -0.540, -5.230, -5.260, -6.110, -6.640, -5.690, -6.000, -6.200, -6.400, -6.600, -6.000],
        [ -0.380, -0.363, -0.378, -0.386, -0.370, -0.453, -0.550, -0.582, -0.595, -0.637, -1.020, -0.840],
        [  0.061,  0.052,  0.052, -0.012, -0.013, -0.024,  0.050,  0.150,  0.130,  0.158,  0.240,  0.150],
    ]

    _thrust_a = [
        [  1060,    670,    880,   1140,   1500,   1860],
        [   635,    425,    690,   1010,   1330,   1700],
        [    60,     25,    345,    755,   1130,   1525],
        [ -1020,   -170,   -300,    350,    910,   1360],
        [ -2700,  -1900,  -1300,   -247,    600,   1100],
        [ -3600,  -1400,   -595,   -342,   -200,    700],
    ]
    
    _thrust_b = [
        [ 12680,   9150,   6200,   3950,   2450,   1400],
        [ 12680,   9150,   6313,   4040,   2470,   1400],
        [ 12610,   9312,   6610,   4290,   2600,   1560],
        [ 12640,   9839,   7090,   4660,   2840,   1660],
        [ 12390,  10176,   7750,   5320,   3250,   1930],
        [ 11680,   9848,   8050,   6100,   3800,   2310],
    ]
    
    _thrust_c = [
        [ 20000,  15000,  10800,   7000,   4000,   2500],
        [ 21420,  15700,  11225,   7323,   4435,   2600],
        [ 22700,  16860,  12250,   8154,   5000,   2835],
        [ 24240,  18910,  13760,   9285,   5700,   3215],
        [ 26070,  21075,  15975,  11115,   6860,   3950],
        [ 28886,  23319,  18300,  13484,   8642,   5057],
    ]
    # fmt: on

    def __init__(self, seed=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(13,), dtype=float)
        self.state_space = Box(low=-np.inf, high=np.inf, shape=(13,), dtype=float)
        self.action_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=float)

        # Parameters.
        self._xcg = 0.35

        self._axx = 9496.0
        self._ayy = 55814.0
        self._azz = 63100.0
        self._axz = 982.0
        self._axzs = self._axz**2
        self._xpq = self._axz * (self._axx - self._ayy + self._azz)
        self._gam = self._axx * self._azz - self._axz**2
        self._xqr = self._azz * (self._azz - self._ayy) + self._axzs
        self._zpq = (self._axx - self._ayy) * self._axx + self._axzs
        self._ypr = self._azz - self._axx
        self._weight = 20500.0
        self._g = 32.17
        self._mass = self._weight / self._g
        self._s = 300
        self._b = 30
        self._cbar = 11.32
        self._xcgr = 0.35
        self._hx = 160.0
        self._rtod = 57.29578

        # Control limits.
        self._u_bounds = np.array(
            [[0.0, 1.0], [-25.0, 25.0], [-21.5, 21.5], [-30.0, 30.0]]
        )

        self.state = None

        self.seed(seed=seed)

    def step(self, action, time=0):
        action = np.asarray(action, dtype=float)

        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        # Clip the control inputs.
        action = self._clip_control(action)

        disturbance = self.generate_disturbance(time, self.state, action)

        # solve the initial value problem
        if self._euler is True:
            next_state = self.state + self.sampling_time * self.dynamics(
                time, self.state, action, disturbance
            )
            self.state = next_state
        else:
            # solve the initial value problem
            sol = solve_ivp(
                self.dynamics,
                [time, time + self.sampling_time],
                self.state,
                args=(
                    action,
                    disturbance,
                ),
            )
            *_, self.state = sol.y.T

        # Correct altitude if it goes below zero.
        if self.state[11] < 0:
            self.state[11] = 0.0

        observation = self.generate_observation(time, self.state, action)

        cost = self.cost(time, self.state, action)

        done = False
        info = self._info

        return observation, cost, done, info

    def _damp(self, alpha):
        """Compute the damping coefficients.

        Args:
            alpha: Angle of attack [rad].

        Returns:
            Damping coefficients.
                D = [CXq, CYr, CYp, CZq, Clr, Clp, Cmq, Cnr, Cnp]

        """

        S = 0.2 * alpha
        K = int(np.fix(S))
        if K <= -2:
            K = -1
        if K >= 9:
            K = 8
        DA = S - K
        L = K + int(np.fix(np.copysign(1.1, DA)))

        K += 2
        L += 2

        # Return damping coefficients.
        D = np.zeros((9,))
        for i in range(9):
            # V = self._damp_a[K][i] + abs(DA) * (self._damp_a[L][i] - self._damp_a[K][i])
            V = self._damp_a[i][K] + abs(DA) * (self._damp_a[i][L] - self._damp_a[i][K])
            D[i] = V
        return D

    def _adc(self, VT, alt):
        """Compute the air data computer (ADC) outputs.

        Args:
            VT: [ft/s] airspeed
            alt: [ft] altitude

        Returns:
            amach: [-] mach number
            qbar: [psf] dynamic pressure

        """

        R0 = 2.377e-3
        TFAC = 1.0 - 0.703e-5 * alt
        T = 519.0 * TFAC
        if alt >= 35000.0:
            T = 390.0
        RHO = R0 * TFAC**4.14
        AMACH = VT / np.sqrt(1.4 * 1716.3 * T)
        QBAR = 0.5 * RHO * VT**2

        return AMACH, QBAR

    def _rtau(self, dp):
        """Compute the thrust lag reciprocal time constant.

        Args:
            dp: [hp] power difference

        Returns:
            tau: [s] thrust lag time constant

        """

        if dp <= 25.0:
            rtau = 1.0
            # reciprocal time constant
        elif dp >= 50.0:
            rtau = 0.1
        else:
            rtau = 1.9 - 0.036 * dp
        return rtau

    def _pdot(self, P3, P1):
        """Compute the rate of change in engine power level
        using a first order lag as a function of actual power,
        power, and commanded power, cpower.

        Args:
            P3: [%] actual engine power
            P1: [%] commanded engine power

        """

        if P1 >= 50.0:
            if P3 >= 50.0:
                T = 5.0
                P2 = P1
            else:
                P2 = 60.0
                T = self._rtau(P2 - P3)
        else:
            if P3 >= 50.0:
                T = 5.0
                P2 = 40.0
            else:
                P2 = P1
                T = self._rtau(P2 - P3)
        PDOT = T * (P2 - P3)
        return PDOT

    def _thrust(self, power, alt, rmach):
        """Compute the thrust.

        Args:
            power: [%] engine power
            alt: [ft] altitude
            rmach: [-] mach number

        Returns:
            thrust: [lbf] thrust

        """

        # Row index for altitude.
        H = 0.0001 * alt
        I = int(np.fix(H))
        if I >= 5:
            I = 4
        DH = H - I

        # Column index for mach number.
        RM = 5.0 * rmach
        M = int(np.fix(RM))
        if M >= 5:
            M = 4
        if M <= 0:
            M = 0
        DM = RM - M
        CDH = 1.0 - DH

        # Compute mil thrust.
        # S = self._thrust_b[I][M] * CDH + self._thrust_b[I + 1][M] * DH
        # T = self._thrust_b[I][M + 1] * CDH + self._thrust_b[I + 1][M + 1] * DH
        S = self._thrust_b[M][I] * CDH + self._thrust_b[M][I + 1] * DH
        T = self._thrust_b[M + 1][I] * CDH + self._thrust_b[M + 1][I + 1] * DH
        # Mach number interpolation.
        TMIL = S + (T - S) * DM

        # Interpolate with idle or max thrust, depending on power.
        if power <= 50.0:
            # Compute idle thrust.
            # S = self._thrust_a[I][M] * CDH + self._thrust_a[I + 1][M] * DH
            # T = self._thrust_a[I][M + 1] * CDH + self._thrust_a[I + 1][M + 1] * DH
            S = self._thrust_a[M][I] * CDH + self._thrust_a[M][I + 1] * DH
            T = self._thrust_a[M + 1][I] * CDH + self._thrust_a[M + 1][I + 1] * DH
            # Mach number interpolation.
            TIDL = S + (T - S) * DM
            THRUST = TIDL + (TMIL - TIDL) * power / 50.0
        else:
            # Compute max thrust.
            # S = self._thrust_c[I][M] * CDH + self._thrust_c[I + 1][M] * DH
            # T = self._thrust_c[I][M + 1] * CDH + self._thrust_c[I + 1][M + 1] * DH
            S = self._thrust_c[M][I] * CDH + self._thrust_c[M][I + 1] * DH
            T = self._thrust_c[M + 1][I] * CDH + self._thrust_c[M + 1][I + 1] * DH
            # Mach number interpolation.
            TMAX = S + (T - S) * DM
            THRUST = TMIL + (TMAX - TMIL) * (power - 50.0) / 50.0

        return THRUST

    def _tgear(self, thtl):
        """Compute the engine power level command.

        Args:
            thtl: [-] throttle command

        Returns:
            tgear: [-] engine power level command

        """

        if thtl <= 0.77:
            TGEAR = 64.94 * thtl
        else:
            TGEAR = 217.38 * thtl - 117.38
        return TGEAR

    def _components(self, state, action):
        """Component buildup."""

        raise NotImplementedError

    def generate_disturbance(self, time, state, action):
        """Generate the disturbance.

        Args:
            time: Current time.
            state: Current state of the system.
            action: Current action of the system.

        Returns:
            The disturbance.

        """

        # w = self._np_random.standard_normal(size=self.state_space.shape)
        w = self._np_random.normal(
            scale=[1, 0.01, 0.01, 0.1, 0.1, 0.1, 0, 0, 0, 1, 1, 1, 0],
            size=self.state_space.shape,
        )
        return 1e-9 * w

    def _clip_control(self, action):
        """Clip the control inputs.

        Args:
            action: Current action of the system.

        Returns:
            The action with control limits applied.

        """

        action = np.clip(action, self._u_bounds[:, 0], self._u_bounds[:, 1], out=action)
        return action

    def dynamics(self, time, state, action, disturbance):
        """Dynamics of the F-16 aircraft.

        dx/dt = f(x, u, w)

        Args:
            time: Current time.
            state: Current state of the system.
            action: Current action of the system.
            disturbance: Current disturbance of the system.

        Returns:
            The derivative of the state.

        """

        VT, alpha, beta, phi, theta, psi, p, q, r, pn, pe, alt, power = state
        thtl, el, ail, rdr = action

        alpha *= self._rtod
        beta *= self._rtod

        # Compute the air data computer outputs.
        amach, qbar = self._adc(VT, alt)
        cpow = self._tgear(thtl)

        dx13 = self._pdot(power, cpow)
        T = self._thrust(power, alt, amach)

        # Lookup tables and component buildup.
        CXT, CYT, CZT, CLT, CMT, CNT = self._components(
            [VT, alpha, beta, phi, theta, psi, p, q, r, pn, pe, alt, power], action
        )

        # Add damping derivatives.
        CBTA = np.cos(state[2])
        U = VT * np.cos(state[1]) * CBTA
        V = VT * np.sin(state[2])
        W = VT * np.sin(state[1]) * CBTA
        TVT = 0.5 / VT
        B2V = self._b * TVT
        CQ = self._cbar * q * TVT

        D = self._damp(alpha)

        CXT += CQ * D[0]
        CYT += B2V * (D[1] * r + D[2] * p)
        CZT += CQ * D[3]
        CLT += B2V * (D[4] * r + D[5] * p)
        CMT += CQ * D[6] + CZT * (self._xcgr - self._xcg)
        CNT += (
            B2V * (D[7] * r + D[8] * p)
            - CYT * (self._xcgr - self._xcg) * self._cbar / self._b
        )

        STH = np.sin(theta)
        CTH = np.cos(theta)
        SPH = np.sin(phi)
        CPH = np.cos(phi)
        SPSI = np.sin(psi)
        CPSI = np.cos(psi)

        QS = qbar * self._s
        QSB = QS * self._b
        RMQS = QS / self._mass
        GCTH = self._g * CTH
        QSPH = q * SPH
        AY = RMQS * CYT
        AZ = RMQS * CZT

        # Force equations.
        u_dot = r * V - q * W - self._g * STH + (QS * CXT + T) / self._mass
        v_dot = p * W - r * U + GCTH * SPH + AY
        w_dot = q * U - p * V + GCTH * CPH + AZ
        dum = U**2 + W**2

        dx1 = (U * u_dot + V * v_dot + W * w_dot) / VT
        dx2 = (U * w_dot - W * u_dot) / dum
        dx3 = (VT * v_dot - V * dx1) * CBTA / dum

        # Kinematics.
        dx4 = p + (STH / CTH) * (QSPH + r * CPH)
        dx5 = q * CPH - r * SPH
        dx6 = (QSPH + r * CPH) / CTH

        # Moments.
        roll = QSB * CLT
        pitch = QS * self._cbar * CMT
        yaw = QSB * CNT
        PQ = p * q
        QR = q * r
        QHX = q * self._hx

        dx7 = (
            self._xpq * PQ - self._xqr * QR + self._azz * roll + self._axz * (yaw + QHX)
        ) / self._gam
        dx8 = (
            self._ypr * p * r - self._axz * (p**2 - r**2) + pitch - r * self._hx
        ) / self._ayy
        dx9 = (
            self._zpq * PQ - self._xpq * QR + self._axz * roll + self._axx * (yaw + QHX)
        ) / self._gam

        # Navigation.
        T1 = SPH * CPSI
        T2 = CPH * STH
        T3 = SPH * SPSI
        S1 = CTH * CPSI
        S2 = CTH * SPSI
        S3 = T1 * STH - CPH * SPSI
        S4 = T3 * STH + CPH * CPSI
        S5 = SPH * CTH
        S6 = T2 * CPSI + T3
        S7 = T2 * SPSI - T1
        S8 = CPH * CTH

        dx10 = U * S1 + V * S3 + W * S6
        dx11 = U * S2 + V * S4 + W * S7
        dx12 = U * STH - V * S5 - W * S8

        output = np.array(
            [dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8, dx9, dx10, dx11, dx12, dx13],
            dtype=float,
        )

        # Apply disturbances.
        output += disturbance

        # Outputs.
        AN = -AZ / self._g
        ALAT = AY / self._g
        AX = (QS * CXT + T) / self._g

        self._info = {
            "AN": AN,
            "ALAT": ALAT,
            "AX": AX,
            "QBAR": qbar,
            "AMACH": amach,
            "Q": q,
            "ALPHA": alpha,
        }

        return output


class MorelliF16Env(_BaseF16Env):
    """F-16 system.

    Bases: :py:class:`gym_socks.envs.aero._BaseF16Env`

    E. A. Morelli, "Global nonlinear parametric modelling with application to
          F-16 aerodynamics,"
    Proceedings of the 1998 American Control Conference. ACC (IEEE Cat.
    No.98CH36207), Philadelphia, PA, USA, 1998, pp. 997-1001 vol.2, doi:
    10.1109/ACC.1998.703559.

    """

    _available_renderers = ["matplotlib"]

    def _cx(self, alpha, de):
        # a = [
        #     -1.943367e-2,
        #     2.136104e-1,
        #     -2.903457e-1,
        #     -3.348641e-3,
        #     -2.060504e-1,
        #     6.988016e-1,
        #     -9.035381e-1,
        # ]

        return (
            -1.943367e-2
            + 2.136104e-1 * alpha
            - 2.903457e-1 * de**2
            - 3.348641e-3 * de
            - 2.060504e-1 * alpha * de
            + 6.988016e-1 * alpha**2
            - 9.035381e-1 * alpha**3
        )

    def _cxq(self, alpha):
        # b = [
        #     4.833383e-1,
        #     8.644627,
        #     1.131098e1,
        #     -7.422961e1,
        #     6.075776e1,
        # ]

        return (
            4.833383e-1
            + 8.644627 * alpha
            + 1.131098e1 * alpha**2
            - 7.422961e1 * alpha**3
            + 6.075776e1 * alpha**4
        )

    def _cy(self, beta, da, dr):
        # c = [
        #     -1.145916,
        #     6.016057e-2,
        #     1.642479e-1,
        # ]

        return -1.145916 * beta + 6.016057e-2 * da + 1.642479e-1 * dr

    def _cyp(self, alpha):
        # d = [
        #     -1.006733e-1,
        #     8.679799e-1,
        #     4.260586,
        #     -6.923267,
        # ]

        return (
            -1.006733e-1
            + 8.679799e-1 * alpha
            + 4.260586 * alpha**2
            - 6.923267 * alpha**3
        )

    def _cyr(self, alpha):
        # e = [
        #     8.071648e-1,
        #     1.189633e-1,
        #     4.177702,
        #     -9.162236,
        # ]

        return (
            8.071648e-1
            + 1.189633e-1 * alpha
            + 4.177702 * alpha**2
            - 9.162236 * alpha**3
        )

    def _cz(self, alpha, beta, de):
        # f = [
        #     -1.378278e-1,
        #     -4.211369,
        #     4.775187,
        #     -1.026225e1,
        #     8.399763,
        #     -4.354000e-1,
        # ]

        return (
            -1.378278e-1
            - 4.211369 * alpha
            + 4.775187 * alpha**2
            - 1.026225e1 * alpha**3
            + 8.399763 * alpha**4
        ) * (1 - beta**2) - 4.354000e-1 * de

    def _czq(self, alpha):
        # g = [
        #     -3.054956e1,
        #     -4.132305e1,
        #     3.292788e2,
        #     -6.848038e2,
        #     4.080244e2,
        # ]

        return (
            -3.054956e1
            - 4.132305e1 * alpha
            + 3.292788e2 * alpha**2
            - 6.848038e2 * alpha**3
            + 4.080244e2 * alpha**4
        )

    def _cl(self, alpha, beta):
        # h = [
        #     -1.05853e-1,
        #     -5.776677e-1,
        #     -1.672435e-2,
        #     1.357256e-1,
        #     2.172952e-1,
        #     3.464156,
        #     -2.835451,
        #     -1.098104,
        # ]

        return (
            -1.05853e-1 * beta
            - 5.776677e-1 * alpha * beta
            - 1.672435e-2 * alpha**2 * beta
            + 1.357256e-1 * beta**2
            + 2.172952e-1 * alpha * beta**2
            + 3.464156 * alpha**3 * beta
            - 2.835451 * alpha**4 * beta
            - 1.098104 * alpha**2 * beta**2
        )

    def _clp(self, alpha):
        # i = [
        #    -4.126806e-1,
        #    -1.189974e-1,
        #    1.247721,
        #    -7.391132e-1,
        # ]

        return (
            -4.126806e-1
            - 1.189974e-1 * alpha
            + 1.247721 * alpha**2
            - 7.391132e-1 * alpha**3
        )

    def _clr(self, alpha):
        # j = [
        #     6.250437e-2,
        #     6.067723e-1,
        #     -1.101964,
        #     9.100087,
        #     -1.192672e1,
        # ]

        return (
            6.250437e-2
            + 6.067723e-1 * alpha
            - 1.101964 * alpha**2
            + 9.100087 * alpha**3
            - 1.192672e1 * alpha**4
        )

    def _clda(self, alpha, beta):
        # k = [
        #     -1.463144e-1,
        #     -4.07391e-2,
        #     3.253159e-2,
        #     4.851209e-1,
        #     2.978850e-1,
        #     -3.746393e-1,
        #     -3.213068e-1,
        # ]

        return (
            -1.463144e-1
            - 4.07391e-2 * alpha
            + 3.253159e-2 * beta
            + 4.851209e-1 * alpha**2
            + 2.978850e-1 * alpha * beta
            - 3.746393e-1 * alpha**2 * beta
            - 3.213068e-1 * alpha**3
        )

    def _cldr(self, alpha, beta):
        # l = [
        #     2.635729e-2,
        #     -2.192910e-2,
        #     -3.152901e-3,
        #     -5.817803e-2,
        #     4.516159e-1,
        #     -4.928702e-1,
        #     -1.579864e-2,
        # ]

        return (
            2.635729e-2
            - 2.192910e-2 * alpha
            - 3.152901e-3 * beta
            - 5.817803e-2 * alpha * beta
            + 4.516159e-1 * alpha**2 * beta
            - 4.928702e-1 * alpha**3 * beta
            - 1.579864e-2 * beta**2
        )

    def _cm(self, alpha, de):
        # m = [
        #     -2.029370e-2,
        #     4.660702e-2,
        #     -6.012308e-1,
        #     -8.062977e-2,
        #     8.320429e-2,
        #     5.018538e-1,
        #     6.378864e-1,
        #     4.226356e-1,
        # ]

        return (
            -2.029370e-2
            + 4.660702e-2 * alpha
            - 6.012308e-1 * de
            - 8.062977e-2 * alpha * de
            + 8.320429e-2 * de**2
            + 5.018538e-1 * alpha**2 * de
            + 6.378864e-1 * de**3
            + 4.226356e-1 * alpha * de**2
        )

    def _cmq(self, alpha):
        # n = [
        #     -5.19153,
        #     -3.554716,
        #     -3.598636e1,
        #     2.247355e2,
        #     -4.120991e2,
        #     2.411750e2,
        # ]

        return (
            -5.19153
            - 3.554716 * alpha
            - 3.598636e1 * alpha**2
            + 2.247355e2 * alpha**3
            - 4.120991e2 * alpha**4
            + 2.411750e2 * alpha**5
        )

    def _cn(self, alpha, beta):
        # o = [
        #     2.993363e-1,
        #     6.594004e-2,
        #     -2.003125e-1,
        #     -6.233977e-2,
        #     -2.107885,
        #     2.141420,
        #     8.476901e-1,
        # ]

        return (
            2.993363e-1 * beta
            + 6.594004e-2 * alpha * beta
            - 2.003125e-1 * beta**2
            - 6.233977e-2 * alpha * beta**2
            - 2.107885 * alpha**2 * beta
            + 2.141420 * alpha**2 * beta**2
            + 8.476901e-1 * alpha**3 * beta
        )

    def _cnp(self, alpha):
        # p = [
        #     2.677652e-2,
        #     -3.298246e-1,
        #     1.926178e-1,
        #     4.013325,
        #     -4.404302,
        # ]

        return (
            2.677652e-2
            - 3.298246e-1 * alpha
            + 1.926178e-1 * alpha**2
            + 4.013325 * alpha**3
            - 4.404302 * alpha**4
        )

    def _cnr(self, alpha):
        # q = [
        #     -3.698756e-1,
        #     -1.167551e-1,
        #     -7.641297e-1,
        # ]

        return -3.698756e-1 - 1.167551e-1 * alpha - 7.641297e-1 * alpha**2

    def _cnda(self, alpha, beta):
        # r = [
        #     -3.348717e-2,
        #     4.276655e-2,
        #     6.573646e-3,
        #     3.535831e-1,
        #     -1.373308,
        #     1.237582,
        #     2.302543e-1,
        #     -2.512876e-1,
        #     1.588105e-1,
        #     -5.199526e-1,
        # ]

        return (
            -3.348717e-2
            + 4.276655e-2 * alpha
            + 6.573646e-3 * beta
            + 3.535831e-1 * alpha * beta
            - 1.373308 * alpha**2 * beta
            + 1.237582 * alpha**3 * beta
            + 2.302543e-1 * alpha**2
            - 2.512876e-1 * alpha**3
            + 1.588105e-1 * beta**3
            - 5.199526e-1 * alpha * beta**3
        )

    def _cndr(self, alpha, beta):
        # s = [
        #     -8.115894e-2,
        #     -1.156580e-2,
        #     2.514167e-2,
        #     2.038748e-1,
        #     -3.337476e-1,
        #     1.004297e-1,
        # ]

        return (
            -8.115894e-2
            - 1.156580e-2 * alpha
            + 2.514167e-2 * beta
            + 2.038748e-1 * alpha * beta
            - 3.337476e-1 * alpha**2 * beta
            + 1.004297e-1 * alpha**2
        )

    def _components(self, state, action):
        """Polynomial approximation."""

        VT, alpha, beta, phi, theta, psi, p, q, r, pn, pe, alt, power = state
        thtl, el, ail, rdr = action

        # Convert to radians.
        alpha = np.deg2rad(alpha)
        beta = np.deg2rad(beta)

        de = np.deg2rad(el)
        da = np.deg2rad(ail)
        dr = np.deg2rad(rdr)

        p_tilde = p * self._b / (2 * VT)
        q_tilde = q * self._cbar / (2 * VT)
        r_tilde = r * self._b / (2 * VT)

        CX = self._cx(alpha, de)
        CXQ = self._cxq(alpha)
        CY = self._cy(beta, da, dr)
        CYP = self._cyp(alpha)
        CYR = self._cyr(alpha)
        CZ = self._cz(alpha, beta, de)
        CZQ = self._czq(alpha)
        CL = self._cl(alpha, beta)
        CLP = self._clp(alpha)
        CLR = self._clr(alpha)
        CLDA = self._clda(alpha, beta)
        CLDR = self._cldr(alpha, beta)
        CM = self._cm(alpha, de)
        CMQ = self._cmq(alpha)
        CN = self._cn(alpha, beta)
        CNP = self._cnp(alpha)
        CNR = self._cnr(alpha)
        CNDA = self._cnda(alpha, beta)
        CNDR = self._cndr(alpha, beta)

        CXT = CX + CXQ * q_tilde
        CYT = CY + CYP * p_tilde + CYR * r_tilde
        CZT = CZ + CZQ * q_tilde
        CLT = CL + CLP * p_tilde + CLR * r_tilde + CLDA * da + CLDR * dr
        CMT = CM + CMQ * q_tilde + CZT * (self._xcgr - self._xcg)
        CNT = (
            CN
            + CNP * p_tilde
            + CNR * r_tilde
            + CNDA * da
            + CNDR * dr
            - CYT * (self._xcgr - self._xcg) * (self._cbar / self._b)
        )

        return CXT, CYT, CZT, CLT, CMT, CNT


class StevensF16Env(_BaseF16Env):
    """F-16 system.

    Bases: :py:class:`gym_socks.envs.aero._BaseF16Env`

    """

    _available_renderers = ["matplotlib"]

    # Turn off black formatting for this section.
    # fmt: off
    _cl_a = [
        [ 0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000],
        [-0.001, -0.004, -0.008, -0.012, -0.016, -0.022, -0.022, -0.021, -0.015, -0.008, -0.013, -0.015],
        [-0.003, -0.009, -0.017, -0.024, -0.030, -0.041, -0.045, -0.040, -0.016, -0.002, -0.010, -0.019],
        [-0.001, -0.010, -0.020, -0.030, -0.039, -0.054, -0.057, -0.054, -0.023, -0.006, -0.014, -0.027],
        [ 0.000, -0.010, -0.022, -0.034, -0.047, -0.060, -0.069, -0.067, -0.033, -0.036, -0.035, -0.035],
        [ 0.007, -0.010, -0.023, -0.034, -0.049, -0.063, -0.081, -0.079, -0.060, -0.058, -0.062, -0.059],
        [ 0.009, -0.011, -0.023, -0.037, -0.050, -0.068, -0.089, -0.088, -0.091, -0.076, -0.077, -0.076],
    ]

    _cm_a = [
        [ 0.205,  0.168,  0.186,  0.196,  0.213,  0.251,  0.245,  0.238,  0.252,  0.231,  0.198,  0.192],
        [ 0.081,  0.077,  0.107,  0.110,  0.110,  0.141,  0.127,  0.119,  0.133,  0.108,  0.081,  0.093],
        [-0.046, -0.020, -0.009, -0.005, -0.006,  0.010,  0.006, -0.001,  0.014,  0.000, -0.013,  0.032],
        [-0.174, -0.145, -0.121, -0.127, -0.129, -0.102, -0.097, -0.113, -0.087, -0.084, -0.069, -0.006],
        [-0.259, -0.202, -0.184, -0.193, -0.199, -0.150, -0.160, -0.167, -0.104, -0.076, -0.041, -0.005],
    ]

    _cn_a = [
        [ 0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000],
        [ 0.018,  0.019,  0.018,  0.019,  0.019,  0.018,  0.013,  0.007,  0.004, -0.014, -0.017, -0.033],
        [ 0.038,  0.042,  0.042,  0.042,  0.043,  0.039,  0.030,  0.017,  0.004, -0.035, -0.047, -0.057],
        [ 0.056,  0.057,  0.059,  0.058,  0.058,  0.053,  0.032,  0.012,  0.002, -0.046, -0.071, -0.073],
        [ 0.064,  0.077,  0.076,  0.074,  0.073,  0.057,  0.029,  0.007,  0.012, -0.034, -0.065, -0.041],
        [ 0.074,  0.086,  0.093,  0.089,  0.080,  0.062,  0.049,  0.022,  0.028, -0.012, -0.002, -0.013],
        [ 0.079,  0.090,  0.106,  0.106,  0.096,  0.080,  0.068,  0.030,  0.064,  0.015,  0.011, -0.001],
    ]

    _cx_a = [
        [ -0.099, -0.081, -0.081, -0.063, -0.025,  0.044,  0.097,  0.113,  0.145,  0.167,  0.174,  0.166], 
        [ -0.048, -0.038, -0.040, -0.021,  0.016,  0.083,  0.127,  0.137,  0.162,  0.177,  0.179,  0.167], 
        [ -0.022, -0.020, -0.021, -0.004,  0.032,  0.094,  0.128,  0.130,  0.154,  0.161,  0.155,  0.138], 
        [ -0.040, -0.038, -0.039, -0.025,  0.006,  0.062,  0.087,  0.085,  0.100,  0.110,  0.104,  0.091], 
        [ -0.083, -0.073, -0.076, -0.072, -0.046,  0.012,  0.024,  0.025,  0.043,  0.053,  0.047,  0.040],
    ]

    _cz_a = [
        [ 0.770,  0.241, -0.100, -0.416, -0.731, -1.053, -1.366, -1.646, -1.917, -2.120, -2.248, -2.229],
    ]

    _dlda_a = [
        [-0.041, -0.052, -0.053, -0.056, -0.050, -0.056, -0.082, -0.059, -0.042, -0.038, -0.027, -0.017],
        [-0.041, -0.053, -0.053, -0.053, -0.050, -0.051, -0.066, -0.043, -0.038, -0.027, -0.023, -0.016],
        [-0.042, -0.053, -0.052, -0.051, -0.049, -0.049, -0.043, -0.035, -0.026, -0.016, -0.018, -0.014],
        [-0.040, -0.052, -0.051, -0.052, -0.048, -0.048, -0.042, -0.037, -0.031, -0.026, -0.017, -0.012],
        [-0.043, -0.049, -0.048, -0.049, -0.043, -0.042, -0.042, -0.036, -0.025, -0.021, -0.016, -0.011],
        [-0.044, -0.048, -0.048, -0.047, -0.042, -0.041, -0.020, -0.028, -0.013, -0.014, -0.011, -0.010],
        [-0.043, -0.049, -0.047, -0.045, -0.042, -0.037, -0.003, -0.013, -0.010, -0.003, -0.007, -0.008],
    ]

    _dldr_a = [
        [ 0.005,  0.017,  0.014,  0.010, -0.005,  0.009,  0.019,  0.005, -0.000, -0.005, -0.011,  0.008],
        [ 0.007,  0.016,  0.014,  0.014,  0.013,  0.009,  0.012,  0.005,  0.000,  0.004,  0.009,  0.007],
        [ 0.013,  0.013,  0.011,  0.012,  0.011,  0.009,  0.008,  0.005, -0.002,  0.005,  0.003,  0.005],
        [ 0.018,  0.015,  0.015,  0.014,  0.014,  0.014,  0.014,  0.015,  0.013,  0.011,  0.006,  0.001],
        [ 0.015,  0.014,  0.013,  0.013,  0.012,  0.011,  0.011,  0.010,  0.008,  0.008,  0.007,  0.003],
        [ 0.021,  0.011,  0.010,  0.011,  0.010,  0.009,  0.008,  0.010,  0.006,  0.005,  0.000,  0.001],
        [ 0.023,  0.010,  0.011,  0.011,  0.011,  0.010,  0.008,  0.010,  0.006,  0.014,  0.020,  0.000],
    ]

    _dnda_a = [
        [ 0.001, -0.027, -0.017, -0.013, -0.012, -0.016,  0.001,  0.017,  0.011,  0.017,  0.008,  0.016],
        [ 0.002, -0.014, -0.016, -0.016, -0.014, -0.019, -0.021,  0.002,  0.012,  0.016,  0.015,  0.011],
        [-0.006, -0.008, -0.006, -0.006, -0.005, -0.008, -0.005,  0.007,  0.004,  0.007,  0.006,  0.006],
        [-0.011, -0.011, -0.010, -0.009, -0.008, -0.006,  0.000,  0.004,  0.007,  0.010,  0.004,  0.010],
        [-0.015, -0.015, -0.014, -0.012, -0.011, -0.008, -0.002,  0.002,  0.006,  0.012,  0.011,  0.011],
        [-0.024, -0.010, -0.004, -0.002, -0.001,  0.003,  0.014,  0.006, -0.001,  0.004,  0.004,  0.006],
        [-0.022,  0.002, -0.003, -0.005, -0.003, -0.001, -0.009, -0.009, -0.001,  0.003, -0.002,  0.001],
    ]
    
    _dndr_a = [
        [-0.018, -0.052, -0.052, -0.052, -0.054, -0.049, -0.059, -0.051, -0.030, -0.037, -0.026, -0.013],
        [-0.028, -0.051, -0.043, -0.046, -0.045, -0.049, -0.057, -0.052, -0.030, -0.033, -0.030, -0.008],
        [-0.037, -0.041, -0.038, -0.040, -0.040, -0.038, -0.037, -0.030, -0.027, -0.024, -0.019, -0.013],
        [-0.048, -0.045, -0.045, -0.045, -0.044, -0.045, -0.047, -0.048, -0.049, -0.045, -0.033, -0.016],
        [-0.043, -0.044, -0.041, -0.041, -0.040, -0.038, -0.034, -0.035, -0.035, -0.029, -0.022, -0.009],
        [-0.052, -0.034, -0.036, -0.036, -0.035, -0.028, -0.024, -0.023, -0.020, -0.016, -0.010, -0.014],
        [-0.062, -0.034, -0.027, -0.028, -0.027, -0.027, -0.023, -0.023, -0.019, -0.009, -0.025, -0.010],
    ]
    # fmt: on

    def _cl(self, alpha, beta):
        """Compute the X body axis aerodynamic moment coefficient.

        Args:
            alpha: Angle of attack [rad].
            beta: Sideslip angle [rad].

        Returns:
            X body axis aerodynamic moment coefficient.

        """

        S = 0.2 * alpha
        K = int(np.fix(S))
        if K <= -2:
            K = -1
        if K >= 9:
            K = 8
        DA = S - K
        L = K + int(np.fix(np.copysign(1.1, DA)))
        S = 0.2 * np.abs(beta)
        M = int(np.fix(S))
        if M == 0:
            M = 1
        if M >= 6:
            M = 5
        DB = S - M
        N = M + int(np.fix(np.copysign(1.1, DB)))

        K += 2
        L += 2

        # V = self._cl_a[K][M] + abs(DA) * (self._cl_a[L][M] - self._cl_a[K][M])
        # W = self._cl_a[K][N] + abs(DA) * (self._cl_a[L][N] - self._cl_a[K][N])
        V = self._cl_a[M][K] + abs(DA) * (self._cl_a[M][L] - self._cl_a[M][K])
        W = self._cl_a[N][K] + abs(DA) * (self._cl_a[N][L] - self._cl_a[N][K])
        return (V + abs(DB) * (W - V)) * np.sign(beta)

    def _cm(self, alpha, el):
        """Compute the Y body axis aerodynamic moment coefficient.

        Args:
            alpha: Angle of attack [rad].
            el: Elevator deflection [rad].

        Returns:
            Y body axis aerodynamic moment coefficient.

        """

        S = 0.2 * alpha
        K = int(np.fix(S))
        if K <= -2:
            K = -1
        if K >= 9:
            K = 8
        DA = S - K
        L = K + int(np.fix(np.copysign(1.1, DA)))
        S = el / 12.0
        M = int(np.fix(S))
        if M <= -2:
            M = -1
        if M >= 2:
            M = 1
        DE = S - M
        N = M + int(np.fix(np.copysign(1.1, DE)))

        K += 2
        L += 2
        M += 2
        N += 2

        # V = self._cm_a[K][M] + abs(DA) * (self._cm_a[L][M] - self._cm_a[K][M])
        # W = self._cm_a[K][N] + abs(DA) * (self._cm_a[L][N] - self._cm_a[K][N])
        V = self._cm_a[M][K] + abs(DA) * (self._cm_a[M][L] - self._cm_a[M][K])
        W = self._cm_a[N][K] + abs(DA) * (self._cm_a[N][L] - self._cm_a[N][K])
        return V + abs(DE) * (W - V)

    def _cn(self, alpha, beta):
        """Compute the Z body axis aerodynamic moment coefficient.

        Args:
            alpha: Angle of attack [rad].
            beta: Sideslip angle [rad].

        Returns:
            Z body axis aerodynamic moment coefficient.

        """

        S = 0.2 * alpha
        K = int(np.fix(S))
        if K <= -2:
            K = -1
        if K >= 9:
            K = 8
        DA = S - K
        L = K + int(np.fix(np.copysign(1.1, DA)))
        S = 0.2 * np.abs(beta)
        M = int(np.fix(S))
        if M == 0:
            M = 1
        if M >= 6:
            M = 5
        DB = S - M
        N = M + int(np.fix(np.copysign(1.1, DB)))

        K += 2
        L += 2

        # V = self._cn_a[K][M] + abs(DA) * (self._cn_a[L][M] - self._cn_a[K][M])
        # W = self._cn_a[K][N] + abs(DA) * (self._cn_a[L][N] - self._cn_a[K][N])
        V = self._cn_a[M][K] + abs(DA) * (self._cn_a[M][L] - self._cn_a[M][K])
        W = self._cn_a[N][K] + abs(DA) * (self._cn_a[N][L] - self._cn_a[N][K])
        return (V + abs(DB) * (W - V)) * np.sign(beta)

    def _cx(self, alpha, el):
        """Compute the X axis force coefficient.

        Args:
            alpha: Angle of attack [rad].
            el: Elevator deflection [rad].

        Returns:
            X axis force coefficient.

        """

        S = 0.2 * alpha
        K = int(np.fix(S))
        if K <= -2:
            K = -1
        if K >= 9:
            K = 8
        DA = S - K
        L = K + int(np.fix(np.copysign(1.1, DA)))
        S = el / 12.0
        M = int(np.fix(S))
        if M <= -2:
            M = -1
        if M >= 2:
            M = 1
        DE = S - M
        N = M + int(np.fix(np.copysign(1.1, DE)))

        K += 2
        L += 2
        M += 2
        N += 2

        # V = self._cx_a[K][M] + abs(DA) * (self._cx_a[L][M] - self._cx_a[K][M])
        # W = self._cx_a[K][N] + abs(DA) * (self._cx_a[L][N] - self._cx_a[K][N])
        V = self._cx_a[M][K] + abs(DA) * (self._cx_a[M][L] - self._cx_a[M][K])
        W = self._cx_a[N][K] + abs(DA) * (self._cx_a[N][L] - self._cx_a[N][K])
        CX = V + (W - V) * abs(DE)
        return CX

    def _cy(self, beta, ail, rdr):
        """Compute the Y axis force coefficient.

        Args:
            beta: Sideslip angle [rad].
            ail: Aileron deflection [rad].
            rdr: Rudder deflection [rad].

        Returns:
            Y axis force coefficient.

        """

        return -0.02 * beta + 0.021 * (ail / 20.0) + 0.086 * (rdr / 30.0)

    def _cz(self, alpha, beta, el):
        """Compute the Z axis force coefficient.

        Args:
            alpha: Angle of attack [rad].
            beta: Sideslip angle [rad].
            el: Elevator deflection [rad].

        Returns:
            Z axis force coefficient.

        """

        S = 0.2 * alpha
        K = int(np.fix(S))
        if K <= -2:
            K = -1
        if K >= 9:
            K = 8
        DA = S - K
        L = K + int(np.fix(np.copysign(1.1, DA)))

        K += 2
        L += 2

        S = self._cz_a[0][K] + abs(DA) * (self._cz_a[0][L] - self._cz_a[0][K])
        CZ = S * (1 - (beta / 57.3) ** 2) - 0.19 * (el / 25.0)

        return CZ

    def _dlda(self, alpha, beta):
        """Compute the rolling moment due to aileron deflection.

        Args:
            alpha: Angle of attack [rad].
            beta: Sideslip angle [rad].

        Returns:
            Rolling moment due to aileron deflection.

        """

        S = 0.2 * alpha
        K = int(np.fix(S))
        if K <= -2:
            K = -1
        if K >= 9:
            K = 8
        DA = S - K
        L = K + int(np.fix(np.copysign(1.1, DA)))
        S = 0.1 * beta
        M = int(np.fix(S))
        if M <= -3:
            M = -2
        if M >= 3:
            M = 2
        DB = S - M
        N = M + int(np.fix(np.copysign(1.1, DB)))

        K += 2
        L += 2
        M += 3
        N += 3

        # V = self._dlda_a[K][M] + abs(DA) * (self._dlda_a[L][M] - self._dlda_a[K][M])
        # W = self._dlda_a[K][N] + abs(DA) * (self._dlda_a[L][N] - self._dlda_a[K][N])
        V = self._dlda_a[M][K] + abs(DA) * (self._dlda_a[M][L] - self._dlda_a[M][K])
        W = self._dlda_a[N][K] + abs(DA) * (self._dlda_a[N][L] - self._dlda_a[N][K])
        return V + abs(DB) * (W - V)

    def _dldr(self, alpha, beta):
        """Compute the rolling moment due to rudder deflection.

        Args:
            alpha: Angle of attack [rad].
            beta: Sideslip angle [rad].

        Returns:
            Rolling moment due to rudder deflection.

        """

        S = 0.2 * alpha
        K = int(np.fix(S))
        if K <= -2:
            K = -1
        if K >= 9:
            K = 8
        DA = S - K
        L = K + int(np.fix(np.copysign(1.1, DA)))
        S = 0.1 * beta
        M = int(np.fix(S))
        if M <= -3:
            M = -2
        if M >= 3:
            M = 2
        DB = S - M
        N = M + int(np.fix(np.copysign(1.1, DB)))

        K += 2
        L += 2
        M += 3
        N += 3

        # V = self._dldr_a[K][M] + abs(DA) * (self._dldr_a[L][M] - self._dldr_a[K][M])
        # W = self._dldr_a[K][N] + abs(DA) * (self._dldr_a[L][N] - self._dldr_a[K][N])
        V = self._dldr_a[M][K] + abs(DA) * (self._dldr_a[M][L] - self._dldr_a[M][K])
        W = self._dldr_a[N][K] + abs(DA) * (self._dldr_a[N][L] - self._dldr_a[N][K])
        return V + abs(DB) * (W - V)

    def _dnda(self, alpha, beta):
        """Compute the yawing moment due to aileron deflection.

        Args:
            alpha: Angle of attack [rad].
            beta: Sideslip angle [rad].

        Returns:
            Yawing moment due to aileron deflection.

        """

        S = 0.2 * alpha
        K = int(np.fix(S))
        if K <= -2:
            K = -1
        if K >= 9:
            K = 8
        DA = S - K
        L = K + int(np.fix(np.copysign(1.1, DA)))
        S = 0.1 * beta
        M = int(np.fix(S))
        if M <= -3:
            M = -2
        if M >= 3:
            M = 2
        DB = S - M
        N = M + int(np.fix(np.copysign(1.1, DB)))

        K += 2
        L += 2
        M += 3
        N += 3

        # V = self._dnda_a[K][M] + abs(DA) * (self._dnda_a[L][M] - self._dnda_a[K][M])
        # W = self._dnda_a[K][N] + abs(DA) * (self._dnda_a[L][N] - self._dnda_a[K][N])
        V = self._dnda_a[M][K] + abs(DA) * (self._dnda_a[M][L] - self._dnda_a[M][K])
        W = self._dnda_a[N][K] + abs(DA) * (self._dnda_a[N][L] - self._dnda_a[N][K])
        return V + abs(DB) * (W - V)

    def _dndr(self, alpha, beta):
        """Compute the yawing moment due to rudder deflection.

        Args:
            alpha: Angle of attack [rad].
            beta: Sideslip angle [rad].

        Returns:
            Yawing moment due to rudder deflection.

        """

        S = 0.2 * alpha
        K = int(np.fix(S))
        if K <= -2:
            K = -1
        if K >= 9:
            K = 8
        DA = S - K
        L = K + int(np.fix(np.copysign(1.1, DA)))
        S = 0.1 * beta
        M = int(np.fix(S))
        if M <= -3:
            M = -2
        if M >= 3:
            M = 2
        DB = S - M
        N = M + int(np.fix(np.copysign(1.1, DB)))

        K += 2
        L += 2
        M += 3
        N += 3

        # V = self._dndr_a[K][M] + abs(DA) * (self._dndr_a[L][M] - self._dndr_a[K][M])
        # W = self._dndr_a[K][N] + abs(DA) * (self._dndr_a[L][N] - self._dndr_a[K][N])
        V = self._dndr_a[M][K] + abs(DA) * (self._dndr_a[M][L] - self._dndr_a[M][K])
        W = self._dndr_a[N][K] + abs(DA) * (self._dndr_a[N][L] - self._dndr_a[N][K])
        return V + abs(DB) * (W - V)

    def _components(self, state, action):
        """Lookup table."""

        VT, alpha, beta, phi, theta, psi, p, q, r, pn, pe, alt, power = state
        thtl, el, ail, rdr = action

        CXT = self._cx(alpha, el)
        CYT = self._cy(beta, ail, rdr)
        CZT = self._cz(alpha, beta, el)

        DAIL = ail / 20.0
        DRDR = rdr / 30.0

        CLT = (
            self._cl(alpha, beta)
            + self._dlda(alpha, beta) * DAIL
            + self._dldr(alpha, beta) * DRDR
        )
        CMT = self._cm(alpha, el)
        CNT = (
            self._cn(alpha, beta)
            + self._dnda(alpha, beta) * DAIL
            + self._dndr(alpha, beta) * DRDR
        )

        return CXT, CYT, CZT, CLT, CMT, CNT


class F16LQRController(BasePolicy):
    """F16 LQR policy."""

    _K = np.zeros((4, 13))

    def __init__(self, action_space: Space = None):
        if action_space is not None:
            self.action_space = action_space
        else:
            raise ValueError("action space must be provided")

    def __call__(self, state: np.ndarray):
        return -self._K @ state
