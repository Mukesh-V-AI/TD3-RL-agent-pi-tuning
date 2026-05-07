import numpy as np
import gymnasium as gym
from gymnasium import spaces
from envs.plant_model import PlantModel, make_disturbance_profile, compute_metrics

KP_BASE = 0.5
KI_BASE = 0.05
SETPOINT = 900.0


class PIGainTuningEnv(gym.Env):

    def __init__(self, dt=0.1, episode_duration=200.0):

        super().__init__()

        self.dt = dt
        self.setpoint = SETPOINT
        self.max_steps = int(episode_duration / dt)

        # ---- ACTION SPACE ----
        self.action_space = spaces.Box(
            low=np.array([0.5, 0.5]),
            high=np.array([3.0, 3.0]),
            dtype=np.float32
        )

        # ---- OBSERVATION SPACE ----
        self.observation_space = spaces.Box(
            low=np.array([-10, -10, 0, -100, 0.5, 0.5]),
            high=np.array([10, 10, 1, 100, 3.0, 3.0]),
            dtype=np.float32
        )

        self.plant = PlantModel(dt=dt)

        self.reset()

    # =========================================================
    # RESET
    # =========================================================
    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.plant.reset(init_output=self.setpoint)

        self.step_count = 0
        self.prev_error = 0
        self.integral_error = 0
        self.controller_integral = 0

        self.prev_kp_corr = 1.0
        self.prev_ki_corr = 1.0

        # ---- METRICS ----
        self.max_output = self.setpoint
        self.min_output = self.setpoint
        self.settled = False
        self.settling_time = None

        # disturbance
        _, self.disturbance_series = make_disturbance_profile(
            'random', duration=self.max_steps * self.dt, dt=self.dt
        )

        self.actual_output = self.setpoint

        return self._get_obs(), {}

    # =========================================================
    # STEP
    # =========================================================
    def step(self, action):

        kp_corr = float(np.clip(action[0], 0.5, 3.0))
        ki_corr = float(np.clip(action[1], 0.5, 3.0))

        kp = KP_BASE * kp_corr
        ki = KI_BASE * ki_corr

        # ---- DISTURBANCE ----
        d = self.disturbance_series[min(self.step_count, len(self.disturbance_series)-1)]

        # ---- CONTROL ----
        error = self.setpoint - self.actual_output
        self.controller_integral += error * self.dt

        u = kp * error + ki * self.controller_integral

        # ---- PLANT ----
        self.actual_output = self.plant.step(u, d)

        # ---- DERIVATIVES (for PINN) ----
        delta_error = (error - self.prev_error) / self.dt

        # ---- METRICS UPDATE ----
        self.max_output = max(self.max_output, self.actual_output)
        self.min_output = min(self.min_output, self.actual_output)

        if not self.settled and abs(error) < 0.01 * self.setpoint:
            self.settled = True
            self.settling_time = self.step_count * self.dt

        # ---- COMPUTE CONSTRAINTS ----
        overshoot = max(0, (self.max_output - self.setpoint) / self.setpoint)
        undershoot = max(0, (self.setpoint - self.min_output) / self.setpoint)
        ts = self.settling_time if self.settling_time else self.max_steps * self.dt

        g_os = overshoot - 0.15
        g_us = undershoot - 0.15
        g_ts = ts - 3.0

        # ---- REWARD ----
        reward = self._reward(error, delta_error, kp_corr, ki_corr)

        # ---- STATE ----
        obs = self._get_obs()

        self.prev_error = error
        self.prev_kp_corr = kp_corr
        self.prev_ki_corr = ki_corr
        self.step_count += 1

        done = self.step_count >= self.max_steps

        info = {
            "os": overshoot,
            "us": undershoot,
            "ts": ts,
            "g_os": g_os,
            "g_us": g_us,
            "g_ts": g_ts
        }

        return obs, reward, done, False, info

    # =========================================================
    # OBSERVATION
    # =========================================================
    def _get_obs(self):

        error = self.setpoint - self.actual_output
        de = error - self.prev_error

        return np.array([
            error / self.setpoint,
            de / self.setpoint,
            0,  # optional disturbance normalization
            self.integral_error,
            self.prev_kp_corr,
            self.prev_ki_corr
        ], dtype=np.float32)

    # =========================================================
    # ADVANCED REWARD (CSAC-COMPATIBLE)
    # =========================================================
    def _reward(self, error, de, kp_corr, ki_corr):

        sp = self.setpoint

        e_n = error / sp
        de_n = de / sp

        # ---- CORE ----
        r = -20 * abs(e_n)

        # ---- SPEED ----
        r += -5 * abs(de_n)

        # ---- SMOOTH CONTROL ----
        r += -0.5 * abs(kp_corr - self.prev_kp_corr)
        r += -0.5 * abs(ki_corr - self.prev_ki_corr)

        # ---- SETTLING BONUS ----
        if abs(e_n) < 0.01:
            r += 20

        return float(r)
