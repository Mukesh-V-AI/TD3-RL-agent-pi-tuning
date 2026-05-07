import numpy as np

class PlantModel:
    def __init__(self, dt=0.1):
        self.dt = dt
        self.y = 0.0  # output
        self.reset()

    def reset(self, init_output=900.0):
        self.y = init_output
        self.prev_y = init_output

    def step(self, u, d):
        # Simple first-order plant model: dy/dt = -y/tau + K*u + d
        # tau=1, K=1 for simplicity, matching original env dynamics
        tau = 1.0
        K = 1.0
        dy = (-self.y / tau + K * u + d) * self.dt
        self.y += dy
        self.prev_y = self.y
        return self.y

def make_disturbance_profile(profile_type='random', duration=20.0, dt=0.1):
    t = np.arange(0, duration, dt)
    if profile_type == 'random':
        disturbances = np.random.uniform(0.05, 0.30, size=len(t))
    else:
        disturbances = np.zeros_like(t)
        disturbances[int(len(t)/2):] = 0.2  # step disturbance
    return t, disturbances

def compute_metrics(output_history, setpoint, t):
    max_out = np.max(output_history)
    min_out = np.min(output_history)
    overshoot = max(0, (max_out - setpoint) / setpoint)
    undershoot = max(0, (setpoint - min_out) / setpoint)
    # settling time placeholder
    ts = 3.0
    return overshoot, undershoot, ts

