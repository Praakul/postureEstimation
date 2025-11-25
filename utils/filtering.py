import math
import time

class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        """
        OneEuroFilter for smoothing noisy signals (jitter).
        
        Params:
        - min_cutoff: Lower = smoother (less jitter), but more lag.
        - beta: Higher = faster response (less lag), but more jitter during movement.
        - d_cutoff: Cutoff for the derivative (velocity).
        """
        self.t_prev = t0
        self.x_prev = x0
        self.dx_prev = dx0
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.alpha = self._alpha(self.min_cutoff)

    def _alpha(self, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau * 30.0) # Assuming ~30fps

    def __call__(self, t, x):
        """
        Update the filter with a new measurement 'x' at time 't'.
        """
        t_e = t - self.t_prev
        
        # Avoid division by zero if timestamps are identical
        if t_e <= 0:
            return self.x_prev

        # Estimate velocity (jittery)
        a_d = self._alpha(self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        
        # Calculate adaptive cutoff frequency based on velocity
        # If moving fast (high dx), cutoff increases -> less smoothing, less lag.
        # If standing still (low dx), cutoff decreases -> heavy smoothing, stable signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        
        self.t_prev = t
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        
        return x_hat