"""Causal backward-Euler endpoint convolution of an ideal-step kernel."""

import numpy as np
from scipy.signal import fftconvolve


def reconstruct(kernel, command):
    """Use the positive-time kernel and return a zero pre-load sample."""
    kernel = np.asarray(kernel, dtype=np.float64)
    command = np.asarray(command, dtype=np.float64)
    if kernel.ndim != 2 or command.ndim != 1 or len(command) != len(kernel) or len(command) < 2:
        raise ValueError("The load and kernel must share a time grid with at least two samples")
    if not np.isfinite(command).all() or not np.isfinite(kernel).all() or command[0] != 0.0:
        raise ValueError("Use finite values and a zero pre-load sample")
    jumps = np.diff(command)
    response = fftconvolve(jumps[:, None], kernel[1:], axes=0)[: len(jumps)]
    return np.vstack((np.zeros((1, kernel.shape[1])), response))


def uniform_grid(time):
    time = np.asarray(time, dtype=np.float64)
    if time.ndim != 1 or time.size < 2 or not np.isfinite(time).all() or time[0] != 0.0:
        raise ValueError("Provide a uniform time grid starting at zero")
    steps = np.diff(time)
    if (steps <= 0).any() or not np.allclose(steps, steps[0], atol=2e-12, rtol=1e-9):
        raise ValueError("The time samples must be strictly increasing and uniformly spaced")
    return time
