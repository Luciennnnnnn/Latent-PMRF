import numpy as np

from ..schedulers.scheduling_lcm import cubic_root, square_root, cosine, cosinev2, linear, square, cubic

def ema_default(t, s0, s1, u0, u1, time_step_func):
    t = (time_step_func(t, s0, s1) - s1) / (s0 - s1)
    return np.exp(t * np.log(u0))

def ema_cubic_root(t, s0, s1, u0, u1, time_step_func):
    t = (time_step_func(t, s0, s1) - s0) / (s1 - s0)
    return cubic_root(t, u0, u1)

def ema_square_root(t, s0, s1, u0, u1, time_step_func):
    t = (time_step_func(t, s0, s1) - s0) / (s1 - s0)
    return square_root(t, u0, u1)

def ema_linear(t, s0, s1, u0, u1, time_step_func):
    t = (time_step_func(t, s0, s1) - s0) / (s1 - s0)
    return linear(t, u0, u1)

def ema_cosine(t, s0, s1, u0, u1, time_step_func):
    t = (time_step_func(t, s0, s1) - s0) / (s1 - s0)
    return cosine(t, u0, u1)

def ema_cosinev2(t, s0, s1, u0, u1, time_step_func):
    t = (time_step_func(t, s0, s1) - s0) / (s1 - s0)
    return cosinev2(t, u0, u1)

def ema_square(t, s0, s1, u0, u1, time_step_func):
    t = (time_step_func(t, s0, s1) - s0) / (s1 - s0)
    return square(t, u0, u1)

def ema_cubic(t, s0, s1, u0, u1, time_step_func):
    t = (time_step_func(t, s0, s1) - s0) / (s1 - s0)
    return cubic(t, u0, u1)

ema_decay_schedule_functions = {
    'default': ema_default,
    'cubic_root': ema_cubic_root,
    'square_root': ema_square_root,
    'linear': ema_linear,
    'cosine': ema_cosine,
    'cosinev2': ema_cosinev2,
    'square': ema_square,
    'cubic': ema_cubic,
}