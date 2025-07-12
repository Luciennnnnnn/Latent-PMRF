import numpy as np

import torch


def discrete_timesteps(N, l, r, func):
    if func == 'linear':
        return np.linspace(l, r, N, dtype=np.int32)
    else:
        raise NotImplementedError("??")


def default(t, s0, s1):
    s0_square = s0 ** 2
    s1_square = s1 ** 2

    s = np.sqrt(t * (s1_square - s0_square) + s0_square)
    return s
    
def cubic_root(t, s0, s1):
    return s0 + t ** (1 / 3) * (s1 - s0)

def square_root(t, s0, s1):
    return s0 + np.sqrt(t) * (s1 - s0)

def linear(t, s0, s1):
    return s0 + t * (s1 - s0)

def cosine(t, s0, s1):
    return s0 + (1 - np.cos(t * np.pi / 2)) * (s1 - s0)

def cosinev2(t, s0, s1):
    return s0 + (1 - np.cos(t * np.pi)) / 2 * (s1 - s0)

def square(t, s0, s1):
    return s0 + t**2 * (s1 - s0)

def cubic(t, s0, s1):
    return s0 + t**3 * (s1 - s0)

def exp(t, s0, s1):
    return s0 + (np.exp(t) - 1) / (np.e - 1) * (s1 - s0)

def Exp(step, K, s0, s1):
    if torch.is_tensor(s0) or torch.is_tensor(s1):
        K_prime = torch.floor(K / (torch.log2(s1 // s0) + 1)).to(torch.int32)
        num_discretization_step = torch.minimum(s0 * 2 ** (step // K_prime), s1) + 1
    else:
        K_prime = np.floor(K / (np.log2(s1 // s0) + 1)).astype(int)
        num_discretization_step = np.minimum(s0 * 2 ** (step // K_prime), s1) + 1 
    return num_discretization_step

def Exp_generalized(step, total_step, s0, s1, r=1):
    if torch.is_tensor(s0) or torch.is_tensor(s1):
        d = torch.log2(s1 // s0) + 1

        if r == 1:
            x = torch.floor(step / total_step * d).to(torch.int32)
        else:
            x = torch.floor(np.emath.logn(r, 1 - step / total_step * (1 - r ** d))).to(torch.int32)


        K_prime = torch.floor(total_step / (torch.log2(s1 // s0) + 1)).to(torch.int32)
        num_discretization_step = torch.minimum(s0 * 2 ** (step // K_prime), s1) + 1
    else:
        d = np.log2(s1 // s0) + 1

        if r == 1:
            x = np.floor(step / total_step * d).astype(int)
        else:
            x = np.floor(np.emath.logn(r, 1 - step / total_step * (1 - r ** d))).astype(int)
        
        num_discretization_step = np.minimum(s0 * 2 ** x, s1) + 1 
    return num_discretization_step

def tan(t, s0, s1, x0, x1):
    a = (x1 - x0) / (s1 - s0)
    b = x0 - a * s0

    print(s0, s1)
    print(s0 * a + b, s1 * a + b)
    
    return (np.tan((1 - t) * np.arctan(s0 * a + b) + t * np.arctan(s1 * a + b)) - b) / a

def inverse_tanh(x):
    return np.log((1 + x) / (1 - x)) / 2

def tanh(t, s0, s1, x0, x1):
    a = (x1 - x0) / (s1 - s0)
    b = x0 - a * s0
    
    print(s0, s1)
    print(s0 * a + b, s1 * a + b)

    return (np.tanh((1 - t) * inverse_tanh(s0 * a + b) + t * inverse_tanh(s1 * a + b)) - b) / a

schedule_functions = {
    'default': default,
    'cubic_root': cubic_root,
    'square_root': square_root,
    'linear': linear,
    'cosine': cosine,
    'cosinev2': cosinev2,
    'square': square,
    'cubic': cubic,
    'exp': exp,
    'Exp': Exp,
    'Exp_generalized': Exp_generalized
}