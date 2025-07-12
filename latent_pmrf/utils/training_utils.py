from typing import Optional

import math

import numpy as np
from scipy.stats import gennorm, rv_continuous

import torch

def exponential_pdf(x, a):
    C = a / (np.exp(a) - 1)
    return C * np.exp(a * x)

# Define a custom probability density function
class ExponentialPDF(rv_continuous):
    def _pdf(self, x, a):
        return exponential_pdf(x, a)

u_exponential_distribution = ExponentialPDF(a=0, b=1, name='ExponentialPDF')


def compute_density_for_timestep_sampling(
    weighting_scheme: str,
    batch_size: int,
    logit_mean: float = None,
    logit_std: float = None,
    mode_scale: float = None,
    u_scale: float = None,
    process_index: Optional[int] = None,
    num_processes: Optional[int] = None,
):
    """Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device="cpu")
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    elif weighting_scheme == "u_shape":
        u = u_exponential_distribution.rvs(size=batch_size, a=u_scale)
        u = torch.from_numpy(u).float()
        u = torch.cat([u, 1 - u], dim=0)
        u = u[torch.randperm(u.shape[0])]
        u = u[:batch_size]
    elif weighting_scheme == 'stratified_uniform':
        n = batch_size * num_processes
        offsets = torch.arange(process_index, n, num_processes)
        u = torch.rand(size=(batch_size,))
        return ((offsets + u) / n)
    else:
        u = torch.rand(size=(batch_size,), device="cpu")
    return u


def compute_loss_weighting_for_adv(
    weighting_scheme: str,
    adv_weight: int,
    mid_adv_weight: int, 
    num_timesteps_per_phase: int,
    timesteps: torch.LongTensor,
    end_timesteps: torch.LongTensor,
):
    """Computes loss weighting scheme for SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "adaptive":
        d = (timesteps - end_timesteps - 1) / (num_timesteps_per_phase - 1)
        weighting = adv_weight * d ** math.log(mid_adv_weight / adv_weight, 0.5)
        return weighting
    else:
        weighting = torch.full_like(timesteps, adv_weight)
    return weighting

def lognormal_timestep_distribution(x, mean=-1.1, std=2.0):
    pdf = torch.erf((torch.log(x[1:]) - mean) / (std * math.sqrt(2))) - torch.erf(
            (torch.log(x[:-1]) - mean) / (std * math.sqrt(2))
        )
    pdf = pdf / pdf.sum()
    return pdf

def exponential_distribution(x, rate=3):
    # pdf = 1 - torch.exp(-rate[1:] * x) - (1 - torch.exp(-rate[:-1] * x))
    pdf = torch.exp(-rate * x[:-1]) - torch.exp(-rate * x[1:])
    pdf = pdf / pdf.sum()
    return pdf

def foldednormal_distribution(x, rate=3):
    # pdf = 1 - torch.exp(-rate[1:] * x) - (1 - torch.exp(-rate[:-1] * x))
    pdf = torch.exp(-rate * x[:-1]) - torch.exp(-rate * x[1:])
    pdf = pdf / pdf.sum()
    return pdf

def folded_normal_distribution(x, mean=0, std=1):
    cdf_1 = 0.5 * (torch.erf((x[1:] + mean) / (math.sqrt(2) * std)) + torch.erf((x[1:] - mean) / (math.sqrt(2) * std)))
    cdf_2 = 0.5 * (torch.erf((x[:-1] + mean) / (math.sqrt(2) * std)) + torch.erf((x[:-1] - mean) / (math.sqrt(2) * std)))

    pdf = cdf_1 - cdf_2
    pdf = pdf / pdf.sum()
    return pdf

def log_folded_normal_distribution(x, mean=0, std=1):
    a = torch.log(torch.min(x))

    x = torch.log(x) - a

    cdf_1 = 0.5 * (torch.erf((x[1:] + mean) / (math.sqrt(2) * std)) + torch.erf((x[1:] - mean) / (math.sqrt(2) * std)))
    cdf_2 = 0.5 * (torch.erf((x[:-1] + mean) / (math.sqrt(2) * std)) + torch.erf((x[:-1] - mean) / (math.sqrt(2) * std)))

    pdf = cdf_1 - cdf_2
    pdf = pdf / pdf.sum()
    return pdf

def folded_generalized_normal_distribution(x, beta, loc=0, scale=1):
    pdf = gennorm.pdf(x, beta, loc, scale) + gennorm.pdf(-x, beta, loc, scale)
    return pdf

class logfoldgennorm(object):
    def __init__(self, beta, loc=0, scale=1) -> None:
        self.beta = beta
        self.loc = loc
        self.scale = scale
    
    @staticmethod
    def sample(beta, loc=0, scale=1, size=1, min_v=0.002, max_v=80.0, positive_part: bool = True):
        x = gennorm.rvs(beta, loc, scale, size=size)
        
        if positive_part:
            x = np.exp(np.abs(x) + np.log(min_v))
        else:
            x = np.exp(-np.abs(x) + np.log(max_v))
        return x
    
    @staticmethod
    def pdf_discrete(x, beta, loc=0, scale=1, min_v=0.002, max_v=80.0, positive_part: bool = True):
        '''
            to sample from, see following example
            pdf = pdf_discrete(x, beta, loc, scale, min_v)
            r = torch.multinomial(torch.tensor(pdf), num_samples, replacement=True)
            r = x[r]
        '''
        x = torch.log(x) - math.log(min_v if positive_part else max_v)

        cdf_1 = gennorm.cdf(torch.abs(x[:-1]), beta, loc, scale) - gennorm.cdf(-torch.abs(x[:-1]), beta, loc, scale)
        cdf_2 = gennorm.cdf(torch.abs(x[1:]), beta, loc, scale) - gennorm.cdf(-torch.abs(x[1:]), beta, loc, scale)

        if not positive_part:
            cdf_1 = 1 - cdf_1
            cdf_2 = 1 - cdf_2

        cdf = cdf_2 - cdf_1
        pdf = cdf / cdf.sum()
        return pdf
    
    @staticmethod
    def pdf(x, beta, loc=0, scale=1, min_v=0.002, max_v=80.0, positive_part: bool = True):
        y = torch.log(x) - math.log(min_v if positive_part else max_v)
        pdf = (gennorm.pdf(y, beta, loc, scale) + gennorm.pdf(-y, beta, loc, scale)) / x
        return pdf
    
    def cdf(self, x, beta, loc=0, scale=1, min_v=0.002, max_v=80.0, positive_part: bool = True):
        x = torch.log(x) - math.log(min_v if positive_part else max_v)

        cdf = gennorm.cdf(torch.abs(x), beta, loc, scale) - gennorm.cdf(-torch.abs(x), beta, loc, scale)

        if not positive_part:
            cdf = 1 - cdf
        return cdf
    
def log_folded_generalized_normal_distribution(x, beta, loc=0, scale=1):
    a = torch.log(torch.min(x))

    y = torch.log(x) - a

    pdf = (gennorm.pdf(y, beta, loc, scale) + gennorm.pdf(-y, beta, loc, scale)) / x
    return pdf

def log_folded_generalized_normal_distribution_discrete(x, beta, loc=0, scale=1):
    a = torch.log(torch.min(x))

    x = torch.log(x) - a

    cdf_1 = gennorm.cdf(x[1:], beta, loc, scale) - gennorm.cdf(-x[1:], beta, loc, scale)
    cdf_2 = gennorm.cdf(x[:-1], beta, loc, scale) - gennorm.cdf(-x[:-1], beta, loc, scale)

    pdf = cdf_1 - cdf_2
    pdf = pdf / pdf.sum()
    return pdf

# def log_folded_normal_pdf(x, mu, sigma):
#     y = np.log(x)
#     return (norm.pdf(y, mu, sigma) + norm.pdf(-y, mu, sigma)) / x

# def log_folded_normal_cdf(x, mu, sigma):
#     y = np.log(x)
#     return norm.cdf(y, mu, sigma) - norm.cdf(-y, mu, sigma)

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

def tan(t, s0, s1):
    return np.tan((1 - t) * np.arctan(s0) + t * np.arctan(s1))

schedule_functions = {
    'default': default,
    'cubic_root': cubic_root,
    'square_root': square_root,
    'linear': linear,
    'cosine': cosine,
    'cosinev2': cosinev2,
    'square': square,
    'cubic': cubic,
    'Exp': Exp,
    'Exp_generalized': Exp_generalized
}