import numpy as np

import torch
from einops import repeat
from torch.nn.attention import SDPBackend, sdpa_kernel

def batch_jacobian(func, z, batch_size=64):
    b = z.shape[0]
    z_dim = np.prod(z.shape[1:])

    v = repeat(torch.eye(z_dim, dtype=z.dtype, device=z.device), 'c d -> (b c) d', b=b)
    v = v.view(-1, *z.shape[1:])

    z = repeat(z, 'b ... -> (b c) ...', c=z_dim)

    jac = []
    for i in range(0, z_dim, batch_size):
        v_i = v[i:i+batch_size]
        z_i = z[i:i+batch_size]
        with sdpa_kernel(SDPBackend.MATH):
            jvp_i = torch.autograd.functional.jvp(func, z_i, v=v_i)[1]
        jac.append(jvp_i)
        
        # if i == 0:
        #     print(f"{i=} {z_i=} {v_i=} {jvp_i=}", flush=True)
    jac = torch.cat(jac, dim=0)
    
    x_dim = jac.shape[1:]

    jac = jac.reshape(b, z_dim, *x_dim).transpose(1, -1).reshape(b, *x_dim, *z.shape[1:])
    return jac