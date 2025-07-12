import math

import torch

from ..schedulers.scheduling_lctm import LCTMScheduler


def solver(latents, start_timesteps, end_timesteps, predict, scheduler, solver_steps, eta: float = 0):
    start_timesteps = start_timesteps[:, None]
    end_timesteps = end_timesteps[:, None]

    timesteps = start_timesteps + torch.arange(solver_steps + 1, device=latents.device)[None, :] / solver_steps * (end_timesteps - start_timesteps)
    timesteps = timesteps.round().to(dtype=torch.long).T

    timesteps = timesteps.squeeze()

    print(f"{timesteps=}", flush=True)
    scheduler.set_timesteps(num_inference_steps=solver_steps, timesteps=timesteps.cpu().numpy(), device=latents.device)
    
    extra_step_kwargs = {}

    if isinstance(scheduler, LCTMScheduler):
        extra_step_kwargs['eta'] = eta

    for i in range(solver_steps):
        t = scheduler.timesteps[i]
        
        if isinstance(scheduler, LCTMScheduler):
            if i + 1 < len(scheduler.timesteps):
                t_prev = scheduler.timesteps[i + 1]
            else:
                t_prev = 0
                
            t_s = torch.floor((1 - extra_step_kwargs['eta']) * t_prev).to(torch.long)
        else:
            t_s = t

        model_pred = predict(noisy_latents=latents, timesteps=t, target_timesteps=t_s)

        latents = scheduler.step(model_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
    return latents