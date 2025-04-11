import torch, math
import torch.nn as nn
from transformers import MambaConfig
from diffusers import DDIMScheduler
from DiT import DiT
from tqdm import tqdm
import copy

import importlib, InitMamba, VAE
importlib.reload(VAE)
importlib.reload(InitMamba)
from InitMamba import MambaModel
from VAE import MambaVAE


class Diffuser(nn.Module):
    def __init__(self, T = 1000):
        super().__init__()
        self.T = T
        # denoiser
        self.denoiser = DiT(in_dim=512, out_dim=256)
        # diffusion
        self.noise_scheduler = DDIMScheduler(num_train_timesteps = T, prediction_type="epsilon")
        self.loss_fn = torch.nn.MSELoss()

    def predict(self, states, condition, timesteps):
        states = torch.cat(states.chunk(2, dim = 2), 1)
        condition = torch.cat(condition.chunk(2, dim = 2), 1)
        hidden_states = torch.cat([states, condition], dim = 2)
        # denoiser
        pred = self.denoiser(hidden_states, timesteps).sample
        return torch.cat(pred.chunk(2, dim = 1), 2)

    @torch.no_grad()
    def generate(self, condition, T = None):
        self.eval()
        self.noise_scheduler.set_timesteps(T if T is not None else self.T)
        states = torch.randn(condition.shape[0], 8, 512, device = 'cuda')
        for t in tqdm(self.noise_scheduler.timesteps):
            timestep = torch.tensor([t]*condition.shape[0], device = condition.device, dtype = torch.long)
            pred = self.predict(states, condition, timestep)
            states = self.noise_scheduler.step(pred, t, states).prev_sample
        return states

    def forward(self, **data):
        target, condition = data['target'], data['condition'] # condition
        # Adding noise
        device = target.device
        timesteps = torch.randint(0, self.T, (target.shape[0],), device = device, dtype = torch.long)
        noise = torch.randn_like(target, device = device, dtype = target.dtype)
        noisy_states = self.noise_scheduler.add_noise(target, noise, timesteps).to(target.dtype)
        # Predict and loss
        pred = self.predict(noisy_states, condition, timesteps)
        loss = self.loss_fn(pred, noise)
        return (loss, timesteps, pred)