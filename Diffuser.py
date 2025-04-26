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
        self.denoiser = DiT(in_dim=512, out_dim=256, num_layers=12, dropout = 0.1)
        # vae
        self.vae = MambaVAE()
        self.proj_in_2 = nn.Linear(768, 1024)
        nn.init.constant_(self.proj_in_2.weight, 0.0)
        nn.init.constant_(self.proj_in_2.bias, 0.0)
        self.norm_in = nn.LayerNorm(512, elementwise_affine=False, eps=1e-6)
        for name, param in self.named_parameters():
            if 'denoiser' not in name:
                param.requires_grad = False
        # diffusion
        self.noise_scheduler = DDIMScheduler(num_train_timesteps = T, beta_schedule = 'squaredcos_cap_v2')
        self.loss_fn = torch.nn.MSELoss()

    def tf_predict(self, states, condition, timesteps):
        states = torch.cat(states.chunk(2, dim = 2), 1)
        condition = torch.cat(condition.chunk(2, dim = 2), 1)
        hidden_states = torch.cat([states, condition], dim = 2)
        # denoiser
        pred = self.denoiser(hidden_states, timesteps).sample
        return torch.cat(pred.chunk(2, dim = 1), 2)
    
    def mamba_predict(self, states, question_ids, question_mask, timesteps):
        class_labels = torch.zeros(timesteps.shape[0], device = states.device, dtype=torch.long)
        conditioning = self.denoiser.transformer_blocks[0].norm1.emb(timesteps, class_labels, hidden_dtype=states.dtype)
        shift, scale = self.proj_in_2(torch.nn.functional.silu(conditioning)).chunk(2, dim=1)
        states = self.norm_in(states) * (1 + scale[:, None]) + shift[:, None]
        states = self.vae.mlp2(self.vae.get_states(states).transpose(1,2))
        states =  self.vae.encoder(input_ids = question_ids,
                                attention_mask = question_mask,
                                inputs_ssm_states = states,
                                inputs_ssm_layer = 0,
                                output_ssm_last_states = True).ssm_last_states
        states =  self.vae.get_mean(self.vae.mlp1(states).transpose(1,2))
        return states

    @torch.no_grad()
    def generate(self, input_ids, attention_mask = None, T = None, progress_bar = True):
        self.eval()
        if T is None: T = self.T
        self.noise_scheduler.set_timesteps(T)
        states = torch.randn(input_ids.shape[0], 8, 512, device = input_ids.device)
        for t in tqdm(self.noise_scheduler.timesteps, desc="Generating samples", disable = not progress_bar):
            timestep = torch.tensor([t]*input_ids.shape[0], device = input_ids.device, dtype = torch.long)
            condition = self.mamba_predict(states, input_ids, attention_mask, timestep)
            pred = self.tf_predict(states, condition, timestep)
            states = self.noise_scheduler.step(pred, t, states).prev_sample
        return states

    def forward(self, **data):
        target = data['target']
        # Adding noise
        device = target.device
        timesteps = torch.randint(0, self.T, (target.shape[0],), device = device, dtype = torch.long)
        noise = torch.randn_like(target, device = device, dtype = target.dtype)
        noisy_states = self.noise_scheduler.add_noise(target, noise, timesteps).to(target.dtype)
        # Predict and loss
        condition = self.mamba_predict(noisy_states, data['question_ids'], data['question_mask'], timesteps)
        pred = self.tf_predict(noisy_states, condition, timesteps)
        mse_loss = self.loss_fn(pred, noise)
        # logits_loss = self.vae.decode(pred, data['full_ids'], data['full_mask'], data['full_loss_mask']).loss
        return (mse_loss, timesteps, pred)