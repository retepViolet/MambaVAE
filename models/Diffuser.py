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
        self.denoiser = DiT(in_dim = 128, out_dim = 64, length = 64, num_layers = 6, dropout = 0.1)
        # vae
        self.vae = MambaVAE()
        self.proj_in_2 = nn.Linear(768, 1024)
        nn.init.constant_(self.proj_in_2.weight, 0.0)
        nn.init.constant_(self.proj_in_2.bias, 0.0)
        self.norm_in = nn.LayerNorm(512, elementwise_affine=False, eps=1e-6)
        self.norm_out = nn.LayerNorm(1536, elementwise_affine=False, eps=1e-6)
        # diffusion
        self.noise_scheduler = DDIMScheduler(num_train_timesteps = T, 
                                             beta_schedule = 'squaredcos_cap_v2', #'squaredcos_cap_v2',
                                            #  beta_start = 0.005, beta_end = 0.005,
                                             prediction_type = 'v_prediction')
        # print(self.noise_scheduler.alphas_cumprod)
        self.loss_fn = torch.nn.MSELoss()

    def tf_predict(self, states, condition, timesteps):
        states = states.reshape(-1, 64, 64)
        condition = condition.reshape(-1, 64, 64)
        hidden_states = torch.cat([states, condition], dim=-1) # (bs, 64, 128)
        pred = self.denoiser(hidden_states, timesteps).sample  # pred: (bs, 64, 64)
        pred = pred.reshape(-1, 8, 512)
        return pred
    
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
        states = self.vae.mlp1(states).transpose(1,2)
        states =  self.vae.get_mean(self.norm_out(states))
        return states

    @torch.no_grad()
    def generate(self, input_ids, attention_mask = None, T = None, progress_bar = True):
        self.eval()
        # return self.mamba_predict(None, input_ids, attention_mask, None)
        if T is None: T = self.T
        self.noise_scheduler.set_timesteps(T)
        states = torch.randn(input_ids.shape[0], 8, 512, device = input_ids.device)
        # states = self.noise_scheduler.add_noise(target, states, torch.tensor([300]*input_ids.shape[0], device = input_ids.device, dtype = torch.long)).to(target.dtype)
        for t in tqdm(self.noise_scheduler.timesteps, desc="Generating samples", disable = not progress_bar):
            timestep = torch.full((input_ids.shape[0],), t, device=input_ids.device, dtype=torch.long)
            condition = self.mamba_predict(states, input_ids, attention_mask, timestep)
            pred_uncond = self.tf_predict(states, torch.zeros_like(condition), timestep)
            pred_cond = self.tf_predict(states, condition, timestep)
            pred = pred_uncond + 1 * (pred_cond - pred_uncond)
            states = self.noise_scheduler.step(pred, t, states).prev_sample
        return states
    
    def velocity(self, x0, noise, timesteps):
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(x0.device).detach()
        sqrt_alpha_t   = alphas_cumprod[timesteps].sqrt()[:, None, None] 
        sqrt_one_minus = (1.0 - alphas_cumprod[timesteps]).sqrt()[:, None, None]
        v = sqrt_alpha_t * noise - sqrt_one_minus * x0
        return v.to(x0.dtype)
    
    def diff_prepare(self, target):
        device = target.device
        timesteps = torch.randint(0, self.T, (target.shape[0],), device = device, dtype = torch.long)
        noise = torch.randn_like(target, device = device, dtype = target.dtype)
        noisy_states = self.noise_scheduler.add_noise(target, noise, timesteps).to(target.dtype)
        return noisy_states, timesteps, noise

    def forward(self, **data):
        target = data['target']
        noisy_states, timesteps, noise = self.diff_prepare(target)
        v = self.velocity(target, noise, timesteps)
        # Predict and loss 
        # condition = self.mamba_predict(noisy_states, data['full_ids'], data['full_mask'], timesteps)
        condition = self.mamba_predict(noisy_states, data['question_ids'], data['question_mask'], timesteps)
        mask = (torch.rand(condition.shape[0], device=condition.device) < 0.9).to(condition.dtype)
        condition = condition * mask[:, None, None]
        pred = self.tf_predict(noisy_states, condition, timesteps)
        mse_loss = self.loss_fn(pred, v)
        # logits_loss = self.vae.decode(condition, data['full_ids'], data['full_mask'], data['full_loss_mask']).loss
        # logits_loss = logits_loss + self.loss_fn(condition, target)
        return (mse_loss, mse_loss, 0)



# class Diffuser(nn.Module):
#     def __init__(self, T = 1000):
#         super().__init__()
#         self.T = T
#         # denoiser
#         self.denoiser = DiT(in_dim=128, out_dim=64, length = 64, num_layers = 12, dropout = 0.1)
#         # diffusion
#         self.noise_scheduler = DDIMScheduler(num_train_timesteps = T, beta_schedule = 'squaredcos_cap_v2')
#         self.loss_fn = torch.nn.MSELoss()

#     def tf_predict(self, states, condition, timesteps):
#         states = states.reshape(-1, 64, 64)
#         condition = condition.reshape(-1, 64, 64)
#         hidden_states = torch.cat([states, condition], dim=-1) # (bs, 64, 128)
#         pred = self.denoiser(hidden_states, timesteps).sample  # pred: (bs, 64, 64)
#         pred = pred.reshape(-1, 8, 512)
#         return pred
    
#     @torch.no_grad()
#     def generate(self, condition, T = None, progress_bar = True):
#         self.eval()
#         if T is None: T = self.T
#         self.noise_scheduler.set_timesteps(T)
#         states = torch.randn(condition.shape[0], 8, 512, device = condition.device)
#         for t in tqdm(self.noise_scheduler.timesteps, desc="Generating samples", disable = not progress_bar):
#             timestep = torch.tensor([t]*condition.shape[0], device = condition.device, dtype = torch.long)
#             pred = self.tf_predict(states, condition, timestep)
#             states = self.noise_scheduler.step(pred, t, states).prev_sample
#         return states
    
#     def diff_prepare(self, target):
#         device = target.device
#         timesteps = torch.randint(0, self.T, (target.shape[0],), device = device, dtype = torch.long)
#         noise = torch.randn_like(target, device = device, dtype = target.dtype)
#         noisy_states = self.noise_scheduler.add_noise(target, noise, timesteps).to(target.dtype)
#         return noisy_states, timesteps, noise
    
#     def compute_snr(self, timesteps):
#         alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(timesteps.device)
#         alpha_t_squared = alphas_cumprod[timesteps]
#         snr = alpha_t_squared / (1. - alpha_t_squared)
#         return snr

#     def forward(self, **data):
#         noisy_states, timesteps, noise = self.diff_prepare(data['target'])
#         pred = self.tf_predict(noisy_states, data['condition'], timesteps)
#         mse_loss_per_sample = torch.mean((pred - noise)**2, dim=(1,2))
#         snr = self.compute_snr(timesteps)
#         weights = torch.clamp(snr, max = 5.0) / snr
#         weighted_loss = (weights.detach() * mse_loss_per_sample).mean()
#         return (weighted_loss,)