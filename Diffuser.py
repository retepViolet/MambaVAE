import torch
import torch.nn as nn
from transformers import MambaConfig
from transformers import BertConfig, BertModel
from diffusers import DDIMScheduler
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
        # denoiser
        # vae = MambaVAE()
        # vae.load_state_dict(torch.load("results/vae/model.pth", weights_only=True))
        # self.mamba = copy.deepcopy(vae.encoder)
        self.denoiser = nn.Transformer(
            d_model = 512, 
            nhead = 8, 
            num_encoder_layers = 3, 
            num_decoder_layers = 12, 
            dim_feedforward = 512*4, 
            dropout = 0,
            norm_first = True, 
            activation = 'gelu'
        )

        self.T = T
        self.time_embeds = self.init_embeds(128, T, requires_grad = False)
        self.time_linear = nn.Linear(128, 512)
        self.pos_embeds = self.init_embeds(512, 8)
        self.noise_scheduler = DDIMScheduler(num_train_timesteps = T, prediction_type="epsilon")
        self.loss_fn = torch.nn.MSELoss()

    def init_embeds(self, dim, length, power = 1000, requires_grad = True):
        pos = torch.arange(length, dtype=torch.float).unsqueeze(1)
        i = torch.arange(dim, dtype=torch.float).unsqueeze(0)
        angle_rates = 1 / torch.pow(power, (i // 2) * 2 / dim)
        angles = pos * angle_rates
        embeds = torch.zeros_like(angles)
        embeds[:, 0::2] = torch.sin(angles[:, 0::2])
        embeds[:, 1::2] = torch.cos(angles[:, 1::2])
        return nn.Parameter(embeds * 0.2, requires_grad = requires_grad)

    def predict(self, states, condition, timesteps):
        pos_embeds = self.pos_embeds.unsqueeze(0)
        time_embeds = self.time_linear(self.time_embeds[timesteps]).unsqueeze(1)
        pred = self.denoiser(
            src = condition + pos_embeds + time_embeds,
            tgt = states + pos_embeds + time_embeds, 
        )
        return pred

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
        target, condition = data['target'], data['condition']
        # Adding noise
        device = target.device
        timesteps = torch.randint(0, self.T, (target.shape[0],), device = device, dtype = torch.long)
        noise = torch.randn_like(target, device = device, dtype = target.dtype)
        noisy_states = self.noise_scheduler.add_noise(target, noise, timesteps).to(target.dtype)
        # Predict and loss
        pred = self.predict(noisy_states, condition, timesteps)
        loss = self.loss_fn(pred, noise)
        return (loss, timesteps, pred)


# class Diffuser(nn.Module):
#     def __init__(self, T = 1000):
#         super().__init__()
#         # denoiser
#         self.vae = MambaVAE()
#         self.vae.load_state_dict(torch.load("results/vae/model.pth", weights_only=True))
#         for param in self.vae.parameters():
#             param.requires_grad = False
#         self.denoiser = denoiser(12)

#         self.T = T
#         self.time_embeds = self.init_time_embeds(1536, T, 1000)
#         self.noise_scheduler = DDIMScheduler(num_train_timesteps = T, clip_sample = False, prediction_type = 'sample')
#         self.loss_fn = torch.nn.MSELoss()

#     def init_time_embeds(self, dim, length, power):
#         pos = torch.arange(length, dtype=torch.float).unsqueeze(1)
#         i = torch.arange(dim, dtype=torch.float).unsqueeze(0)
#         angle_rates = 1 / torch.pow(power, (i // 2) * 2 / dim)
#         angles = pos * angle_rates
#         pos_code = torch.zeros_like(angles)
#         pos_code[:, 0::2] = torch.sin(angles[:, 0::2])
#         pos_code[:, 1::2] = torch.cos(angles[:, 1::2])
#         return nn.Parameter(pos_code * 0.1)

#     def predict(self, states, condition, timesteps):
#         return self.denoiser(states, condition, self.time_embeds[timesteps])

#     @torch.no_grad()
#     def generate(self, input_ids, attention_mask = None, T = None):
#         self.eval()
#         self.noise_scheduler.set_timesteps(T if T is not None else self.T)
#         condition = self.vae.encode(input_ids = input_ids, 
#                             attention_mask = attention_mask)[0].squeeze(-1)
#         states = torch.randn(input_ids.shape[0], 1536, device = 'cuda')
#         for t in tqdm(self.noise_scheduler.timesteps):
#             timestep = torch.tensor([t]*input_ids.shape[0], device = input_ids.device, dtype = torch.long)
#             pred = self.predict(states, condition, timestep)
#             states = self.noise_scheduler.step(pred, t, states).prev_sample
#         return pred.unsqueeze(-1)
    
#     @torch.no_grad()
#     def proccess_data(self, data):
#         self.vae.eval()
#         condition = self.vae.encode(input_ids = data['question_ids'], 
#                             attention_mask = data['question_mask'])[0].squeeze(-1)
#         states = self.vae.encode(input_ids = data['full_ids'], 
#                             attention_mask = data['full_mask'])[0].squeeze(-1)
#         return states, condition

#     def forward(self, **data):
#         clean_states, condition = self.proccess_data(data)
#         # Adding noise
#         device = clean_states.device
#         timesteps = torch.randint(0, self.T, (clean_states.shape[0],), device = device, dtype = torch.long)
#         noise = torch.randn_like(clean_states, device = device, dtype = clean_states.dtype)
#         noisy_states = self.noise_scheduler.add_noise(clean_states, noise, timesteps).to(clean_states.dtype)
#         # Predict and loss
#         pred = self.predict(noisy_states, condition, timesteps)
#         loss = self.loss_fn(pred, clean_states)
#         res = self.vae.decode(pred.unsqueeze(-1), data['full_ids'], data['full_mask'], data['full_loss_mask'])
#         return (loss, res.loss, timesteps, pred, clean_states)