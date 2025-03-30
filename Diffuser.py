from datasets.arrow_dataset import sample
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


class mlp_layer(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Sequential(
              nn.Linear(1536*2+128, 768),
              nn.GELU(),
              nn.Linear(768, 1536*2+128),
              nn.LayerNorm(1536*2+128),
            ))
        self.last = nn.Linear(1536*2+128, 1536)
    
    def forward(self, x, condition, time_embeds):
        x = torch.cat((x, condition, time_embeds), -1)
        for layer in self.layers:
            x = x + layer(x)
        return self.last(x)


class Diffuser(nn.Module):
    def __init__(self, T = 1000):
        super().__init__()
        # denoiser
        self.vae = MambaVAE()
        self.vae.load_state_dict(torch.load("results/vae/model.pth", weights_only=True))
        self.backbone = copy.deepcopy(self.vae.encoder)
        for name, param in self.backbone.named_parameters():
          # if '11' not in name and '10' not in name and '9' not in name:
            param.requires_grad = False
        for param in self.vae.parameters():
            param.requires_grad = False
        self.mlp = mlp_layer(12)

        self.T = T
        self.time_embeds = self.init_time_embeds(128, T, 100)
        self.noise_scheduler = DDIMScheduler(num_train_timesteps = T, clip_sample = True, clip_sample_range = 3, prediction_type="epsilon") # prediction_type = 'sample'
        self.loss_fn = torch.nn.MSELoss()

    def init_time_embeds(self, dim, length, power):
        pos = torch.arange(length, dtype=torch.float).unsqueeze(1)
        i = torch.arange(dim, dtype=torch.float).unsqueeze(0)
        angle_rates = 1 / torch.pow(power, (i // 2) * 2 / dim)
        angles = pos * angle_rates
        pos_code = torch.zeros_like(angles)
        pos_code[:, 0::2] = torch.sin(angles[:, 0::2])
        pos_code[:, 1::2] = torch.cos(angles[:, 1::2])
        return nn.Parameter(pos_code)

    def predict(self, states, condition, timesteps):
        pred = self.mlp(states, condition, self.time_embeds[timesteps])
        return pred

    @torch.no_grad()
    def generate(self, input_ids, attention_mask = None, T = None):
        self.eval()
        self.noise_scheduler.set_timesteps(T if T is not None else self.T)
        states = torch.randn(input_ids.shape[0], 1536, device = 'cuda')
        condition = self.get_condition(input_ids, attention_mask)
        for t in tqdm(self.noise_scheduler.timesteps):
            timestep = torch.tensor([t]*input_ids.shape[0], device = input_ids.device, dtype = torch.long)
            pred = self.predict(states, condition, timestep)
            states = self.noise_scheduler.step(pred, t, states).prev_sample
        return states.unsqueeze(-1)
    
    def get_condition(self, input_ids = None, attention_mask = None, hidden_states = None):
        return torch.zeros(attention_mask.shape[0], 1536).to('cuda')
        condition = self.backbone(input_ids = input_ids,
                      inputs_embeds = hidden_states, 
                      attention_mask = attention_mask, 
                      layer_range = range(9, 12) if hidden_states is not None else None,
                      output_ssm_last_states = True).ssm_last_states
        condition = self.vae.mlp1(condition).squeeze(-1)
        return self.vae.get_mean(condition)

    @torch.no_grad()
    def proccess_data(self, data):
        return torch.zeros(data['full_ids'].shape[0], 1536).to('cuda'), None
        self.vae.eval()
        hidden_states = self.vae.encoder(input_ids = data['full_ids'], 
                            attention_mask = data['full_mask'], 
                            layer_range = range(9))[0]
        states = self.vae.encode(input_ids = data['full_ids'], 
                            attention_mask = data['full_mask'],
                            hidden_states = hidden_states)[0]
        return states.squeeze(-1), hidden_states

    def forward(self, **data):
        clean_states, hidden_states = self.proccess_data(data)
        condition = self.get_condition(attention_mask = data['question_mask'], hidden_states = hidden_states)
        # Adding noise
        device = clean_states.device
        timesteps = torch.randint(0, self.T, (clean_states.shape[0],), device = device, dtype = torch.long)
        noise = torch.randn_like(clean_states, device = device, dtype = clean_states.dtype)
        noisy_states = self.noise_scheduler.add_noise(clean_states, noise, timesteps).to(clean_states.dtype)
        # Predict and loss
        pred = self.predict(noisy_states, condition, timesteps)
        loss = self.loss_fn(pred, noise)
        # res = self.vae.decode(pred, data['full_ids'], data['full_mask'], data['full_loss_mask'])
        return (loss, timesteps, pred, clean_states)


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