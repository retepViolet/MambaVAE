import torch
import torch.nn as nn
from transformers import MambaConfig
from diffusers import DDIMScheduler

import importlib, InitMamba, VAE
importlib.reload(InitMamba)
importlib.reload(VAE)
from InitMamba import MambaModel
from VAE import MambaVAE

class Denoiser(nn.Module):
    def __init__(self, T = 100):
        super().__init__()
        self.backbone = MambaModel.from_pretrained('state-spaces/mamba-130m-hf')
        self.T = T
        self.init_time_embeds(self)

    def init_time_embeds(self):
        pos = torch.arange(self.T, dtype=torch.float).unsqueeze(1)
        i = torch.arange(768, dtype=torch.float).unsqueeze(0)
        angle_rates = 1 / torch.pow(10, (i // 2) * 2 / self.d)
        angles = pos * angle_rates
        pos_code = torch.zeros_like(angles)
        pos_code[:, 0::2] = torch.sin(angles[:, 0::2])
        pos_code[:, 1::2] = torch.cos(angles[:, 1::2])
        self.time_embeds = nn.Parameter(pos_code)
    
    def forward(self, states, timesteps, input_ids, attention_mask = None):
        hidden_states = self.backbone(input_ids = input_ids, 
                            attention_mask = attention_mask, 
                            layer_range = range(21))[0]
        hidden_states = hidden_states + self.time_embeds[timesteps].unsqueeze(1)
        return self.backbone(inputs_embeds = hidden_states, 
                            attention_mask = attention_mask, 
                            layer_range = range(21, 24),
                            inputs_ssm_states = states,
                            inputs_ssm_layer = 21,
                            output_ssm_last_states = True).ssm_last_states
