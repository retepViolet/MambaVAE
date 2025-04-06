import torch, math
import torch.nn as nn
from transformers import MambaConfig
from transformers import BertConfig, BertModel
from diffusers import DDIMScheduler, DiTTransformer2DModel
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
        self.time_embeds = self.init_embeds(512, T, requires_grad = False)
        self.time_linear = nn.Linear(512, 1024)
        nn.init.zeros_(self.time_linear.weight)
        nn.init.zeros_(self.time_linear.bias)
        self.pos_embeds = self.init_embeds(512, 8)
        self.noise_scheduler = DDIMScheduler(num_train_timesteps = T, prediction_type="epsilon")
        self.loss_fn = torch.nn.MSELoss()

    def init_embeds(self, dim, length, power = 10000, requires_grad = True):
        pos = torch.arange(length, dtype=torch.float).unsqueeze(1)
        i = torch.arange(dim, dtype=torch.float).unsqueeze(0)
        angle_rates = 1 / torch.pow(power, (i // 2) * 2 / dim)
        angles = pos * angle_rates
        embeds = torch.zeros_like(angles)
        embeds[:, 0::2] = torch.sin(angles[:, 0::2])
        embeds[:, 1::2] = torch.cos(angles[:, 1::2])
        return nn.Parameter(embeds, requires_grad = requires_grad)

    def predict(self, states, condition, timesteps):
        pos_embeds = self.pos_embeds.unsqueeze(0)
        scale, shift = self.time_linear(self.time_embeds[timesteps]).unsqueeze(1).chunk(2, dim=-1)
        scale = scale + 1.0
        pred = self.denoiser(
            src = (condition + pos_embeds) * scale + shift,
            tgt = states + pos_embeds, 
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
        target, condition = data['target'], data['target'] # data['condition'] ！！！！！！！！！
        # Adding noise
        device = target.device
        timesteps = torch.randint(0, self.T, (target.shape[0],), device = device, dtype = torch.long)
        noise = torch.randn_like(target, device = device, dtype = target.dtype)
        noisy_states = self.noise_scheduler.add_noise(target, noise, timesteps).to(target.dtype)
        # Predict and loss
        pred = self.predict(noisy_states, condition, timesteps)
        loss = self.loss_fn(pred, noise)
        return (loss, timesteps, pred)