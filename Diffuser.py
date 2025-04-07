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


def init_embeds(dim, length, power = 10000, requires_grad = True):
        pos = torch.arange(length, dtype=torch.float).unsqueeze(1)
        i = torch.arange(dim, dtype=torch.float).unsqueeze(0)
        angle_rates = 1 / torch.pow(power, (i // 2) * 2 / dim)
        angles = pos * angle_rates
        embeds = torch.zeros_like(angles)
        embeds[:, 0::2] = torch.sin(angles[:, 0::2])
        embeds[:, 1::2] = torch.cos(angles[:, 1::2])
        return nn.Parameter(embeds, requires_grad = requires_grad)


class Diffuser(nn.Module):
    def __init__(self, T = 1000):
        super().__init__()
        self.T = T
        # denoiser
        config = MambaConfig.from_pretrained('state-spaces/mamba-130m-hf')
        config.num_hidden_layers  = 12
        self.denoiser = MambaModel.from_pretrained('state-spaces/mamba-130m-hf', config = config)
        self.time_embeds = init_embeds(256, T, requires_grad = False)
        # Mamba states
        self.expand_states = nn.Linear(512, 1536)
        self.reduce_states1 = nn.Linear(16, 8)
        self.reduce_states2 = nn.Linear(1536, 512)
        # embeds
        self.pos_embeds = init_embeds(768, 16)
        self.time_linear = nn.Linear(256, 1536)
        nn.init.zeros_(self.time_linear.weight)
        nn.init.zeros_(self.time_linear.bias)
        # diffusion
        self.noise_scheduler = DDIMScheduler(num_train_timesteps = T, prediction_type="epsilon")
        self.loss_fn = torch.nn.MSELoss()

    def predict(self, states, condition, timesteps):
        # embeds
        time_embeds = self.time_embeds[timesteps].unsqueeze(1)
        scale, shift = self.time_linear(time_embeds).chunk(2, dim = -1)
        pos_embeds = self.pos_embeds.unsqueeze(0).expand(states.shape[0], -1, -1)
        input_embds = pos_embeds * (scale + 1) + shift
        # states
        states = torch.cat([states, condition], dim = 1)
        states = self.expand_states(states).transpose(1, 2)
        # denoiser
        pred = self.denoiser(inputs_embeds = input_embds, 
                             attention_mask = torch.ones(input_embds.shape[0], 16, device = input_embds.device, dtype = torch.long),
                             inputs_ssm_states = states, 
                             inputs_ssm_layer = -1, 
                             output_ssm_last_states = True).ssm_last_states
        pred = self.reduce_states2(self.reduce_states1(pred).transpose(1, 2))
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
        target, condition = data['target'], data['target'] # condition
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
#         # vae = MambaVAE()
#         # vae.load_state_dict(torch.load("results/vae/model.pth", weights_only=True))
#         # self.mamba = copy.deepcopy(vae.encoder)
#         # self.denoiser = nn.Transformer(
#         #     d_model = 1024, 
#         #     nhead = 16,
#         #     num_encoder_layers = 6, 
#         #     num_decoder_layers = 6,
#         #     dim_feedforward = 1024,
#         #     dropout = 0,
#         #     norm_first = True, 
#         #     activation = 'gelu'
#         # )
#         self.denoiser = nn.TransformerEncoder(
#             encoder_layer = nn.TransformerEncoderLayer(
#                 d_model = 1024,
#                 nhead = 16,
#                 dim_feedforward = 1024,
#                 dropout = 0,
#                 batch_first = True,
#                 norm_first = True,
#                 activation = 'gelu',
#             ),
#             num_layers = 12,
#             norm = nn.LayerNorm(1024),
#         )

#         self.T = T
#         self.time_embeds = init_embeds(256, T, requires_grad = False)
#         # self.first_linear = nn.Linear(512, 768)
#         self.last_linear = nn.Linear(1024, 512)
#         self.time_linear = nn.Linear(256, 2048)
#         nn.init.zeros_(self.time_linear.weight)
#         nn.init.zeros_(self.time_linear.bias)
#         self.pos_embeds = init_embeds(1024, 8)
#         self.noise_scheduler = DDIMScheduler(num_train_timesteps = T, prediction_type="epsilon")
#         self.loss_fn = torch.nn.MSELoss()

#     def predict(self, states, condition, timesteps):
#         pos_embeds = self.pos_embeds.unsqueeze(0)
#         # time
#         time_embeds = self.time_embeds[timesteps].unsqueeze(1) #.expand(-1, 8, -1)
#         scale, shift = self.time_linear(time_embeds).chunk(2, dim = -1)
#         scale = scale + 1.0
#         # input
#         input_embeds = torch.cat([states, condition], dim = -1)
#         input_embeds = input_embeds * scale + shift + pos_embeds
#         pred = self.denoiser(
#             src = input_embeds + pos_embeds,
#         )
#         return self.last_linear(pred)

#     @torch.no_grad()
#     def generate(self, condition, T = None):
#         self.eval()
#         self.noise_scheduler.set_timesteps(T if T is not None else self.T)
#         states = torch.randn(condition.shape[0], 8, 512, device = 'cuda')
#         for t in tqdm(self.noise_scheduler.timesteps):
#             timestep = torch.tensor([t]*condition.shape[0], device = condition.device, dtype = torch.long)
#             pred = self.predict(states, condition, timestep)
#             states = self.noise_scheduler.step(pred, t, states).prev_sample
#         return states

#     def forward(self, **data):
#         target, condition = data['target'], data['condition']
#         # Adding noise
#         device = target.device
#         timesteps = torch.randint(0, self.T, (target.shape[0],), device = device, dtype = torch.long)
#         noise = torch.randn_like(target, device = device, dtype = target.dtype)
#         noisy_states = self.noise_scheduler.add_noise(target, noise, timesteps).to(target.dtype)
#         # Predict and loss
#         pred = self.predict(noisy_states, condition, timesteps)
#         loss = self.loss_fn(pred, noise)
#         return (loss, timesteps, pred)