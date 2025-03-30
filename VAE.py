import torch
import torch.nn as nn
from transformers import MambaConfig
from random import random

import importlib, InitMamba
importlib.reload(InitMamba)
from InitMamba import MambaModel, MambaForCausalLM


class MambaVAE(nn.Module):
    def __init__(self):
        super().__init__()
        config = MambaConfig.from_pretrained('state-spaces/mamba-130m-hf')
        self.decoder = MambaForCausalLM.from_pretrained('state-spaces/mamba-130m-hf', config = config)
        config.num_hidden_layers  = 12
        self.encoder = MambaModel.from_pretrained('state-spaces/mamba-130m-hf', config = config)

        self.get_log_var = nn.Linear(1536, 128)
        self.get_mean = nn.Linear(1536, 128)
        self.get_states = nn.Linear(128, 1536)

    def encode(self, input_ids, attention_mask = None):
        states = self.encoder(input_ids = input_ids, 
                    attention_mask = attention_mask, 
                    output_ssm_last_states = True).ssm_last_states
        self.ssm_dtype = states.dtype
        states = states.transpose(1,2)
        return self.get_mean(states), self.get_log_var(states)
    
    def decode(self, states, input_ids, attention_mask = None, loss_mask = None):
        states = self.get_states(states).transpose(1,2)
        return self.decoder(input_ids = input_ids,
                  attention_mask = attention_mask,
                  loss_mask = loss_mask,
                  inputs_ssm_states = states.to(self.ssm_dtype),
                  inputs_ssm_layer = 11,
                  labels = input_ids)
    
    def sample(self, mu, log_var):
        kl_loss = torch.tensor(0, device = mu.device)
        states = mu.clamp(-1, 1)
        if self.training: 
          kl_loss = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
          kl_loss = torch.max(kl_loss, torch.tensor(0.05)).sum() / mu.shape[0]
          states = states + torch.randn_like(states) * torch.exp(0.5 * log_var)
        return states, kl_loss

    def forward(self, **data):
        input_ids, attention_mask, loss_mask = data['full_ids'], data['full_mask'], data['full_loss_mask']
        
        index = torch.rand(input_ids.shape[0]) < 0.5
        attention_mask[index] = data['question_mask'][index]
        loss_mask[index] = data['question_mask'][index]

        mu, log_var = self.encode(input_ids, attention_mask)
        states, kl_loss = self.sample(mu, log_var)
        res = self.decode(states, input_ids, attention_mask, loss_mask)
        return res.loss, kl_loss, res.logits, states




# class MambaVAE(nn.Module):
#     def __init__(self):
#         super().__init__()
#         config = MambaConfig.from_pretrained('state-spaces/mamba-130m-hf')
#         self.decoder = MambaForCausalLM.from_pretrained('state-spaces/mamba-130m-hf', config = config)
#         config.num_hidden_layers  = 12
#         self.encoder = MambaModel.from_pretrained('state-spaces/mamba-130m-hf', config = config)

#         # for name, param in self.named_parameters():
#         #   if '23' not in name and '22' not in name and '21' not in name:
#         #     param.requires_grad = False
        
#         self.mlp1 = nn.Linear(16, 1)
#         self.mlp2 = nn.Linear(1, 16)
#         self.get_log_var = nn.Linear(1536, 1536)
#         self.get_mean = nn.Linear(1536, 1536)

#     def encode(self, input_ids, attention_mask = None, hidden_states = None):
#         if hidden_states is None:
#           states = self.encoder(input_ids = input_ids, 
#                               attention_mask = attention_mask, 
#                               output_ssm_last_states = True).ssm_last_states
#         else:
#           states = self.encoder(inputs_embeds = hidden_states, 
#                             attention_mask = attention_mask, 
#                             layer_range = range(9, 12),
#                             output_ssm_last_states = True).ssm_last_states
#         self.ssm_dtype = states.dtype
#         states = self.mlp1(states).transpose(1,2)
#         return self.get_mean(states).transpose(1,2), self.get_log_var(states).transpose(1,2)
    
#     def decode(self, states, input_ids, attention_mask = None, loss_mask = None, hidden_states = None):
#         # states = states.clamp(-1, 1)
#         states = self.mlp2(states)
#         if hidden_states is None:
#           return self.decoder(input_ids = input_ids,
#                               attention_mask = attention_mask,
#                               loss_mask = loss_mask,
#                               inputs_ssm_states = states.to(self.ssm_dtype),
#                               inputs_ssm_layer = 11,
#                               labels = input_ids)
#         else:
#           return self.decoder(inputs_embeds = hidden_states, 
#                             layer_range = range(21, 24),
#                             attention_mask = attention_mask,
#                             loss_mask = loss_mask,
#                             inputs_ssm_states = states,
#                             labels = input_ids)
    
#     def sample(self, mu, log_var):
#         kl_loss = torch.tensor(0, device = mu.device)
#         states = mu
#         if self.training: 
#           kl_loss = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
#           kl_loss = torch.max(kl_loss, torch.tensor(0.05)).sum() / mu.shape[0]
#           states = states + torch.randn_like(states) * torch.exp(0.5 * log_var)
#         return states, kl_loss

#     def forward(self, **data):
#         input_ids, attention_mask, loss_mask = data['full_ids'], data['full_mask'], data['full_loss_mask']
#         # hidden_states = self.encoder(input_ids = input_ids, 
#         #                     attention_mask = attention_mask, 
#         #                     layer_range = range(21))[0]
#         mu, log_var = self.encode(input_ids, attention_mask)
#         states, kl_loss = self.sample(mu, log_var)
#         res = self.decode(states, input_ids, attention_mask, loss_mask)
#         return res.loss, kl_loss, res.logits, states