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

        self.mlp1 = nn.Linear(16, 8)
        self.mlp2 = nn.Linear(8, 16)
        self.get_log_var = nn.Linear(1536, 512)
        self.get_mean = nn.Linear(1536, 512)
        self.get_states = nn.Linear(512, 1536)

    def encode(self, input_ids, attention_mask = None):
        states = self.encoder(input_ids = input_ids, 
                    attention_mask = attention_mask, 
                    output_ssm_last_states = True).ssm_last_states
        states = self.mlp1(states).transpose(1,2)
        return self.get_mean(states), self.get_log_var(states)
    
    def decode(self, states, input_ids, attention_mask = None, loss_mask = None):
        states = self.mlp2(self.get_states(states).transpose(1,2))
        return self.decoder(input_ids = input_ids,
                  attention_mask = attention_mask,
                  loss_mask = loss_mask,
                  inputs_ssm_states = states,
                  inputs_ssm_layer = 11,
                  labels = input_ids)
    
    def sample(self, mu, log_var):
        kl_loss = torch.tensor(0, device = mu.device)
        states = mu.clamp(-1, 1)
        if self.training: 
          kl_loss = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
          kl_loss = torch.max(kl_loss, torch.tensor(0.05, device = mu.device)).sum() / mu.shape[0]
          states = states + torch.randn_like(states) * torch.exp(0.5 * log_var)
        return states, kl_loss

    def forward(self, **data):
        input_ids, attention_mask, loss_mask = data['full_ids'], data['full_mask'], data['full_loss_mask']
        
        index = torch.rand(input_ids.shape[0]) < 0.5
        input_ids[index] = data['question_ids'][index]
        attention_mask[index] = data['question_mask'][index]
        loss_mask[index] = data['question_mask'][index]

        mu, log_var = self.encode(input_ids, attention_mask)
        states, kl_loss = self.sample(mu, log_var)
        res = self.decode(states, input_ids, attention_mask, loss_mask)
        return res.loss, kl_loss, res.logits, states
