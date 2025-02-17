import torch
import torch.nn as nn
from transformers import MambaConfig

import importlib, InitMamba
importlib.reload(InitMamba)
from InitMamba import MambaModel, MambaForCausalLM


class MambaVAE(nn.Module):
    def __init__(self):
        super().__init__()
        config = MambaConfig.from_pretrained('state-spaces/mamba-130m-hf')
        config.num_hidden_layers  = 12
        self.encoder = MambaModel.from_pretrained('state-spaces/mamba-130m-hf', config=config)
        self.decoder = MambaForCausalLM.from_pretrained('state-spaces/mamba-130m-hf')
        # for name, param in self.named_parameters():
        #   if '23' not in name and '22' not in name and '21' not in name:
        #     param.requires_grad = False

    def encode(self, input_ids, attention_mask = None):
        return self.encoder(input_ids = input_ids, 
                            attention_mask = attention_mask, 
                            output_ssm_last_states = True).ssm_last_states
    
    def decode(self, states, input_ids, attention_mask = None):
        return self.decoder(input_ids = self.add_rand_token(input_ids) if self.training else input_ids,
                            attention_mask = attention_mask,
                            inputs_ssm_states = states,
                            labels = input_ids)
    
    def sample(self, mu):
        kl_loss = torch.tensor(0, device = mu.device)
        if self.training: 
          kl_loss = 0.5 * mu.pow(2).sum()
          mu = mu + torch.randn_like(mu)
        return mu, kl_loss / mu.shape[0]
    
    def add_rand_token(self, input_ids):
        batch_size = input_ids.shape[0]
        length = input_ids.shape[1]
        pos = (torch.rand(length//20, batch_size, device=input_ids.device) * length).long()
        rand_token = (torch.rand(length//20, batch_size, device=input_ids.device) * 50276 + 1).long()
        new_input_ids = input_ids.clone()
        new_input_ids[torch.arange(batch_size),pos] = rand_token
        return new_input_ids

    # (logits_loss, kl_loss, logits, hidden_states)
    def forward(self, input_ids, attention_mask = None):
        # hidden_states = self.encoder(input_ids = input_ids, 
        #                     attention_mask = attention_mask, 
        #                     layer_range = range(21))[0]
        # states = self.encoder(inputs_embeds = hidden_states, 
        #                     attention_mask = attention_mask, 
        #                     layer_range = range(21, 24),
        #                     output_ssm_last_states = True).ssm_last_states
        states = self.encode(input_ids, attention_mask)
        states, kl_loss = self.sample(states)
        res = self.decode(states, input_ids, attention_mask)
        # res = self.decoder(inputs_embeds = hidden_states, 
        #                     layer_range = range(21, 24),
        #                     attention_mask = attention_mask,
        #                     inputs_ssm_states = states,
        #                     labels = input_ids)
        return res.loss, kl_loss, res.logits, states