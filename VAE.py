import torch
import torch.nn as nn
from transformers import MambaConfig

import importlib, InitMamba
importlib.reload(InitMamba)
from InitMamba import MambaModel, MambaForCausalLM


class MambaVAE(nn.Module):
    def __init__(self, max_length = 128):
        super().__init__()
        config = MambaConfig.from_pretrained("state-spaces/mamba-130m-hf")
        # config.num_hidden_layers = 12
        self.encoder = MambaModel.from_pretrained('state-spaces/mamba-130m-hf', config=config)
        self.decoder = MambaForCausalLM.from_pretrained('state-spaces/mamba-130m-hf', config=config)
        for name, param in self.named_parameters():
          if '23' not in name and '22' not in name and '21' not in name:
            param.requires_grad = False

    def encode(self, input_ids, attention_mask = None):
        return self.encoder(input_ids = input_ids, 
                            attention_mask = attention_mask, 
                            output_ssm_last_states = True).ssm_last_states
    
    def decode(self, states, input_ids, attention_mask = None):
        return self.decoder(input_ids = input_ids,
                            attention_mask = attention_mask,
                            inputs_ssm_states = states,
                            labels = input_ids)
    
    def sample(self, mu, do_sample):
        kl_loss = torch.tensor(0, device = mu.device)
        if do_sample or self.training: 
          kl_loss = 0.5 * mu.pow(2).sum()
          mu = mu + torch.randn_like(mu) * 0.1
        return mu, kl_loss / mu.shape[0]

    # (logits_loss, kl_loss, logits, hidden_states)
    def forward(self, input_ids, attention_mask = None, do_sample = False):
        hidden_states = self.encoder(input_ids = input_ids, 
                            attention_mask = attention_mask, 
                            layer_range = range(21))[0]
        states = self.encoder(inputs_embeds = hidden_states, 
                            attention_mask = attention_mask, 
                            layer_range = range(21, 24),
                            output_ssm_last_states = True).ssm_last_states
        # states = self.encode(input_ids, attention_mask)
        states, kl_loss = self.sample(states, do_sample)
        # res = self.decode(states, input_ids, attention_mask)
        res = self.decoder(inputs_embeds = hidden_states, 
                            layer_range = range(21, 24),
                            attention_mask = attention_mask,
                            inputs_ssm_states = states,
                            labels = input_ids)
        return res.loss, kl_loss, res.logits, states