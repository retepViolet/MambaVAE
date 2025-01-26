import torch
import torch.nn as nn
from transformers import MambaConfig
from InitMamba import MambaModel, MambaForCausalLM


class MambaVAE(nn.Module):
    def __init__(self, max_length = 128):
        super().__init__()
        config = MambaConfig.from_pretrained("state-spaces/mamba-130m-hf")
        config.num_hidden_layers = 12
        self.encoder = MambaModel.from_pretrained('state-spaces/mamba-130m-hf', config=config)
        self.decoder = MambaForCausalLM.from_pretrained('state-spaces/mamba-130m-hf', config=config)
        # for name, param in self.decoder.named_parameters():
        #   if '23' not in name:
        #     param.requires_grad = False

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
        kl_loss = 0
        if do_sample or self.training: 
          kl_loss = 0.5 * mu.pow(2).sum()
          mu = mu + torch.randn_like(mu) * 0.5
        return mu, kl_loss

    # (logits_loss, kl_loss, logits, hidden_states)
    def forward(self, input_ids, attention_mask = None, do_sample = False):
        states = self.encode(input_ids, attention_mask)
        states, kl_loss = self.sample(states, do_sample)
        res = self.decode(states, input_ids, attention_mask)
        return res.loss, kl_loss / input_ids.shape[0], res.logits, states