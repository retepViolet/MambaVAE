import torch
import torch.nn as nn
import InitMamba, importlib
importlib.reload(InitMamba)
from transformers import MambaConfig
from InitMamba import MambaModel, MambaForCausalLM


class MambaVAE(nn.Module):
    def __init__(self, max_length = 128):
        super().__init__()
        config = MambaConfig.from_pretrained("state-spaces/mamba-130m-hf")
        config.num_hidden_layers = 6
        self.encoder = MambaModel.from_pretrained('state-spaces/mamba-130m-hf', config=config)
        self.decoder = MambaForCausalLM.from_pretrained('state-spaces/mamba-130m-hf', config=config)
        # for param in self.encoder.parameters():
        #   param.requires_grad = False
    #     self.loss_fn = nn.CrossEntropyLoss(reduction='none')
    #     self.init_wpe(max_length)

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
          mu = mu + torch.randn_like(mu)
        return mu, kl_loss

    # def init_wpe(self, max_length):
    #     pos = torch.arange(max_length, dtype = torch.float).unsqueeze(1)
    #     i = torch.arange(768, dtype = torch.float).unsqueeze(0)
    #     angle_rates = 1 / torch.pow(1e3, (i // 2) * 2 / 768)
    #     angles = pos * angle_rates
    #     pos_code = torch.zeros_like(angles)
    #     pos_code[:, 0::2] = torch.sin(angles[:, 0::2])
    #     pos_code[:, 1::2] = torch.cos(angles[:, 1::2])
    #     self.wpe = nn.Parameter(pos_code.unsqueeze(0)/(768**0.5))

    
    # def get_logits_loss(self, logits, labels, attention_mask):
    #     logits = logits.view(-1, logits.size(-1))
    #     labels = labels.view(-1)
    #     attention_mask = attention_mask.view(-1)
    #     loss = self.loss_fn(logits, labels) * attention_mask
    #     return loss.sum() / attention_mask.sum()

    def forward(self, input_ids, attention_mask = None, do_sample = False): # (logits_loss, kl_loss, logits, hidden_states)
        states = self.encode(input_ids, attention_mask)
        states, kl_loss = self.sample(states, do_sample)
        res = self.decode(states, input_ids, attention_mask)
        # inputs_embeds = self.wpe[0,:input_ids.shape[1]].expand(len(input_ids), -1, -1)
        #self.get_logits_loss(logits, input_ids, attention_mask)
        return res.loss, kl_loss, res.logits, states