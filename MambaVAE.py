import torch
import torch.nn as nn
import Mamba, importlib
importlib.reload(Mamba)
from Mamba import MambaModel, MambaForCausalLM


class MambaVAE(nn.Module):
    def __init__(self, max_length = 128):
        super().__init__()
        self.encoder = MambaModel.from_pretrained('state-spaces/mamba-130m-hf')
        self.decoder = MambaForCausalLM.from_pretrained('state-spaces/mamba-130m-hf')
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.init_wpe(max_length)

    def init_wpe(self, max_length):
        pos = torch.arange(max_length, dtype = torch.float).unsqueeze(1)
        i = torch.arange(768, dtype = torch.float).unsqueeze(0)
        angle_rates = 1 / torch.pow(1e3, (i // 2) * 2 / 768)
        angles = pos * angle_rates
        pos_code = torch.zeros_like(angles)
        pos_code[:, 0::2] = torch.sin(angles[:, 0::2])
        pos_code[:, 1::2] = torch.cos(angles[:, 1::2])
        self.wpe = nn.Parameter(pos_code.unsqueeze(0)/(768**0.5))

    def sample(self, mu, do_sample):
        kl_loss = 0.5 * mu.pow(2).sum()
        if do_sample: mu = mu + torch.randn_like(mu)
        return mu, kl_loss
    
    def get_logits_loss(self, logits, labels, attention_mask):
        logits = logits.view(-1, logits.size(-1))
        labels = labels.to(logits.device).view(-1)
        attention_mask = attention_mask.to(logits.device).view(-1)
        loss = self.loss_fn(logits, labels) * attention_mask
        return loss.sum() / attention_mask.sum()

    def forward(self, input_ids, attention_mask = None, do_sample = True): # (logits_loss, kl_loss, logits, hidden_states)
        states = self.encoder(input_ids = input_ids, 
                              attention_mask = attention_mask, 
                              output_ssm_last_states = True).ssm_last_states
        states, kl_loss = self.sample(states, do_sample) # 采样
        logits = self.decoder(inputs_embeds = self.wpe[0,:input_ids.shape[1]].expand(len(input_ids), -1, -1), 
                              attention_mask = attention_mask,
                              inputs_ssm_states = states).logits
        logits_loss = self.get_logits_loss(logits, input_ids, attention_mask)
        return (logits_loss, kl_loss, logits, states)