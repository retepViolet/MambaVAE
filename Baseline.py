import torch
import torch.nn as nn
# from InitMamba import MambaForCausalLM
from transformers import MambaForCausalLM


class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = MambaForCausalLM.from_pretrained('state-spaces/mamba-130m-hf')

    def forward(self, **data):
        input_ids, attention_mask, loss_mask = data['full_ids'], data['full_mask'], data['full_loss_mask']
        res = self.backbone(input_ids = input_ids[:,:8], 
                            attention_mask = attention_mask[:,:8],)
                            # loss_mask = loss_mask)
        return res