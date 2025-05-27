import torch
import torch.nn as nn
from InitMamba import MambaForCausalLM


class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = MambaForCausalLM.from_pretrained('state-spaces/mamba-130m-hf')

    def forward(self, **data):
        question_states = self.backbone(data['question_ids'], data['question_mask'], output_ssm_layer = -1)
        res = self.backbone(data['answer_ids'], data['answer_mask'])
        return (res.loss + question_states.loss, )