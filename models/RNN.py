import torch
import torch.nn as nn
from InitMamba import MambaForCausalLM


class MambaRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = MambaForCausalLM.from_pretrained('state-spaces/mamba-130m-hf')

    def generate(self, input_ids, attention_mask):
        states = self.backbone(input_ids, attention_mask, output_ssm_layer = -1).ssm_last_states
        batch_size = input_ids.shape[0]
        device = input_ids.device
        tokens = self.backbone.generate(input_ids = torch.full((batch_size, 1), 50279, dtype=torch.long, device=device), 
                                        max_length = 64, inputs_ssm_states = states)
        return tokens

    def forward(self, question_ids, question_mask, answer_ids, answer_mask):
        states = self.backbone(question_ids, question_mask, output_ssm_layer = -1).ssm_last_states
        res = self.backbone(answer_ids, answer_mask, inputs_ssm_states = states)
        return res


if __name__ == '__main__':
    model = MambaRNN().cuda()
    model(question_ids = torch.randint(0, 10000, (3, 100)).cuda(), question_mask = torch.ones((3, 100)).cuda(), 
          answer_ids = torch.randint(0, 10000, (3, 100)).cuda(), answer_mask = torch.ones((3, 100)).cuda())