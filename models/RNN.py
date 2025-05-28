import torch
import torch.nn as nn
from InitMamba import MambaForCausalLM


class MambaRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = MambaForCausalLM.from_pretrained('state-spaces/mamba-130m-hf')
        self.tf = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=768,  # Mamba-130M的隐藏维度
                nhead=12,     # 12个注意力头
                dim_feedforward=3072,  # 前馈网络维度
                dropout=0.0,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            ),
            num_layers=1
        )
        self.C = nn.Linear(16, 1, bias = False)
        self.proj_out = nn.Linear(1536, 768, bias = False)
        self.B = nn.Linear(1, 16, bias = False)
        self.proj_in = nn.Linear(768, 1536, bias = False)
        nn.init.zeros_(self.proj_in.weight)

    def generate(self, input_ids, attention_mask):
        states = self.backbone(input_ids, attention_mask, output_ssm_layer = -1).ssm_last_states
        states = self.tf_forward(states)
        batch_size = input_ids.shape[0]
        device = input_ids.device
        tokens = self.backbone.generate(input_ids = torch.full((batch_size, 1), 50279, dtype=torch.long, device=device), 
                                        max_length = 64, inputs_ssm_states = states)
        return tokens

    def tf_forward(self, states):
        states = torch.stack(states, dim = 1)
        for _ in range(12):
            delta = self.C(states).squeeze(-1)
            delta = self.proj_out(delta)
            delta = self.tf(delta)
            delta = self.proj_in(delta)
            delta = self.B(delta.unsqueeze(-1))
            states = states + delta
        return states

    def forward(self, question_ids, question_mask, answer_ids, answer_mask):
        states = self.backbone(question_ids, question_mask, output_ssm_layer = -1).ssm_last_states
        states = self.tf_forward(states)
        res = self.backbone(answer_ids, answer_mask, inputs_ssm_states = states)
        return res


if __name__ == '__main__':
    model = MambaRNN().cuda()
    model(question_ids = torch.randint(0, 10000, (3, 100)).cuda(), question_mask = torch.ones((3, 100)).cuda(), 
          answer_ids = torch.randint(0, 10000, (3, 100)).cuda(), answer_mask = torch.ones((3, 100)).cuda())