import torch
import torch.nn as nn
from InitMamba import MambaForCausalLM


class MambaRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = MambaForCausalLM.from_pretrained('state-spaces/mamba-130m-hf')

    def forward(self, question_ids, question_mask, answer_ids, answer_mask):
        res1 = self.backbone(question_ids, question_mask, output_ssm_layer = -1)
        res = self.backbone(question_ids[:, :99], question_mask[:, :99], output_ssm_layer = -1, use_cache = True)
        res2 = self.backbone(question_ids[:, 99:], question_mask[:, 99:], output_ssm_layer = -1, inputs_ssm_states = res.ssm_last_states,
                               use_cache = False, cache_params = res.cache_params, 
                               cache_position = torch.full((question_ids.size(0),), 1, device='cuda'))
        print((torch.cat([res.logits, res2.logits], dim = 1) - res1.logits).abs().sum())
        # self.backbone.generate(question_ids, max_new_tokens = 128, inputs_ssm_states = states,
        #                        use_cache = True, cache_params = res.cache_params, 
        #                        cache_position = torch.full((question_ids.size(0),), 1, device='cuda'))
        return res


if __name__ == '__main__':
    model = MambaRNN().cuda()
    model(question_ids = torch.randint(0, 10000, (3, 100)).cuda(), question_mask = torch.ones((3, 100)).cuda(), 
          answer_ids = torch.randint(0, 10000, (3, 100)).cuda(), answer_mask = torch.ones((3, 100)).cuda())