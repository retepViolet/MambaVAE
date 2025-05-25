import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MambaConfig
import importlib, InitMamba
importlib.reload(InitMamba)
from InitMamba import MambaModel, MambaForCausalLM


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim,
                 commitment_cost=0.25, decay=0.99,
                 epsilon=1e-5, pretrained_embeddings=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        if pretrained_embeddings is not None:
            idx = torch.randperm(pretrained_embeddings.size(0))[:num_embeddings]
            self.embedding.weight.data.copy_(pretrained_embeddings[idx])
        else:
            nn.init.xavier_uniform_(self.embedding.weight.data)
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', torch.zeros(num_embeddings, embedding_dim))

    def forward(self, z):
        with torch.no_grad():
            z_flat = z.reshape(-1, self.embedding_dim)
            dist = (
                z_flat.pow(2).sum(1, keepdim=True)
                + self.embedding.weight.pow(2).sum(1)
                - 2 * z_flat @ self.embedding.weight.t()
            )
            codes = torch.argmin(dist, 1)
            if self.training:
                encodings = torch.zeros(z_flat.size(0), self.num_embeddings, device=z.device)
                encodings.scatter_(1, codes.unsqueeze(1), 1)
                self.ema_cluster_size.mul_(self.decay).add_(encodings.sum(0), alpha=1.0 - self.decay)
                dw = encodings.t() @ z_flat
                self.ema_w.mul_(self.decay).add_(dw, alpha = 1.0 - self.decay)
                n = self.ema_cluster_size.sum()
                cluster_size = (
                    (self.ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon)
                ) * n
                self.embedding.weight.data.copy_(self.ema_w / cluster_size.unsqueeze(1))
        z_q = self.embedding(codes).view_as(z)
        vq_loss = self.commitment_cost * F.mse_loss(z_q.detach(), z)
        z_q = z + (z_q - z).detach()
        codes = codes.view(z.shape[0], -1)
        return vq_loss, z_q, codes


class MambaVQVAE(nn.Module):
    def __init__(self, num_embeddings = 4096):
        super().__init__()
        config = MambaConfig.from_pretrained('state-spaces/mamba-130m-hf')
        self.teacher = MambaForCausalLM.from_pretrained('state-spaces/mamba-130m-hf', config=config)
        self.lin_out = nn.Linear(16, 1)
        self.norm = nn.LayerNorm(1536)
        self.lin_in = nn.Linear(1, 16)
        self.student = MambaForCausalLM.from_pretrained('state-spaces/mamba-130m-hf', config=config)
        self.vector_quantizer = VectorQuantizer(num_embeddings, 64)
    
    def generate(self, input_ids, attention_mask):
        question_states = self.student(input_ids, attention_mask, output_ssm_layer = -1).ssm_last_states
        indices = self.student.generate(
                    input_ids = torch.full((input_ids.shape[0], 1), 0, dtype=torch.long, device=input_ids.device),
                    inputs_ssm_states = question_states,
                    max_length = 33
                )
        states = self.vector_quantizer.embedding(indices).reshape(-1, 1536, 16)
        tokens = self.teacher.generate(
                    input_ids = torch.full((indices.shape[0], 1), 50277, dtype=torch.long, device=indices.device),
                    inputs_ssm_states = states,
                    inputs_ssm_layer = 0,
                    max_length = 64
                )
        return tokens

    def teacher_forward(self, question_states, answer_ids, answer_mask):
        states = self.teacher(answer_ids, answer_mask, inputs_ssm_states = question_states, output_ssm_layer = -1).ssm_last_states
        states = torch.stack(states, dim = 1)   
        states = self.lin_out(states).squeeze(-1)
        states = self.norm(states)
        vq_loss, z_q, indices = self.vector_quantizer(states)
        z_q = self.lin_in(z_q.unsqueeze(-1))
        teacher_loss = self.teacher(answer_ids, answer_mask, inputs_ssm_states = z_q).loss
        return teacher_loss, vq_loss, z_q, indices
     
    def forward(self, **data):
        question_states = self.teacher(data['question_ids'], data['question_mask'], output_ssm_layer = -1).ssm_last_states
        teacher_loss, vq_loss, z_q, indices = self.teacher_forward(question_states, data['answer_ids'], data['answer_mask'])

        # student
        # question_states = self.student(data['question_ids'], data['question_mask'], output_ssm_layer = -1).ssm_last_states
        # first_embedding = self.student.embeddings.word_embeddings.weight[0].unsqueeze(0).unsqueeze(0).expand(z_q.shape[0], 1, -1)
        # padded_z_q = torch.cat([first_embedding, z_q], dim=1)
        # padded_indices = torch.cat([torch.zeros((indices.shape[0], 1), dtype=torch.long, device=indices.device), indices], dim=1)
        # student_res = self.student(inputs_embeds = padded_z_q,
        #                             inputs_ssm_states = question_states,
        #                             labels = padded_indices)
        return teacher_loss, torch.tensor(0.0).cuda(), vq_loss, indices


if __name__ == '__main__':
    model = MambaVQVAE().cuda()
    print(model(question_ids=torch.randint(0, 100, (1, 10)).cuda(), question_mask=torch.ones((1, 10)).cuda(),
                answer_ids=torch.randint(0, 100, (1, 10)).cuda(), answer_mask=torch.ones((1, 10)).cuda()))
