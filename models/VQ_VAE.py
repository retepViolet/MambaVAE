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
        self.vector_quantizer = VectorQuantizer(num_embeddings, 64)
        self.teacher = MambaForCausalLM.from_pretrained('state-spaces/mamba-130m-hf')
        self.lin_out = nn.Linear(16, 1)
        self.norm = nn.LayerNorm(1536)
        self.lin_in = nn.Linear(1, 16)
        self.student = MambaModel.from_pretrained('state-spaces/mamba-130m-hf')
        self.output_layer = nn.Linear(64, num_embeddings)
        self.first_embedding = nn.Parameter(torch.randn(1, 1, 768))
    
    @torch.no_grad()
    def generate(self, input_ids, attention_mask, answer_ids, answer_mask):
        question_states = self.teacher(input_ids, attention_mask, output_ssm_layer = -1).ssm_last_states
        question_states = torch.stack(question_states, dim = 1)
        
        batch_size, device = input_ids.shape[0], input_ids.device
        cache_params = None
        cache_position = torch.zeros((batch_size,), dtype=torch.long, device=device)
        all_indices = []
        
        # for step in range(48):
        #     if step == 0:
        #         inputs_embeds = self.first_embedding.expand(batch_size, -1, -1)
        #         output = self.student(inputs_embeds=inputs_embeds, inputs_ssm_states=question_states, use_cache=True)
        #         cache_params = output.cache_params
        #     else:
        #         last_vq_embeds = self.vector_quantizer.embedding(all_indices[-1])
        #         inputs_embeds = last_vq_embeds.reshape(batch_size, 1, 768)
        #         output = self.student(inputs_embeds=inputs_embeds, use_cache=True, 
        #                             cache_params=cache_params, cache_position=cache_position.clone())
            
        #     cache_position += 1
        #     hidden_reshaped = output.last_hidden_state.reshape(batch_size, 12, 64)
        #     logits = self.output_layer(hidden_reshaped)
        #     all_indices.append(torch.argmax(logits, dim=-1))
        
        # # student生成的indices
        # indices_student = torch.cat(all_indices, dim=1)  # [batch, 48*12=576]
        
        # 使用teacher_forward得到正确的indices作为对比
        _, _, z_q_teacher, indices_teacher = self.teacher_forward(question_states, answer_ids, answer_mask)
        
        # 随机修改50%的indices
        mask = torch.rand_like(indices_teacher.float()) < 0.5
        random_indices = torch.randint(0, self.vector_quantizer.num_embeddings, indices_teacher.shape, device=device)
        indices_teacher = torch.where(mask, random_indices, indices_teacher)
        
        z_q = self.vector_quantizer.embedding(indices_teacher).reshape(batch_size, 24, 1536)  # [batch, 24, 1536]
        states = self.lin_in(z_q.unsqueeze(-1))  # [batch, 24, 1536, 16]
        # print((indices_teacher == indices_student).float().mean())
        
        tokens = self.teacher.generate(
                    input_ids = torch.full((batch_size, 1), 50279, dtype=torch.long, device=device),
                    inputs_ssm_states = states + question_states, max_length = 64)
        return tokens

    def teacher_forward(self, question_states, answer_ids, answer_mask):
        states = self.teacher(answer_ids, answer_mask, inputs_ssm_states = question_states, output_ssm_layer = -1).ssm_last_states
        states = torch.stack(states, dim = 1)   
        states = self.lin_out(states).squeeze(-1)
        states = self.norm(states)
        vq_loss, z_q, indices = self.vector_quantizer(states)
        states = self.lin_in(z_q.unsqueeze(-1))
        teacher_loss = self.teacher(answer_ids, answer_mask, inputs_ssm_states = states + question_states).loss
        return teacher_loss, vq_loss, z_q, indices

    def student_forward(self, question_states, z_q, indices):
        inputs_embeds = z_q.reshape(-1, 48, 768)
        inputs_embeds = torch.cat([self.first_embedding.expand(inputs_embeds.shape[0], -1, -1), 
                                   inputs_embeds[:, :-1]], dim=1)
        embeds = self.student(inputs_embeds = inputs_embeds, inputs_ssm_states = question_states).last_hidden_state
        embeds = embeds.reshape(-1, 48 * 12, self.vector_quantizer.embedding_dim)
        logits = self.output_layer(embeds)
        student_loss = F.cross_entropy(logits.reshape(-1, self.vector_quantizer.num_embeddings), indices.reshape(-1))
        return student_loss
     
    def forward(self, **data):
        # teacher
        question_states = self.teacher(data['question_ids'], data['question_mask'], output_ssm_layer = -1).ssm_last_states
        question_states = torch.stack(question_states, dim = 1)
        teacher_loss, vq_loss, z_q, indices = self.teacher_forward(question_states, data['answer_ids'], data['answer_mask'])
        # student
        student_loss = self.student_forward(question_states, z_q, indices)
        return teacher_loss, student_loss, vq_loss, indices


if __name__ == '__main__':
    model = MambaVQVAE().cuda()
    print(model(question_ids=torch.randint(0, 100, (1, 10)).cuda(), question_mask=torch.ones((1, 10)).cuda(),
                answer_ids=torch.randint(0, 100, (1, 10)).cuda(), answer_mask=torch.ones((1, 10)).cuda()))
