import transformers
from transformers import TrainingArguments
from Dataset import tokenizer
from datasets import load_from_disk
import torch, os
from VQ_VAE import MambaVQVAE
import random
from tqdm import tqdm


class Trainer(transformers.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mean_teacher_loss = 0
        self.mean_student_loss = 0
        self.mean_vq_loss = 0
        self.current_step = 0
        self.all_indices = []

    def calculate_token_match(self, pred, target):
        answer_pos = (pred == 50279).nonzero(as_tuple=True)[0]
        if len(answer_pos) == 0:
            return 0.0
        target_set = set(target[1:].tolist())
        target_set.discard(0)
        pred_set = set(pred[answer_pos[0]+1:].tolist())
        pred_set.discard(0)
        matched = len(target_set.intersection(pred_set))
        if len(target_set) == 0:
            return 0.0
        return matched / len(target_set)

    # @torch.no_grad()
    # def evaluate(self, eval_dataset = None, ignore_keys = None, metric_key_prefix: str = "eval_accuracy"):
    #     if eval_dataset is None: eval_dataset = self.eval_dataset
    #     # random_indices = random.sample(range(len(eval_dataset)), min(128, len(eval_dataset)))
    #     eval_dataloader = self.get_eval_dataloader(eval_dataset)
    #     self.model.eval()
    #     acc = total_logits_loss = total_vq_loss = num_batches = 0
    #     for batch in tqdm(eval_dataloader, desc="Evaluating"):
    #         batch = {k: v.to(self.args.device) for k, v in batch.items()}
    #         logits_loss, vq_loss, _, z_q, _ = self.model(**batch)
    #         total_logits_loss += logits_loss.item()
    #         total_vq_loss += vq_loss.item()
    #         num_batches += 1
    #         if num_batches <= 10:
    #             batch_size = batch['question_ids'].shape[0]
    #             valid_lengths = batch['question_mask'].sum(dim=1)
    #             for i in range(batch_size):
    #                 input_ids = batch['question_ids'][i:i+1, :valid_lengths[i]]
    #                 tokens = self.model.decoder.generate(
    #                     input_ids=input_ids,
    #                     inputs_ssm_states=z_q[i:i+1].reshape(-1, 1536, 16),
    #                     max_length=256
    #                 )
    #                 acc += self.calculate_token_match(tokens[0], batch['answer_ids'][i]) / batch_size * 10
    #                 # if i < 1:
    #                 #     print('=========================')
    #                 #     print('pred: ', tokenizer.decode(tokens[0], skip_special_tokens=True))
    #                 #     print('-------------------------')
    #                 #     print('targ: ', tokenizer.decode(batch['full_ids'][i], skip_special_tokens=True))

    #     avg_logits_loss = total_logits_loss / num_batches
    #     avg_vq_loss = total_vq_loss / num_batches
        
    #     self.model.train()
    #     print(f'logits_loss: {avg_logits_loss:.4f}; vq_loss: {avg_vq_loss:.4f}; token_match: {acc:.4f}')
    #     return {
    #         "eval_accuracy": acc,
    #         "eval_loss": avg_logits_loss,
    #     }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # ratio = 1 / (300 - min(self.current_step, 299))
        outputs = model(**inputs)
        teacher_loss, student_loss, vq_loss, indices = outputs
        total_loss = teacher_loss + student_loss + vq_loss
        # 收集所有indices
        self.all_indices.append(indices.detach())
        self.mean_vq_loss += vq_loss.item() / self.args.logging_steps
        self.mean_teacher_loss += teacher_loss.item() / self.args.logging_steps
        self.mean_student_loss += student_loss.item() / self.args.logging_steps
        return (total_loss, outputs) if return_outputs else total_loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs)
        self.current_step += 1
        if self.current_step % self.args.logging_steps == 0:
            # 计算总的unique indices数量
            all_indices = torch.cat(self.all_indices, dim=0)
            unique_indices = torch.unique(all_indices).numel()
            self.log({
                "----> teacher_loss": self.mean_teacher_loss,
                "student_loss": self.mean_student_loss,
                "vq_loss": self.mean_vq_loss,
                "usage_rate": unique_indices / model.vector_quantizer.num_embeddings,
            })
            self.mean_vq_loss = 0
            self.mean_teacher_loss = 0
            self.mean_student_loss = 0
            self.all_indices = []
        return loss

    def save_model(self, output_dir=None, _internal_call=False):
        print('---saving model---')
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.model.state_dict(), output_dir+'/model.pth')


if __name__ == '__main__':
    dataset = load_from_disk("./data/CoT3").select(range(10000))
    print(dataset)
    tot = len(dataset)
    eval_size = int(tot * 0.05)
    train_dataset = dataset.select(range(eval_size, tot))
    eval_dataset = dataset.select(range(eval_size))

    model = MambaVQVAE()
    # model.load_state_dict(torch.load('./results/vqvae/model.pth'))
    training_args = TrainingArguments(
        learning_rate = 4e-4,
        warmup_steps = 100,
        num_train_epochs = 1,
        logging_steps = 10,
        # weight_decay = 0.01,
        per_device_train_batch_size = 64,
        per_device_eval_batch_size = 64,
        dataloader_num_workers = 16,
        bf16 = True,
        eval_strategy = 'epoch',
        save_strategy = 'epoch',
        save_total_limit = 1,
        load_best_model_at_end = True,
        metric_for_best_model = 'loss',
        greater_is_better = False,
        output_dir = './results',
        report_to = "none",
        max_grad_norm = 1.0,
        label_names = ["question_mask", "question_ids", "answer_ids", "answer_mask"],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    trainer.train()
    # print(trainer.evaluate())

    # final test --------------------
    # model.train()
    # for i in range(3):
    #     data = {}
    #     for key in train_dataset[i].keys():
    #         data[key] = torch.tensor(train_dataset[i][key], device='cuda').unsqueeze(0)
    #     res = model(**data)
    #     tokens = res[2].argmax(-1)
    #     print('=========================')
    #     print('logits_loss: ', res[0].item(), '; vq_loss: ', res[1].item())
    #     print('pred: ', tokenizer.batch_decode(tokens, skip_special_tokens=True)[0])
    #     print('-------------------------')
    #     print('targ: ', tokenizer.batch_decode(data['full_ids'][:,1:], skip_special_tokens=True)[0])
    # print('###################################################')
    # model.eval()
    # for i in range(3):
    #     data = {}
    #     for key in eval_dataset[i].keys():
    #         data[key] = torch.tensor(eval_dataset[i][key], device='cuda').unsqueeze(0)
    #     res = model(**data)
    #     tokens = res[2].argmax(-1)
    #     print('=========================')
    #     print('logits_loss: ', res[0].item(), '; vq_loss: ', res[1].item())
    #     print('pred: ', tokenizer.batch_decode(tokens, skip_special_tokens=True)[0])
    #     print('-------------------------')
    #     print('targ: ', tokenizer.batch_decode(data['full_ids'][:,1:], skip_special_tokens=True)[0])
