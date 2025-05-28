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

    def calculate_em_score(self, pred_tokens, target_tokens):
        pred_clean = pred_tokens[pred_tokens != 0]
        target_clean = target_tokens[target_tokens != 0]
        # print("预测输出:", tokenizer.decode(pred_clean, skip_special_tokens=True))
        # print('--------------------------------')
        # print("目标输出:", tokenizer.decode(target_clean, skip_special_tokens=True))
        # print('================================')
        return 1.0 if len(pred_clean) == len(target_clean) and torch.equal(pred_clean, target_clean) else 0.0

    @torch.no_grad()
    def evaluate(self, eval_dataset = None, ignore_keys = None, metric_key_prefix: str = "eval"):
        if eval_dataset is None: 
            eval_dataset = self.eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        self.model.eval()
        
        total_teacher_loss = total_student_loss = total_vq_loss = total_em_score = num_samples = 0
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = {k: v.to(self.args.device) for k, v in batch.items()}
            # 计算训练损失
            teacher_loss, student_loss, vq_loss, _ = self.model(**batch)
            total_teacher_loss += teacher_loss.item()
            total_student_loss += student_loss.item()
            total_vq_loss += vq_loss.item()
            # batch生成评估
            if num_samples < 10 * batch['question_ids'].shape[0]:
                input_ids = batch['question_ids']
                attention_mask = batch['question_mask']
                answer_ids = batch['answer_ids']
                answer_mask = batch['answer_mask']
                tokens = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, answer_ids=answer_ids, answer_mask=answer_mask)
                # 计算EM分数
                for i in range(batch['question_ids'].shape[0]):
                    total_em_score += self.calculate_em_score(tokens[i], batch['answer_ids'][i])
                    num_samples += 1
        # 计算平均值
        num_batches = len(eval_dataloader)
        avg_teacher_loss = total_teacher_loss / num_batches
        avg_student_loss = total_student_loss / num_batches
        avg_vq_loss = total_vq_loss / num_batches
        avg_em_score = total_em_score / num_samples if num_samples > 0 else 0.0
        
        self.model.train()
        print(f'teacher_loss: {avg_teacher_loss:.4f}; student_loss: {avg_student_loss:.4f}; '
              f'vq_loss: {avg_vq_loss:.4f}; exact_match: {avg_em_score:.4f}')
        
        return {
            f"{metric_key_prefix}_teacher_loss": avg_teacher_loss,
            f"{metric_key_prefix}_student_loss": avg_student_loss,
            f"{metric_key_prefix}_vq_loss": avg_vq_loss,
            f"{metric_key_prefix}_exact_match": avg_em_score,
            f"{metric_key_prefix}_loss": avg_teacher_loss + avg_student_loss + avg_vq_loss,
        }

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
    model.load_state_dict(torch.load('./results/vqCoT/checkpoint-18496/model.pth'))
    training_args = TrainingArguments(
        learning_rate = 4e-4,
        warmup_steps = 100,
        num_train_epochs = 1,
        logging_steps = 100,
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
        output_dir = './results/vqCoT',
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
    # trainer.train()
    print(trainer.evaluate())