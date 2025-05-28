import torch, transformers, os
from transformers import TrainingArguments, AutoTokenizer
from RNN import MambaRNN
from datasets import load_from_disk
from Dataset import tokenizer
from tqdm import tqdm


class Trainer(transformers.Trainer):
    def calculate_em_score(self, pred_tokens, target_tokens):
        pred_clean = pred_tokens[pred_tokens != 0]
        target_clean = target_tokens[target_tokens != 0]
        # print("预测输出:", tokenizer.decode(pred_clean, skip_special_tokens=True))
        # print('--------------------------------')
        # print("目标输出:", tokenizer.decode(target_clean, skip_special_tokens=True))
        # print('================================')
        return 1.0 if len(pred_clean) == len(target_clean) and torch.equal(pred_clean, target_clean) else 0.0

    @torch.no_grad()
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        if eval_dataset is None: 
            eval_dataset = self.eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        self.model.eval()
        
        total_loss = total_em_score = num_samples = 0
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            # 计算训练损失
            outputs = self.model(**batch)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs
            total_loss += loss.item()
            if num_samples < 100 * batch['question_ids'].shape[0]:
                tokens = self.model.generate(batch['question_ids'], batch['question_mask'])
                # 计算EM分数
                for i in range(batch['question_ids'].shape[0]):
                    total_em_score += self.calculate_em_score(tokens[i], batch['answer_ids'][i])
                    num_samples += 1
        # 计算平均值
        num_batches = len(eval_dataloader)
        avg_loss = total_loss / num_batches
        avg_em_score = total_em_score / num_samples if num_samples > 0 else 0.0
        
        self.model.train()
        print(f'loss: {avg_loss:.4f}; exact_match: {avg_em_score:.4f}')
        
        return {
            f"{metric_key_prefix}_loss": avg_loss,
            f"{metric_key_prefix}_exact_match": avg_em_score,
        }

    def save_model(self, output_dir=None, _internal_call=False):
        print('---saving model---')
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.model.state_dict(), output_dir+'/model.pth')


if __name__ == '__main__':
    dataset = load_from_disk("./data/CoT3").select(range(1000))
    print(dataset)
    tot = len(dataset)
    eval_size = int(tot * 0.05)
    train_dataset = dataset.select(range(eval_size, tot))
    eval_dataset = dataset.select(range(eval_size))

    model = MambaRNN()

    training_args = TrainingArguments(
        learning_rate = 1e-4,
        warmup_steps = 100,
        num_train_epochs = 1,
        logging_steps = 100,
        # weight_decay = 0.01,
        ###
        per_device_train_batch_size = 64,
        per_device_eval_batch_size = 64,
        dataloader_num_workers = 16,
        bf16 = True,
        eval_strategy = 'epoch',
        save_strategy = 'epoch',
        load_best_model_at_end = True,
        metric_for_best_model = 'eval_loss',
        greater_is_better = False,
        save_total_limit = 1,
        logging_dir = './logs',
        output_dir = './results/rnn',
        report_to = "none",
        max_grad_norm = 1.0,
        label_names = ['question_ids', 'question_mask', 'answer_ids', 'answer_mask'],
    )

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
    )
    trainer.train()
    # print(trainer.evaluate())