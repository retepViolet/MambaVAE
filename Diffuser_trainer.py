import torch
from datasets import load_from_disk
from Dataset import tokenizer
from Diffuser import Diffuser
import transformers
from transformers import TrainingArguments
import random


class Trainer(transformers.Trainer):
    def evaluate(self, eval_dataset = None, ignore_keys = None, metric_key_prefix: str = "eval"):
        # 获取 dataloader
        random_indices = random.sample(range(len(eval_dataset)), 1280)
        eval_dataloader = self.get_eval_dataloader(eval_dataset.select(random_indices))

        self.model.eval()
        avg_loss = 0

        for batch in eval_dataloader:
            # 放到 device 上
            batch = {k: v.to(self.args.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model.generate(batch['condition'])
            loss = torch.nn.functional.mse_loss(outputs, batch['target'])
            avg_loss += loss / len(eval_dataloader)

        self.model.train()
        return {f"{metric_key_prefix}_loss": avg_loss.item()}


if __name__ == '__main__':
    dataset = load_from_disk("./data/image").select(range(10000))
    print(dataset)
    tot = len(dataset)
    eval_size= int(tot * 0.05)
    train_dataset = dataset.select(range(eval_size, tot))
    eval_dataset = dataset.select(range(eval_size))

    model = Diffuser()
    training_args = TrainingArguments(
        learning_rate = 2e-4,
        lr_scheduler_type = 'cosine',
        warmup_steps = 1000,
        num_train_epochs = 20,
        logging_steps = 1000,
        ###
        per_device_train_batch_size = 128,
        per_device_eval_batch_size = 128,
        dataloader_num_workers = 16,
        bf16 = True,
        eval_strategy = 'epoch', 
        save_strategy = 'epoch',
        save_total_limit = 1,
        load_best_model_at_end = True,
        metric_for_best_model = 'eval_loss',
        greater_is_better = False,
        output_dir = './results',
        report_to = "none",
        max_grad_norm = 1.0,
        label_names = ['target', 'condition'],
    )
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset
    )
    trainer.train()