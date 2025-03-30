import torch, transformers, os
from transformers import TrainingArguments, AutoTokenizer, default_data_collator
from Baseline import Baseline
from datasets import load_from_disk

dataset = load_from_disk("./data/CoT3")
print(dataset)
tot = len(dataset)
eval_size= int(tot * 0.05)
train_dataset = dataset.select(range(eval_size, tot))
eval_dataset = dataset.select(range(eval_size))

model = Baseline().cuda()

training_args = TrainingArguments(
    learning_rate = 6e-4,
    warmup_steps = 1000,
    num_train_epochs = 1,
    logging_steps = 100,
    weight_decay = 0.01,
    ###
    per_device_train_batch_size = 64,
    per_device_eval_batch_size = 64,
    fp16 = True,
    eval_strategy = 'epoch',
    save_strategy = 'epoch',
    load_best_model_at_end = True,
    metric_for_best_model = 'eval_loss',
    greater_is_better = False,
    save_total_limit = 1,
    logging_dir = './logs',
    output_dir = './results',
    report_to = "none",
    max_grad_norm = 1.0,
    label_names = ["full_ids", "full_mask", "full_loss_mask"],
)

class Trainer(transformers.Trainer):
    def save_model(self, output_dir=None, _internal_call=False):
        print('---saving model---')
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.model.state_dict(), output_dir+'/model.pth')

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    data_collator=default_data_collator
)
trainer.train()