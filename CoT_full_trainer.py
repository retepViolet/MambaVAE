import torch, transformers, os
from transformers import TrainingArguments, AutoTokenizer, default_data_collator
from InitMamba import MambaForCausalLM
from datasets import load_dataset


model = MambaForCausalLM.from_pretrained('state-spaces/mamba-130m-hf').cuda()
dataset = load_dataset("arrow", data_files = './cache/CoT_full.arrow', split = 'train').select(range(100000))
print(dataset)
tot = len(dataset)
eval_size= int(tot * 0.05)
train_dataset = dataset.select(range(eval_size, tot))
eval_dataset = dataset.select(range(eval_size))


training_args = TrainingArguments(
    learning_rate = 6e-4,
    warmup_steps = 100,
    num_train_epochs = 2,
    logging_steps = 100,
    ###
    per_device_train_batch_size = 32,
    per_device_eval_batch_size = 32,
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
    label_names = ["input_ids"],
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
trainer.train() # resume_from_checkpoint = './results/result4'