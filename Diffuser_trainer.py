import torch
from datasets import load_from_disk
from Dataset import tokenizer
from Diffuser import Diffuser
import transformers
from transformers import TrainingArguments


class Trainer(transformers.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mean_logits_loss = 0
        self.current_step = 0
        self.mean_mse_loss = 0

    def compute_loss(self, model, inputs, return_outputs = False, num_items_in_batch = None):
        outputs = model(**inputs)
        mse_loss, logits_loss = outputs[0], outputs[1]
        total_loss =  mse_loss + logits_loss * 0
        # 记录
        self.mean_mse_loss += mse_loss.detach() / self.args.logging_steps
        self.mean_logits_loss += logits_loss.detach() / self.args.logging_steps
        return (total_loss, outputs) if return_outputs else total_loss

    def training_step(self, model, inputs, num_items_in_batch = None):
        loss = super().training_step(model, inputs)
        self.current_step += 1
        if self.current_step % self.args.logging_steps == 0:
            self.log({
              "----> mse_loss": self.mean_mse_loss.item(),
              "logits_loss": self.mean_logits_loss.item(),
            })
            self.mean_mse_loss = 0
            self.mean_logits_loss = 0
        return loss


if __name__ == '__main__':
    dataset = load_from_disk("./data/image")
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
    trainer = transformers.Trainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset
    )
    trainer.train()

    # model.eval()
    # for i in range(5):
    #   input_ids1 = torch.tensor(train_dataset[i]['input_ids1']).cuda().unsqueeze(0)
    #   attention_mask1 = torch.tensor(train_dataset[i]['attention_mask1']).cuda().unsqueeze(0)
    #   input_ids2 = torch.tensor(train_dataset[i]['input_ids2']).cuda().unsqueeze(0)
    #   attention_mask2 = torch.tensor(train_dataset[i]['attention_mask2']).cuda().unsqueeze(0)
    #   res = model(input_ids1 = input_ids1, attention_mask1 = attention_mask1, 
    #          input_ids2 = input_ids2, attention_mask2 = attention_mask2)
    #   print('-------------------------')
    #   print(res[0])
    #   tokens = model.vae.decoder.generate(**tokenizer('<|startoftext|>', return_tensors='pt').to('cuda'),
    #                       inputs_ssm_states = res[1],
    #                       inputs_ssm_layer = 11,
    #                       max_length = 128
    #                     )
    #   print('pred: ', tokenizer.batch_decode(tokens, skip_special_tokens=True)[0])
    #   print('targ: ', tokenizer.batch_decode(input_ids2, skip_special_tokens=True)[0])
    # print('###################################################')

    # del model.vae
    # torch.save(model.state_dict(), './results/denoiser.pth')