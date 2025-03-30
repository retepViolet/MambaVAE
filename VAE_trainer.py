import transformers
from transformers import TrainingArguments
from data.Dataset import get_dataset, tokenizer
from datasets import load_from_disk
import torch, os
from VAE import MambaVAE


class Trainer(transformers.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ###
        self.no_kl_step = 9000
        self.kl_warmup = 4500
        self.max_kl_decay = 1
        self.min_kl_decay = 1e-9
        ###
        self.kl_decay = self.min_kl_decay
        self.mean_logits_loss = 0
        self.current_step = 0
        self.mean_kl_loss = 0

    def compute_loss(self, model, inputs, return_outputs = False, num_items_in_batch = None):
        outputs = model(**inputs)
        logits_loss, kl_loss = outputs[0], outputs[1]

        if self.current_step > self.no_kl_step:
          ratio = min(1, (self.current_step - self.no_kl_step) / self.kl_warmup)
          self.kl_decay =  ratio * (self.max_kl_decay - self.min_kl_decay) + self.min_kl_decay
        
        total_loss = logits_loss + kl_loss * self.kl_decay
        # 记录
        self.mean_kl_loss += kl_loss / self.args.logging_steps
        self.mean_logits_loss += logits_loss / self.args.logging_steps
        return (total_loss, outputs) if return_outputs else total_loss

    def training_step(self, model, inputs, num_items_in_batch = None):
        loss = super().training_step(model, inputs)
        self.current_step += 1
        if self.current_step % self.args.logging_steps == 0:
            self.log({
              "----> logits_loss": self.mean_logits_loss.item(),
              "kl_loss": self.mean_kl_loss.item(),
              "kl_decay": self.kl_decay,
            })
            self.mean_kl_loss = 0
            self.mean_logits_loss = 0
        return loss

    def save_model(self, output_dir=None, _internal_call=False):
        print('---saving model---')
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.model.state_dict(), output_dir+'/model.pth')



if __name__ == '__main__':
    dataset = load_from_disk("./data/CoT3")
    print(dataset)
    tot = len(dataset)
    eval_size= int(tot * 0.05)
    train_dataset = dataset.select(range(eval_size, tot))
    eval_dataset = dataset.select(range(eval_size))
    # train_dataset, eval_dataset = get_dataset()

    model = MambaVAE()
    # model.load_state_dict(torch.load('./results/CoT_vae2/model.pth', weights_only=True), strict=False)
    
    training_args = TrainingArguments(
        learning_rate = 6e-4,
        warmup_steps = 1000,
        num_train_epochs = 1,
        logging_steps = 10,
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
        label_names = ["full_ids", "full_mask", "full_loss_mask", "question_mask"]
    )
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset
    )
    trainer.train()
    # print(trainer.evaluate())

    # final test --------------------
    model.train()
    for i in range(20):
      data = {}
      for key in train_dataset[i].keys():
        data[key] = torch.tensor(train_dataset[i][key], device = 'cuda').unsqueeze(0)
      res = model(**data)
      tokens = res[2].argmax(-1)
      print('-------------------------')
      print('pred: ', tokenizer.batch_decode(tokens, skip_special_tokens=False)[0])
      print('targ: ', tokenizer.batch_decode(data['full_ids'][:,1:], skip_special_tokens=False)[0])
      print('logits_loss: ', res[0].item(), '; kl_loss: ', res[1].item())
    print('###################################################')
    model.eval()
    for i in range(20):
      data = {}
      for key in eval_dataset[i].keys():
        data[key] = torch.tensor(eval_dataset[i][key], device = 'cuda').unsqueeze(0)
      res = model(**data)
      tokens = res[2].argmax(-1)
      print('-------------------------')
      print('pred: ', tokenizer.batch_decode(tokens, skip_special_tokens=False)[0])
      print('targ: ', tokenizer.batch_decode(data['full_ids'][:,1:], skip_special_tokens=False)[0])
      print('logits_loss: ', res[0].item(), '; kl_loss: ', res[1].item())