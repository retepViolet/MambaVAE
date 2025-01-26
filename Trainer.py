import transformers
from transformers import TrainingArguments
from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer
from Dataset import get_dataset, tokenizer
import torch, os
from VAE import MambaVAE
import InitMamba


class Trainer(transformers.Trainer):
    def __init__(self, *args, kl_warmup = 300, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_step = 0
        self.kl_warmup = kl_warmup
        self.mean_kl_loss = 0
        self.kl_decay = 0
        self.mean_logits_loss = 0

    def compute_loss(self, model, inputs, return_outputs = False, num_items_in_batch = None):
        outputs = model(**inputs)
        logits_loss, kl_loss = outputs[0], outputs[1]
        self.kl_decay = min(1, self.current_step / self.kl_warmup) * 1e-5
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
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.model.state_dict(), output_dir+'/model.pth')


if __name__ == '__main__':
    # config = MambaConfig.from_pretrained("state-spaces/mamba-130m-hf")
    # config.use_mambapy = True
    model = MambaVAE()
    # model = MambaForCausalLM.from_pretrained('state-spaces/mamba-130m-hf')
    train_dataset, eval_dataset = get_dataset(1000)

    training_args = TrainingArguments(
        output_dir = './results',
        learning_rate = 1e-3,
        warmup_steps = 10,
        num_train_epochs = 10,
        per_device_train_batch_size = 32,
        per_device_eval_batch_size = 16,
        fp16 = True,
        eval_strategy = 'epoch',
        save_strategy = 'epoch',
        load_best_model_at_end = True,
        metric_for_best_model = 'eval_loss',
        greater_is_better = False,
        save_total_limit = 1,
        logging_dir = './logs',
        logging_steps = 29,
        report_to = "none",
        max_grad_norm = 1.0,
        label_names = ["input_ids"],
        weight_decay = 0.1
    )

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset
    )
    trainer.train()


    # final test --------------------

    model.eval()
    for i in range(5):
      input_ids = torch.tensor(train_dataset[i]['input_ids']).cuda().unsqueeze(0)
      attention_mask = torch.tensor(train_dataset[i]['attention_mask']).cuda().unsqueeze(0)
      tokens = model(input_ids, attention_mask)[2].argmax(-1)
      print('-------------------------')
      print('pred: ', tokenizer.batch_decode(tokens, skip_special_tokens=True)[0])
      print('targ: ', tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0])
    print('###################################################')
    for i in range(5):
      input_ids = torch.tensor(eval_dataset[i]['input_ids']).cuda().unsqueeze(0)
      attention_mask = torch.tensor(eval_dataset[i]['attention_mask']).cuda().unsqueeze(0)
      tokens = model(input_ids, attention_mask)[2].argmax(-1)
      print('-------------------------')
      print('pred: ', tokenizer.batch_decode(tokens, skip_special_tokens=True)[0])
      print('targ: ', tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0])