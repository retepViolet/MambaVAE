import transformers
from transformers import TrainingArguments
from VAE import MambaVAE
from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer
from Dataset import get_dataset


class Trainer(transformers.Trainer):
    def __init__(self, *args, kl_warmup = 1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_step = 0
        self.kl_warmup = kl_warmup

    def compute_loss(self, model, inputs, return_outputs = False, num_items_in_batch = None):
        outputs = model(**inputs)
        logits_loss, kl_loss = outputs[0], outputs[1]
        decay = self.current_step / self.kl_warmup
        total_loss = logits_loss + decay * kl_loss
        
        # 记录损失
        self.log({
            "logits_loss": logits_loss.item(),
            "kl_loss": kl_loss.item(),
            "kl_loss_decay": decay,
        })
        
        return (total_loss, outputs) if return_outputs else total_loss

    def training_step(self, model, inputs, num_items_in_batch = None):
        loss = super().training_step(model, inputs)
        self.current_step += 1
        return loss


if __name__ == '__main__':
    config = MambaConfig.from_pretrained("state-spaces/mamba-130m-hf")
    config.use_mambapy = True
    model = MambaForCausalLM.from_pretrained('state-spaces/mamba-130m-hf', config=config) #MambaVAE() #.to('cuda')
    train_dataset, eval_dataset = get_dataset(1000)

    training_args = TrainingArguments(
        output_dir = './results',
        learning_rate = 1e-3,
        warmup_steps = 0,
        num_train_epochs = 3,
        per_device_train_batch_size = 16,
        per_device_eval_batch_size = 16,
        fp16 = True,
        eval_strategy = 'epoch',
        save_strategy = 'epoch',
        load_best_model_at_end = True,
        metric_for_best_model = 'eval_loss',
        greater_is_better = False,
        save_total_limit = 1,
        logging_dir = './logs',
        logging_steps = 100,
        report_to = "none"
    )

    trainer = transformers.Trainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
    )

    trainer.train()