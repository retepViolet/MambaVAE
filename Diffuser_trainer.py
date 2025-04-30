import torch, os
from datasets import load_from_disk, concatenate_datasets
from Dataset import tokenizer
from Diffuser import Diffuser
from VAE import MambaVAE
import transformers
from transformers import TrainingArguments
import random
from safetensors.torch import load_file
from datasets import Dataset
from tqdm import tqdm


class Trainer(transformers.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vae = MambaVAE().to(self.args.device)
        self.vae.load_state_dict(torch.load('./results/vae0.2/model.pth', weights_only=True), strict=False)

    @torch.no_grad()
    def evaluate(self, eval_dataset = None, ignore_keys = None, metric_key_prefix: str = "eval"):
        # 获取 dataloader
        if eval_dataset is None: eval_dataset = self.eval_dataset
        random_indices = random.sample(range(len(eval_dataset)), min(1024, len(eval_dataset))) ####
        eval_dataloader = self.get_eval_dataloader(eval_dataset.select(random_indices))

        self.model.eval()
        logits_loss, mse_loss = 0, 0
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = {k: v.to(self.args.device) for k, v in batch.items()}
            res = self.model(target = batch['target'].float(), 
                                   question_ids = batch['question_ids'], question_mask = batch['question_mask'],
                                   full_ids = batch['full_ids'], full_mask = batch['full_mask'], full_loss_mask = batch['full_loss_mask'])
            mse_loss += res[0] / len(eval_dataloader)
            # print(res[0].item(), self.vae.decode(res[1], batch['full_ids'], batch['full_mask'], batch['full_loss_mask']).loss.item(), res[2], self.model.noise_scheduler.alphas_cumprod[res[2]])
            # self.model.noise_scheduler.set_timesteps(100)
            # for t in self.model.noise_scheduler.timesteps:
            #     timestep = torch.tensor([t]*batch['target'].shape[0], device = self.args.device, dtype = torch.long)
            #     noise = torch.randn_like(batch['target'], device = self.args.device, dtype = torch.float)
            #     noisy_states = self.model.noise_scheduler.add_noise(batch['target'].float(), noise, timestep).to(torch.float)
            #     condition = self.model.mamba_predict(noisy_states, batch['question_ids'], batch['question_mask'], timestep)
            #     print(t, self.model.noise_scheduler.alphas_cumprod[t], self.model.loss_fn(condition, batch['target']).item(), self.vae.decode(condition, batch['full_ids'], batch['full_mask'], batch['full_loss_mask']).loss.item())

            outputs = self.model.generate(batch['question_ids'], batch['question_mask'], T = 100, progress_bar = False)
            logits_loss += self.vae.decode(outputs, batch['full_ids'], batch['full_mask'], batch['full_loss_mask']).loss / len(eval_dataloader)
            
            # mse_loss += self.model(target = batch['target'].float(), condition = batch['condition'].float())[0] / len(eval_dataloader)
            # outputs = self.model.generate(condition = batch['condition'].float(), T = 100, progress_bar = False)
            # logits_loss += self.vae.decode(outputs, batch['full_ids'], batch['full_mask'], batch['full_loss_mask']).loss / len(eval_dataloader)

        self.model.train()
        print(f'logits_loss: {logits_loss.item()}; mse_loss: {mse_loss.item()}')
        return {f"{metric_key_prefix}_loss": logits_loss.item()}
    
    def save_model(self, output_dir = None, _internal_call = False):
        print('---saving model---')
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.model.state_dict(), output_dir+'/model.pth')



if __name__ == '__main__':
    dataset = load_from_disk("./data/CoT") #.select(range(10000))
    print(dataset)
    tot = len(dataset)
    eval_size= int(tot * 0.05)
    train_dataset = dataset.select(range(eval_size, tot))
    eval_dataset = dataset.select(range(eval_size))

    model = Diffuser(1000)
    model.load_state_dict(torch.load('./results/vae0.2/model.pth', weights_only=True), strict=False)
    del model.vae.decoder
    # for param in model.vae.decoder.parameters():
    #     param.requires_grad = False
    # model.load_state_dict(torch.load('./results/test/model.pth', weights_only=True), strict=False)

    training_args = TrainingArguments(
        learning_rate = 2e-4,
        lr_scheduler_type = 'cosine',
        warmup_steps = 100,
        num_train_epochs = 10,
        logging_steps = 1000,
        weight_decay = 0.01,
        ###
        per_device_train_batch_size = 64,
        per_device_eval_batch_size = 16,
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
        label_names = ['target', 'condition', "full_ids", "full_mask", "full_loss_mask", "question_mask", "question_ids"],
    )
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset
    )
    trainer.train()
    # print(trainer.evaluate())