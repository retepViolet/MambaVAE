import torch, os
from datasets import load_from_disk, concatenate_datasets
from Dataset import tokenizer
from Diffuser import Diffuser
from VAE import MambaVAE
from Baseline import Baseline
import transformers
from transformers import TrainingArguments
import random
from safetensors.torch import load_file
from datasets import Dataset
from tqdm import tqdm


class Trainer(transformers.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.baseline = Baseline().to(self.args.device)
        self.baseline.load_state_dict(torch.load('./results/baseline/model.pth', weights_only=True))
        self.vae = MambaVAE().to(self.args.device)
        self.vae.load_state_dict(torch.load('./results/vae0.2/model.pth', weights_only=True))

    def calculate_token_match(self, pred, target):
        answer_pos = (pred == 50279).nonzero(as_tuple=True)[0]
        if len(answer_pos) == 0:
            return 0.0
        target_set = set(target[1:].tolist())
        target_set.discard(0)
        pred_set = set(pred[answer_pos[0]+1:].tolist())
        pred_set.discard(0)
        matched = len(target_set.intersection(pred_set))
        if len(target_set) == 0:
            return 0.0
        return matched / len(target_set)

    @torch.no_grad()
    def evaluate(self, eval_dataset = None, ignore_keys = None, metric_key_prefix: str = "accuracy"):
        if eval_dataset is None: eval_dataset = self.eval_dataset
        random_indices = random.sample(range(len(eval_dataset)), min(128, len(eval_dataset))) ####
        eval_dataloader = self.get_eval_dataloader(eval_dataset.select(random_indices))

        self.model.eval()
        logits_loss, mse_loss = 0, 0
        acc, acc2, acc0 = 0, 0, 0
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = {k: v.to(self.args.device) for k, v in batch.items()}
            res = self.model(target = batch['target'].float(), 
                                   question_ids = batch['question_ids'], question_mask = batch['question_mask'],
                                   full_ids = batch['full_ids'], full_mask = batch['full_mask'], full_loss_mask = batch['full_loss_mask'])
            mse_loss += res[1] / len(eval_dataloader)
            
            outputs = self.model.generate(batch['question_ids'], batch['question_mask'], T = 100, progress_bar = False)
            # outputs2 = self.model.generate(batch['question_ids'], batch['question_mask'], T = 1, progress_bar = False)
            outputs2 = torch.randn_like(outputs)
            logits_loss += self.vae.decode(outputs, batch['full_ids'], batch['full_mask'], batch['full_loss_mask']).loss / len(eval_dataloader)
            
            states = self.vae.mlp2(self.vae.get_states(outputs).transpose(1,2))
            states2 = self.vae.mlp2(self.vae.get_states(outputs2).transpose(1,2))

            # 批量处理输入
            batch_size = batch['question_ids'].shape[0]
            valid_lengths = batch['question_mask'].sum(dim=1)

            # 计算准确率
            for i in range(batch_size):
                input_ids = batch['question_ids'][i:i+1, :valid_lengths[i]]
                tokens = self.vae.decoder.generate(
                    input_ids=input_ids,
                    inputs_ssm_states=states[i:i+1],
                    max_length=256
                )
                tokens2 = self.vae.decoder.generate(
                    input_ids=input_ids,
                    inputs_ssm_states=states2[i:i+1],
                    max_length=256
                )
                tokens0 = self.baseline.backbone.generate(
                    input_ids=input_ids,
                    max_length=256
                )
                acc += self.calculate_token_match(tokens[0], batch['answer_ids'][i]) / len(eval_dataloader.dataset)
                acc2 += self.calculate_token_match(tokens2[0], batch['answer_ids'][i]) / len(eval_dataloader.dataset)
                acc0 += self.calculate_token_match(tokens0[0], batch['answer_ids'][i]) / len(eval_dataloader.dataset)
                # if i == 1:
                #     print(tokenizer.decode(batch['full_ids'][i], skip_special_tokens = True))
                #     print('------------')
                #     print(tokenizer.decode(tokens[0], skip_special_tokens = True))
                #     print('------------')
                #     print(tokenizer.decode(tokens2[0], skip_special_tokens = True))
                #     print('------------')
                #     print(tokenizer.decode(tokens0[0], skip_special_tokens = True))
                #     print('============')

        self.model.train()
        print(f'logits_loss: {logits_loss.item()}; mse_loss: {mse_loss.item()}; diff100: {acc}; diff1: {acc2}; baseline: {acc0}')
        return {metric_key_prefix: acc}
    
    def save_model(self, output_dir = None, _internal_call = False):
        print('---saving model---')
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.model.state_dict(), output_dir+'/model.pth')


if __name__ == '__main__':
    dataset = load_from_disk("./data/CoT") #.select(range(50000))
    print(dataset)
    tot = len(dataset)
    eval_size= int(tot * 0.05)
    train_dataset = dataset.select(range(eval_size, tot))
    eval_dataset = dataset.select(range(eval_size))

    model = Diffuser(1000)
    model.load_state_dict(torch.load('./results/cfg/model.pth', weights_only=True), strict=False)
    # model.vae.load_state_dict(torch.load('./results/vae0.2/model.pth', weights_only=True), strict=False)
    # del model.vae.decoder
    for param in model.vae.decoder.parameters():
        param.requires_grad = False

    training_args = TrainingArguments(
        learning_rate = 2e-4,
        lr_scheduler_type = 'cosine',
        warmup_steps = 1000,
        num_train_epochs = 5,
        logging_steps = 100,
        weight_decay = 0.01,
        ###
        per_device_train_batch_size = 32,
        per_device_eval_batch_size = 32,
        dataloader_num_workers = 16,
        bf16 = True,
        eval_strategy = 'epoch', 
        save_strategy = 'epoch',
        save_total_limit = 1,
        load_best_model_at_end = True,
        metric_for_best_model = 'accuracy',
        greater_is_better = True,
        output_dir = './results',
        report_to = "none",
        max_grad_norm = 1.0,
        label_names = ['target', 'condition', "full_ids", "full_mask", "full_loss_mask", 
                        "question_mask", "question_ids", "answer_ids", "answer_mask"],
    )
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset
    )
    # trainer.train()
    print(trainer.evaluate())