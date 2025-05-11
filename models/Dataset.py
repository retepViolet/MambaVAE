import torch
from datasets import load_dataset, concatenate_datasets, load_from_disk
from transformers import AutoTokenizer
from VAE import MambaVAE


tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
tokenizer.add_tokens(["<|startoftext|>", "<|thoughts|>", "<|answer|>"])


def get_dataset(tot = None, eval_ratio = 0.05):
    dataset1 = load_dataset("arrow", data_files = './data/128.arrow', split = 'train')
    dataset2 = load_dataset("arrow", data_files = './data/bookcorpus.arrow', split = 'train') #.select(range(200000))
    dataset = concatenate_datasets([dataset1, dataset2]).shuffle(42)
    if tot is not None: dataset = dataset.select(range(tot))
    tot = len(dataset)
    eval_size= int(tot * eval_ratio)
    train_dataset = dataset.select(range(eval_size, tot))
    eval_dataset = dataset.select(range(eval_size))
    return train_dataset, eval_dataset



def CoT_seq2seq(tot = None):
    dataset = load_dataset('kaist-ai/CoT-Collection', trust_remote_code=True)['train'].shuffle(42)
    dataset = dataset.filter(lambda example: len(example["source"]) < 500*4 and len(example["rationale"]) + len(example["target"]) < 500)
    if tot is not None: dataset = dataset.select(range(tot))
    tot = len(dataset)

    def tokenize(examples):
        inputs = tokenizer(examples["source"],
                           truncation = True, 
                           max_length = 512, 
                           padding = "max_length")
        # inputs = tokenizer([f'<|startoftext|>{examples["rationale"][i]} Answer: {examples["target"][i]}<|endoftext|>' for i in range(len(examples['rationale']))],
        #                    truncation = True, 
        #                    max_length = 128, 
        #                    padding = "max_length")
        # length = torch.sum(inputs['attention_mask'], dim=-1, dtype=torch.float)
        # print(length.mean().item(), length.std().item())
        return inputs
    
    dataset = dataset.map(tokenize, remove_columns = ['source','target','rationale','task','type'], batched = True, 
                cache_file_name = './data/tokens.arrow', load_from_cache_file = False)
    return dataset



def CoT_full(tot = None):
    dataset = load_dataset('kaist-ai/CoT-Collection', trust_remote_code=True)['train'].shuffle(42)
    dataset = dataset.filter(lambda example: len(example["source"]) < 500*4 and len(example["rationale"]) + len(example["target"]) < 500)
    if tot is not None: dataset = dataset.select(range(tot))
    tot = len(dataset)

    def tokenize(examples):
        inputs = tokenizer([f'{examples["source"][i]} Thinking: {examples["rationale"][i]} Answer: {examples["target"][i]}' for i in range(len(examples['rationale']))],
                           truncation = True, 
                           max_length = 512, 
                           padding = "max_length")
        # length = torch.sum(inputs['attention_mask'], dim=-1, dtype=torch.float)
        # print(length.mean().item(), length.std().item())
        return inputs
    
    dataset = dataset.map(tokenize, remove_columns = ['source','target','rationale','task','type'], batched = True, 
                cache_file_name = './data/tokens.arrow', load_from_cache_file = False)
    return dataset



def CoT(tot = None, max_length = 256):
    dataset = load_dataset('kaist-ai/CoT-Collection', trust_remote_code=True)['train'].shuffle(42)
    dataset = dataset.filter(lambda example: len(example["source"]) + len(example["rationale"]) + len(example["target"]) < 1200)
    if tot is not None: dataset = dataset.select(range(tot))
    tot = len(dataset)

    def tokenize(examples):
        question = tokenizer([f'<|startoftext|>{examples["source"][i]}<|thoughts|>' for i in range(len(examples['source']))],
                    truncation = True, max_length = max_length, padding = "max_length", return_tensors = 'pt')
        # thoughts = tokenizer([f'<|startoftext|>{examples["source"][i]}<|thoughts|>{examples["rationale"][i]}<|answer|>' for i in range(len(examples['source']))],
        #             truncation = True, max_length = max_length, padding = "max_length", return_tensors = 'pt')
        full = tokenizer([f'<|startoftext|>{examples["source"][i]}<|thoughts|>{examples["rationale"][i]}<|answer|>{examples["target"][i]}<|endoftext|>' for i in range(len(examples['source']))],
                  truncation = True, max_length = max_length, padding = "max_length", return_tensors = 'pt')
        answer = tokenizer([f'<|answer|>{examples["target"][i]}<|endoftext|>' for i in range(len(examples['source']))],
                   truncation = True, max_length = 32, padding = "max_length", return_tensors = 'pt')
        full_loss_mask = full['attention_mask'].clone()
        full_loss_mask[question['attention_mask']==1] = 0
        # thoughts_loss_mask = thoughts['attention_mask'].clone()
        # thoughts_loss_mask[question['attention_mask']==1] = 0
        return {
          'question_ids': question['input_ids'].to(torch.int32),
          'question_mask': question['attention_mask'].to(torch.int8),
          # 'thoughts_ids': thoughts['input_ids'].to(torch.int32),
          # 'thoughts_mask': thoughts['attention_mask'].to(torch.int8),
          # 'thoughts_loss_mask': thoughts_loss_mask.to(torch.int8),
          'answer_ids': answer['input_ids'].to(torch.int32),
          'answer_mask': answer['attention_mask'].to(torch.int8),
          'full_ids': full['input_ids'].to(torch.int32),
          'full_mask': full['attention_mask'].to(torch.int8),
          'full_loss_mask': full_loss_mask.to(torch.int8),
        }
    
    dataset = dataset.map(tokenize, remove_columns = ['source','target','rationale','task','type'], batched = True)
    dataset = dataset.filter(lambda example: example["full_ids"][-1]==0 and example["answer_ids"][-1]==0)
    dataset.save_to_disk("./data/CoT3")
    return dataset



def CoT_image(tot = None, batch_size = 1024):
    dataset = load_from_disk("./data/CoT3")
    if tot is not None:
        dataset = dataset.select(range(tot))
    vae = MambaVAE()
    vae.load_state_dict(torch.load('./results/vae0.2/model.pth', weights_only=True), strict=False)
    vae = vae.cuda().half()

    @torch.no_grad()
    def get_image(data):
        target = vae.encode(torch.tensor(data['full_ids'], device = 'cuda'), torch.tensor(data['full_mask'], device = 'cuda'))[0].clamp(-1, 1)
        condition = vae.encode(torch.tensor(data['question_ids'], device = 'cuda'), torch.tensor(data['question_mask'], device = 'cuda'))[0].clamp(-1, 1)
        return {
          'target': target.half().cpu(),
          'condition': condition.half().cpu()
        }
    dataset = dataset.map(get_image, batch_size = batch_size, remove_columns = ['question_ids', 'question_mask', 'answer_ids', 'answer_mask', 'full_ids', 'full_mask', 'full_loss_mask'], batched = True, keep_in_memory=True)
    dataset.save_to_disk("./data/image")
    return dataset


if __name__ == '__main__':
    dataset = load_from_disk("./data/image")
    tokens = load_from_disk("./data/CoT3")
    for col in tokens.column_names:
        print(col)
        dataset = dataset.add_column(col, tokens[col])
    print(dataset)
    dataset.save_to_disk("./data/CoT")