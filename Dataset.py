import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
tokenizer.add_special_tokens({"bos_token": "<|startoftext|>"})


def get_dataset(tot = None, eval_ratio = 0.05):
    dataset1 = load_dataset("arrow", data_files = './cache/'+'new128.arrow', split = 'train')
    dataset2 = load_dataset("arrow", data_files = './cache/'+'bookcorpus.arrow', split = 'train').select(range(200000))
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
                cache_file_name = './cache/tokens.arrow', load_from_cache_file = False)
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
                cache_file_name = './cache/tokens.arrow', load_from_cache_file = False)
    return dataset



if __name__ == '__main__':
    dataset = CoT_seq2seq()
    print(dataset[0])