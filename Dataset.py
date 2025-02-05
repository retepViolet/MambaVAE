import torch
from datasets import load_dataset
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
tokenizer.add_special_tokens({"bos_token": "<|startoftext|>"})

def get_dataset(tot = None, eval_ratio = 0.05, max_length = 128):
    dataset_name =  '20220301.en'
    dataset = load_dataset("arrow", data_files = './cache/'+'20220301.en', split = 'train').shuffle(seed=42)
    # dataset = load_dataset('wikipedia', dataset_name, trust_remote_code=True)['train'].shuffle(seed=42)
    if tot is None: tot = len(dataset)
    else: dataset = dataset.select(range(tot))

    def tokenize(examples):
        text = ['<|startoftext|>' + i.replace('\n', ' ').replace('  ', ' ') + '<|endoftext|>' for i in examples['text']]
        inputs = tokenizer(text,
                           truncation = True, 
                           max_length = max_length, 
                           padding = "max_length",
                           return_tensors = 'pt')
        return inputs
    # dataset = dataset.map(tokenize, remove_columns = ['id','url','title','text'], batched = True,
    #             cache_file_name = './cache/'+dataset_name, load_from_cache_file = False)
    
    eval_size= int(tot * eval_ratio)
    train_dataset = dataset.select(range(eval_size, tot))
    eval_dataset = dataset.select(range(eval_size))
    return train_dataset, eval_dataset


if __name__ == '__main__':
    train_data, val_data = get_dataset(1000000)
    print(train_data[0])

