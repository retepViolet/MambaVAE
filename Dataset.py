import torch
from datasets import load_dataset
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")


def get_dataset(tot = None, eval_ratio = 0.1, max_length = 128):
    dataset = load_dataset('wikipedia', '20220301.simple')['train']
    if tot is None: tot = len(dataset)
    else: dataset = dataset.select(range(tot))

    def tokenize(examples):
        # print(examples['text'])
        text = [i.replace('\n', ' ').replace('  ', ' ') for i in examples['text']]
        inputs = tokenizer(text,
                           truncation = True, 
                           max_length = max_length, 
                           padding = "max_length",
                           return_tensors = 'pt')
        ones = torch.ones(inputs['attention_mask'].size(0), 1, dtype = inputs['attention_mask'].dtype)
        inputs['attention_mask'] = torch.cat([ones, inputs['attention_mask'][:, :-1]], dim=1)
        return inputs
    dataset = dataset.map(tokenize, remove_columns = ['id','url','title','text'], batched = True)
    
    eval_size= int(tot * eval_ratio)
    eval_dataset = dataset.select(range(eval_size))
    train_dataset = dataset.select(range(eval_size, tot))
    return train_dataset, eval_dataset


if __name__ == '__main__':
    train_data, val_data = get_dataset(10)
    print(train_data[0])

