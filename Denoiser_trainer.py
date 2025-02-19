from datasets import load_dataset
import torch, os
from torch.utils.data import DataLoader
from Denoiser import Denoiser
from diffusers import DDIMScheduler
from VAE import MambaVAE
from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup
from tqdm import tqdm

lr = 1e-4
warmup = 1000
epoch = 1
logging_steps = 1000
T = 100

noise_scheduler = DDIMScheduler(num_train_timesteps=T)
loss_fn = torch.nn.MSELoss()
denoiser = Denoiser(T)
vae = MambaVAE()


def get_dataLoader():
    dataset = load_dataset("arrow", data_files = './data/CoT_answer.arrow', split = 'train')
    print(dataset)
    tot = len(dataset)
    eval_size= int(tot * 0.05)
    train_dataset = dataset.select(range(eval_size, tot))
    eval_dataset = dataset.select(range(eval_size))
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=32,       # 每个 batch 的样本数
        shuffle=True,        # 是否在每个 epoch 开始时打乱数据
        num_workers=4        # 加载数据的子进程数，视机器性能而定
    )
    eval_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=32,       # 每个 batch 的样本数
        shuffle=True,        # 是否在每个 epoch 开始时打乱数据
        num_workers=4        # 加载数据的子进程数，视机器性能而定
    )
    return train_loader, eval_loader
trainLoader, evalLoader = get_dataLoader()


@torch.no_grad()
def proccess_data(data):
    input_ids1, mask1, input_ids2, mask2 = data['input_ids1'], data['attention_mask1'], data['input_ids2'], data['attention_mask2']
    states = vae.encode(input_ids2, attention_mask = mask2)
    return states, input_ids1, mask1


def get_loss(data):
    clean_states, input_ids, attention_mask = proccess_data(data)
    # Adding noise
    timesteps = torch.randint(0, T, (clean_states.shape[0],), device = clean_states.device, dtype = torch.long)
    noise = torch.randn_like(clean_states, device = clean_states.device)
    noisy_states = noise_scheduler.add_noise(clean_states, noise, timesteps)
    # Predict and loss
    noise_pred = denoiser(noisy_states, timesteps, input_ids, attention_mask)
    loss = loss_fn(noise_pred, noise)
    return loss


@torch.no_grad()
def evaluate():
    loss = 0
    for data in tqdm(evalLoader):
        loss += get_loss(data).item()
    return loss/len(evalLoader)


def training_loop():
    torch.manual_seed(42)
    # Tainer
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr = lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=warmup, num_training_steps=(epoch*len(trainLoader))
    )
    accelerator = Accelerator(mixed_precision='fp16', gradient_accumulation_steps=1)
    denoiser, optimizer, lr_scheduler, trainLoader = accelerator.prepare(
        denoiser, optimizer, lr_scheduler, trainLoader
    )
    tot_loss, step = 0, 0
    for i in range(epoch):
        for data in tqdm(trainLoader):
            with accelerator.accumulate(denoiser):
                loss = get_loss(data)
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(denoiser.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            step += 1
            tot_loss += loss.item()
            # Evaluation and print
            if (i+1) % logging_steps == 0:
                print(f"epoch: {i+1}, training loss: {round(tot_loss/50, 4)}, lr: {lr_scheduler.get_last_lr()[0]}")
                tot_loss = 0
        eval_loss = evaluate(evalLoader)
        print(f"eval loss:{round(eval_loss, 4)}")
    denoiser = accelerator.unwrap_model(denoiser)
    torch.save(denoiser.state_dict(), './results/denoiser.pth')
training_loop()