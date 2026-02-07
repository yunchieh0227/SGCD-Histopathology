import pickle
import os
from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from torch.autograd import Variable
import torchvision
import random
import argparse
from PIL import Image
from diffusers import DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid
from accelerate import Accelerator
from tqdm.auto import tqdm
from pathlib import Path
from dataclasses import dataclass
from accelerate import notebook_launcher
from diffusers import UNet2DModel
import gc
from torch.utils.data import DataLoader, Dataset
import shutil
import time
import csv
from resize_method import resize
from vahadane import vahadane
import cv2
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score

import numpy as np
np.bool = np.bool_

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device =  "cpu"

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import shutil
import time
from PIL import Image
from os import listdir
from os.path import isfile, isdir, join

class CoordDataset(Dataset):
    def __init__(self, domain, transform=None, mode='testing', balance=False):
        if mode == 'training':
            folder = 'training/training'
        if mode =='val':
            folder = 'training/validation'
        if mode =='testing':
            folder = 'testing'
        self.class_label = {'n':0, 't':1}
        self.files = []
        self.labels = []
        self.transform = transform
        root_path = '/work/twsxuaj274/CAMELYON17_temp/center_'+str(domain)+'_patches_CL0_RL5_256/'+folder
        files = listdir(root_path)
        for f in files:
            fullpath = join(root_path, f)
            if isfile(fullpath):
                self.files.append(fullpath)
                self.labels.append(self.class_label[f[0]])
                
        print('\n============ Domain ',domain)
        print(' dataset info normal(no BL) ', self.labels.count(0), ' tumor ',self.labels.count(1))
        if balance:
            condition = lambda x: x == 0 #因為normal比較多，所以從中隨機選取數量跟tumor數量一致的樣本數
            ind = [i for i, elem in enumerate(self.labels) if condition(elem)]
            ind = random.sample(ind, len(self.labels)-len(ind))
            
            condition = lambda x: x == 1
            ind += [i for i, elem in enumerate(self.labels) if condition(elem)]
            
            self.labels = [self.labels[i] for i in ind]
            self.files = [self.files[i] for i in ind]
        print(' dataset info normal ', self.labels.count(0), ' tumor ',self.labels.count(1))
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image = Image.open(self.files[index]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, int(self.labels[index]), self.files[index]

@dataclass
class TrainingConfig:
    image_size = 96  # the generated image resolution
    eval_batch_size = 8  # how many images to sample during evaluation
    num_epochs = 20
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 50
    test_epochs = 1
    save_model_epochs = 1
    mixed_precision = "no"  # `no` for float32, `fp16` for automatic mixed precision
    
    #######################################################################
    SOURCE_DOMAIN = 5
    TARGET_DOMAIN = 4
    T = 30 
    guide_until = 10 
    #######################################################################
    
    output_dir_S = "ddpm_CAMELYON17/domain_"+str(SOURCE_DOMAIN)+'n'+str(TARGET_DOMAIN)+'_'+str(SOURCE_DOMAIN)  # the model name locally and on the HF Hub
    output_dir_T = "ddpm_CAMELYON17/domain_"+str(SOURCE_DOMAIN)+'n'+str(TARGET_DOMAIN)+'_'+str(TARGET_DOMAIN)  # the model name locally and on the HF Hub
    CLF_save_path = 'CycelDiffusion_Camelyon17_'+str(SOURCE_DOMAIN)+'n'+str(TARGET_DOMAIN)+'_0726.pth'
    mini_batch = 10

config = TrainingConfig()

IMG_SIZE = 96

TRANSFORM = transforms.Compose([
    transforms.Resize(size=(IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
source_dataset = CoordDataset(config.SOURCE_DOMAIN, TRANSFORM, mode='training', balance=True)
target_dataset = CoordDataset(config.TARGET_DOMAIN, TRANSFORM, mode='training', balance=True)
target_test_dataset = CoordDataset(config.TARGET_DOMAIN, TRANSFORM, mode='testing')

BATCH_SIZE = 2
train_loader_S = torch.utils.data.DataLoader(source_dataset, batch_size=BATCH_SIZE,shuffle=True)#, pin_memory=True, num_workers=64)
train_loader_T = torch.utils.data.DataLoader(target_dataset, batch_size=BATCH_SIZE,shuffle=True)#, pin_memory=True, num_workers=64)
test_loader_T = torch.utils.data.DataLoader(test_dataset, batch_size=16,shuffle=True)#, pin_memory=True, num_workers=64)

def read_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (96, 96))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # opencv default color space is BGR, change it to RGB
    p = np.percentile(img, 90)
    img = np.clip(img * 255.0 / p, 0, 255).astype(np.uint8)
    return img

S_path = source_dataset.__getitem__(20)[2]
T_path = target_dataset.__getitem__(20)[2]

source_image = read_image(S_path)
target_image = read_image(T_path)
print('source image size: ', source_image.shape)
print(np.max(source_image), np.min(source_image))
print('target image size: ', target_image.shape)
print(np.max(target_image), np.min(target_image))

from torch.utils.checkpoint import checkpoint

class RBF(nn.Module):
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)
        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)

class MMDLoss(nn.Module):
    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))
        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY

def create_from_numpy(total_images):
    output = []
    total_images = [(images * 255).astype(np.uint8) for images in total_images]
    for images in total_images:
        for i in range(images.shape[0]):
            output.append(Image.fromarray(images[i]))
    return output

def recon_x0(total_images, flag):
    epoch = 0
    bs = total_images[0].shape[0]
    total_images = [images1[:bs] / 2 +0.5 for images1 in total_images]
    total_images = [torch.permute(images1.clamp(0, 1), (0, 2, 3, 1)).cpu().detach().numpy() for images1 in total_images]
    images = create_from_numpy(total_images)
    image_grid = make_image_grid(images, rows=len(total_images), cols=bs)

    if flag == 'S':
        test_dir = os.path.join(config.output_dir_S, "samples")
    if flag == 'T':
        test_dir = os.path.join(config.output_dir_T, "samples")
    
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}_recon.png")
    
@torch.no_grad()
def evaluate_model(model, dataloader):
    size = len(dataloader.dataset)
    model.eval()
    
    correct = 0
    y_true, y_scores = [], []

    for X, y, _ in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        
        y_score = torch.softmax(pred, dim=1)[:, 1].cpu().numpy()
        y_true.extend(y.cpu().numpy())
        y_scores.extend(y_score)
    
    accuracy = correct / size
    print(f"Test Error: \nAccuracy: {(100 * accuracy):>0.1f}%")
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC: {roc_auc:.3f}")
    return accuracy, roc_auc
    

def batch_process_vahadane(images, target_images, vhd):
    processed_images = []
    TR = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    for i, (source_image, target_image) in enumerate(zip(images, target_images)):
        source_image = read_image(source_image)
        target_image = read_image(target_image)
        
        Ws, Hs = vhd.stain_separate(source_image)
        Wt, Ht = vhd.stain_separate(target_image)
        
        processed_image_np = vhd.SPCN(source_image, Ws, Hs, Wt, Ht)
        processed_image = TR(processed_image_np).float()  # Convert back to torch tensor
        if i ==0:
            processed_images = processed_image[None,:,:,:]
        else:
            processed_images = torch.cat([processed_images, processed_image[None,:,:,:]], dim=0)
    
    return processed_images


def denoise_step(sample, t, model, scheduler):
    t_tensor = torch.tensor([t], device=sample.device)
    residual = model(sample, t_tensor, return_dict=False)[0]
    
    alpha_prod_t = scheduler.alphas_cumprod[t]
    alpha_prod_t_prev = scheduler.alphas_cumprod[t - 1] if t > 0 else scheduler.alphas_cumprod[0]
    beta_prod_t = 1 - alpha_prod_t
    
    pred_x0 = (sample - residual * (beta_prod_t).sqrt()) / (alpha_prod_t).sqrt()

    mu_prev = alpha_prod_t_prev.sqrt() * pred_x0 + (1 - alpha_prod_t_prev).sqrt() * residual
    sigma_prev = (beta_prod_t * (1 - alpha_prod_t_prev) / (1 - alpha_prod_t)).sqrt()
    
    next_sample = mu_prev + sigma_prev * torch.randn_like(sample)
    
    return next_sample

import copy

def denoise_step_with_guidance(
    sample, t, 
    model, scheduler, 
    guidance_scale, early_stop, guide_until,
    vhd, x0_path, ref_T_path
):
    t_tensor = torch.tensor([t], device=sample.device)
    t_int = t.item()

    current_scale = guidance_scale
    if early_stop and t_tensor.item() <= guide_until:
        current_scale = 0

    sample = sample.clone().detach()
    model = copy.deepcopy(model)
    sample.requires_grad_(True)
    
    residual = model(sample, t_tensor, return_dict=False)[0].detach()

    alpha_prod_t = scheduler.alphas_cumprod[t_int]
    beta_prod_t = 1 - alpha_prod_t
    
    pred_x0 = (sample - residual * (beta_prod_t).sqrt()) / (alpha_prod_t).sqrt()
    
    if guidance_scale == 0:
        return torch.zeros_like(sample)
    
    ref_x0 = batch_process_vahadane(x0_path, ref_T_path, vhd)

    l2_diff = pred_x0 - ref_x0.to(sample.device)
    l2_norm = torch.norm(l2_diff)

    norm_grad = (l2_diff / l2_norm)
    
    return norm_grad * guidance_scale

def reverse_from_t_wtih_guide_snmf_checkpoint(
    scheduler, model, sample, x0, x0_path, ref_T_path, k, scale=1, early_stop=True, guide_until=10, guide=True
):
    device = sample.device
    step = -10
    timesteps = torch.arange(k, -1, step, device=device)
    scale = 10
    guide = False

    if guide:
        vhd = vahadane(LAMBDA1=0.01, LAMBDA2=0.01, fast_mode=0, getH_mode=0, ITER=50)
        
        for t in timesteps:
            grad_term = denoise_step_with_guidance(sample, t, 
                model, scheduler, 
                scale, early_stop, guide_until,
                vhd, x0_path, ref_T_path)
            sample = checkpoint(
                denoise_step,
                sample, t, model, scheduler
                )  - grad_term 
        del vhd
        return sample

    else:
        for t in timesteps:
            sample = checkpoint(
                denoise_step,
                sample, t, model, scheduler
            )
        return sample

def cycle_train_loop(config, model_S, model_T, classifier_S, classifier_T, noise_scheduler, optimizer, train_dataloader_S, train_dataloader_T, test_dataloader_T, lr_scheduler):
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    if accelerator.is_main_process:
        if config.output_dir_S is not None:
            os.makedirs(config.output_dir_S, exist_ok=True)
        if config.output_dir_T is not None:
            os.makedirs(config.output_dir_T, exist_ok=True)
        accelerator.init_trackers("train_example")

    noise_scheduler, model_S, model_T, classifier_S, classifier_T, optimizer, train_dataloader_S, train_dataloader_T, test_dataloader_T, lr_scheduler = accelerator.prepare(
        noise_scheduler, model_S, model_T, classifier_S, classifier_T, optimizer, train_dataloader_S, train_dataloader_T, test_dataloader_T, lr_scheduler
    )

    criterion = nn.CrossEntropyLoss()
    mmd_loss = MMDLoss()
    STEP = 0
    classifier_S.eval()
    MODE_FLAG = 1
    best_auc = 0
    
    for epoch in range(config.num_epochs):
        model_S.eval()
        model_T.eval()
        classifier_T.train()
        optimizer = torch.optim.AdamW(classifier_T.parameters(), lr=config.learning_rate)
        if MODE_FLAG==1:
            model_S.train()
            model_T.train()
            
            classifier_T.eval()
            mode = 'D'
            optimizer = torch.optim.AdamW(list(model_S.parameters())+list(model_T.parameters()), lr=config.learning_rate)
            MODE_FLAG = 0
        else:
            model_S.eval()
            model_T.eval()
            classifier_T.train()
            mode = 'C'
            optimizer = torch.optim.AdamW(classifier_T.parameters(), lr=config.learning_rate)
            MODE_FLAG = 1

        iter_S = iter(train_dataloader_S)
        iter_T = iter(train_dataloader_T)
        T = config.T 
        guide_until = config.guide_until
        progress_bar = tqdm(total=config.mini_batch, disable=not accelerator.is_local_main_process)
        print('\ntraining epoch ',epoch, ' time step is ', T, 'guide until ', guide_until,' mode ', mode)

        for step in range(config.mini_batch):
            STEP += 1
            try:
                x1, y1, x1_path = next(iter_S)
            except StopIteration:
                iter1 = iter(train_dataloader_S)
                x1, y1, x1_path = next(iter_S)

            try:
                x2, _, x2_path = next(iter_T)
            except StopIteration:
                iter2 = iter(train_dataloader_T)
                x2, _, x2_path = next(iter_T)
            ##############################   S -> T -> S   ##############################
            if mode == 'D':
                clean_images_S = x1.to(device)
                bs_S = clean_images_S.shape[0]
                noise = torch.randn(clean_images_S.shape, device=clean_images_S.device)
                noise2 = torch.randn(clean_images_S.shape, device=clean_images_S.device)
                timesteps_S = torch.ones((bs_S,), device=clean_images_S.device, dtype=torch.int64) * T
                timesteps_S2 = torch.ones((bs_S,), device=clean_images_S.device, dtype=torch.int64) * T

                noisy_images_S = noise_scheduler.add_noise(clean_images_S, noise, timesteps_S).requires_grad_(True)
                T_recon = reverse_from_t_wtih_guide_snmf_checkpoint(
                                 noise_scheduler, model_T, noisy_images_S, clean_images_S, 
                                 x1_path, x2_path, T, guide=True, guide_until=guide_until
                            )
                noisy_images2 = noise_scheduler.add_noise(T_recon, noise2, timesteps_S2).requires_grad_(True)
                S_recon = reverse_from_t_wtih_guide_snmf_checkpoint(
                                noise_scheduler, model_S, noisy_images2, T_recon, 
                                x1_path, x1_path, T, guide=True, guide_until=guide_until
                            )

                recon_x0([clean_images_S, T_recon, S_recon], 'S')
                MSE_loss = F.mse_loss(S_recon, clean_images_S)

                pred_label = classifier_T(T_recon)
                CE_loss_T = criterion(pred_label, y1.to(device))

                pred_label = classifier_S(S_recon)
                CE_loss_S = criterion(pred_label, y1.to(device))

                CE_loss = CE_loss_S + CE_loss_T

                del clean_images_S, bs_S, timesteps_S, timesteps_S2, T_recon, S_recon
                del noise, noisy_images_S, noise2, noisy_images2 , pred_label
                gc.collect()
                torch.cuda.empty_cache()

            if mode == 'C':
                FT = nn.Sequential(*list(classifier_T.children())[:-1])
                FC = torch.nn.Sequential(torch.nn.Flatten(), classifier_T.fc)

                clean_images_T = x2.to(device)
                clean_images_S = x1.to(device)
                bs_T = clean_images_T.shape[0]
                noise3 = torch.randn(clean_images_T.shape, device=clean_images_T.device)
                noise4 = torch.randn(clean_images_T.shape, device=clean_images_T.device)
                timesteps_T = torch.ones((bs_T,), device=clean_images_T.device, dtype=torch.int64) * T
                timesteps_T2 = torch.ones((bs_T,), device=clean_images_T.device, dtype=torch.int64) * T

                noisy_images_T = noise_scheduler.add_noise(clean_images_T, noise3, timesteps_T)
                T_recon = reverse_from_t_wtih_guide_snmf_checkpoint(
                                noise_scheduler, model_T, noisy_images_T, clean_images_T, 
                                x2_path, x2_path, T, guide=True, guide_until=guide_until
                            )

                noisy_images2 = noise_scheduler.add_noise(clean_images_S, noise4, timesteps_T2) 
                S_recon = reverse_from_t_wtih_guide_snmf_checkpoint(
                                noise_scheduler, model_T, noisy_images2, clean_images_S, 
                                x1_path, x2_path, T, guide=True, guide_until=guide_until
                            )

                recon_x0([clean_images_T, T_recon, clean_images_S, S_recon], 'T')

                S_features = FT(S_recon)
                T_features = FT(clean_images_T)

                pred_label = FC(S_features)
                CE_loss = criterion(pred_label, y1.to(device))
                
                S_features = torch.reshape(S_features,(S_features.shape[0],-1))
                T_features = torch.reshape(T_features,(T_features.shape[0],-1))
                CE_loss += mmd_loss(S_features.cpu(),T_features.cpu())
                MSE_loss = 888 * torch.ones((1,))
                MSE_loss2 = 888* torch.ones((1,))

                del clean_images_T, clean_images_S, bs_T, timesteps_T, timesteps_T2, T_recon, S_recon
                del noise3, noisy_images_T, noise4, noisy_images2
                gc.collect()
                torch.cuda.empty_cache()

            #############################   T -> S -> T   ##############################
            if mode == 'D':
                clean_images_T = x2.to(device)
                bs_T = clean_images_T.shape[0]
                noise3 = torch.randn(clean_images_T.shape, device=clean_images_T.device)
                noise4 = torch.randn(clean_images_T.shape, device=clean_images_T.device)
                timesteps_T = torch.ones((bs_T,), device=clean_images_T.device, dtype=torch.int64) * T
                timesteps_T2 = torch.ones((bs_T,), device=clean_images_T.device, dtype=torch.int64) * T

                noisy_images_T = noise_scheduler.add_noise(clean_images_T, noise3, timesteps_T).requires_grad_(True)
                S_recon = reverse_from_t_wtih_guide_snmf_checkpoint(
                                noise_scheduler, model_T, noisy_images_T, clean_images_T, 
                                x2_path, x1_path, T, guide=True, guide_until=guide_until
                            )
                
                noisy_images2 = noise_scheduler.add_noise(S_recon, noise4, timesteps_T2).requires_grad_(True) 
                T_recon = reverse_from_t_wtih_guide_snmf_checkpoint(
                                noise_scheduler, model_T, noisy_images2, S_recon, 
                                x2_path, x2_path, T, guide=True, guide_until=guide_until
                            )

                
                recon_x0([clean_images_T, S_recon, T_recon], 'T')
                MSE_loss2 = F.mse_loss(T_recon, clean_images_T)

                del clean_images_T, bs_T, timesteps_T, timesteps_T2, T_recon, S_recon
                del noise3, noisy_images_T, noise4, noisy_images2
                gc.collect()
                torch.cuda.empty_cache()

            ##############################   Total Loss   ##############################
            if mode == 'D':
                loss = MSE_loss+CE_loss+MSE_loss2
                accelerator.backward(loss / config.gradient_accumulation_steps)

                optimizer.step()
                lr_scheduler.step()
            
                for name, param in model_T.named_parameters():
                    if param.grad is not None:
                        # 檢查梯度的絕對平均值
                        print(f"Model_T parameter '{name}' grad_mean_abs: {torch.abs(param.grad).mean().item()}")
                    else:
                        print(f"Model_T parameter '{name}' has no grad.")
                    break
                for name, param in model_S.named_parameters():
                    if param.grad is not None:
                        print(f"Model_S parameter '{name}' grad_mean_abs: {torch.abs(param.grad).mean().item()}")
                    else:
                        print(f"Model_S parameter '{name}' has no grad.")
                    break
                optimizer.zero_grad()
            if mode == 'C':
                loss = (CE_loss) / config.gradient_accumulation_steps
                accelerator.backward(loss)

            progress_bar.update(1)
            progress_bar.set_description(f"noise loss: {(MSE_loss.item()):.4f} ；CE loss: {CE_loss.item():.4f} noise loss: {(MSE_loss2.item()):.4f}")

        if accelerator.is_main_process:
            pipeline_S = DDPMPipeline(unet=accelerator.unwrap_model(model_S), scheduler=noise_scheduler)
            pipeline_T = DDPMPipeline(unet=accelerator.unwrap_model(model_T), scheduler=noise_scheduler)

            if (epoch + 1) % config.test_epochs == 0 or epoch == config.num_epochs - 1:
                acc, auc = evaluate_model(classifier_T, test_dataloader_T)
                
                if auc >= best_auc:
                    best_auc = auc
                    pipeline_S.save_pretrained(config.output_dir_S)
                    pipeline_T.save_pretrained(config.output_dir_T)
                    torch.save(classifier_T.state_dict(),  config.CLF_save_path)
                    print('model save at AUC = ', auc)

CLF = torchvision.models.resnet50()
CLF.fc = nn.Linear(2048, 2)
CLF = CLF.to(device)
CLF.load_state_dict(torch.load("CAMELYON17_domain_"+str(config.SOURCE_DOMAIN)+"_resnet.pt"))

CLF2 = torchvision.models.resnet50()
CLF2.fc = nn.Linear(2048, 2)
CLF2 = CLF2.to(device)
CLF2.load_state_dict(torch.load("CAMELYON17_domain_"+str(config.SOURCE_DOMAIN)+"_resnet.pt"))

model_id_S = "ddpm_CAMELYON17/domain_"+str(config.SOURCE_DOMAIN)
model_id_T = "ddpm_CAMELYON17/domain_"+str(config.TARGET_DOMAIN)

pipline_S = DDPMPipeline.from_pretrained(model_id_S)
model_S = pipline_S.unet.to(device)

pipline_T = DDPMPipeline.from_pretrained(model_id_T)
model_T = pipline_T.unet.to(device)

print('Load S model from ', model_id_S)
print('Load T model from ', model_id_T)

noise_scheduler = DDIMScheduler.from_config("google/ddpm-church-256")
noise_scheduler.set_timesteps(num_inference_steps=100)

optimizer = torch.optim.AdamW(list(model_S.parameters())+list(model_T.parameters()), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(config.mini_batch * config.num_epochs),
)

print('================= ', config.SOURCE_DOMAIN, 'adapta to ', config.TARGET_DOMAIN, ' =================')

config.T = 5

args = (config, model_S, model_T, CLF, CLF2, noise_scheduler, optimizer, train_loader_S, train_loader_T, test_loader_T, lr_scheduler)

notebook_launcher(cycle_train_loop, args, num_processes=1)