import os
import torch
from torch.utils.data import DataLoader
from torch import Tensor
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from eegdatasets_leaveone import EEGDataset
from einops.layers.torch import Rearrange
from lavis.models.clip_models.loss import ClipLoss
from torch.utils.data import DataLoader
import utils
from utils import wandb_logger
import csv
import itertools
import math
import datetime

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=350):
        super(PositionalEncoding, self).__init__()
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        pe = self.pe[:x.size(0), :].unsqueeze(1).repeat(1, x.size(1), 1)
        x = x + pe
        return x


class EEGAttention(nn.Module):
    def __init__(self, channel, nhead):
        super(EEGAttention, self).__init__()
        print(channel, nhead)
        self.pos_encoder = PositionalEncoding(channel)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=channel, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.channel = channel

    def forward(self, src):
        src = src.permute(2, 0, 1)  # Change shape to [time_length, batch_size, channel]
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output.permute(1, 2, 0)  # Change shape back to [batch_size, channel, time_length]

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        # revised from shallownet
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 5), (1, 1)),
            nn.AvgPool2d((1, 17), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (64, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        # b, _, _, _ = x.shape
        x = x.unsqueeze(1)     
        # print("x", x.shape)   
        x = self.tsconv(x)
        # print("tsconv", x.shape)   
        x = self.projection(x)
        # print("projection", x.shape)  
        return x


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x


class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            FlattenHead()
        )

        
class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=2520, proj_dim=1024, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )


class Proj_img(nn.Sequential):
    def __init__(self, embedding_dim=1024, proj_dim=1024, drop_proj=0.3):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )
    def forward(self, x):
        return x 

class ATM_S_reconstruction_scale_0_1000(nn.Module):    
    def __init__(self, num_channels=64, sequence_length=334, num_subjects=1, num_features=64, num_latents=1024, num_blocks=1):
        super(ATM_S_reconstruction_scale_0_1000, self).__init__()
        self.attention_model = EEGAttention(num_channels, nhead=1)   
        self.subject_wise_linear = nn.ModuleList([nn.Linear(sequence_length, sequence_length) for _ in range(num_subjects)])
        self.enc_eeg = Enc_eeg()
        self.proj_eeg = Proj_eeg()        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()       
         
    def forward(self, x):
        # print(f"Before attention: {x.shape}")
        x = self.attention_model(x)
        # print(f'After attention shape: {x.shape}')
        x = self.subject_wise_linear[0](x)
        # print(f'After subject-specific linear transformation shape: {x.shape}')
        eeg_embedding = self.enc_eeg(x)
        # print(f'After enc_eeg shape: {eeg_embedding.shape}')
        out = self.proj_eeg(eeg_embedding)
        return out  

def train_model(eegmodel, dataloader, optimizer, device):
    eegmodel.train()
    total_loss = 0
    fwd_percent_correct = 0
    bwd_percent_correct = 0
    alpha=0.9
    mse_loss_fn = nn.MSELoss()
    for batch_idx, (eeg_data, text, text_features, img, img_features) in enumerate(dataloader):
        eeg_data = eeg_data.to(device)
        text_features = text_features.to(device).float() # Already normalized
        img_features = img_features.to(device).float() # Already normalized
        
        optimizer.zero_grad()
        eeg_features = eegmodel(eeg_data).float()
        eeg_features_norm = nn.functional.normalize(eeg_features.flatten(1), dim=-1)

        logit_scale = eegmodel.logit_scale
        img_loss = eegmodel.loss_func(eeg_features_norm, img_features, logit_scale)
        text_loss = eegmodel.loss_func(eeg_features_norm, text_features, logit_scale)
        contrastive_loss = img_loss
        regress_loss =  mse_loss_fn(eeg_features, img_features)
  
        loss = (alpha * regress_loss *10 + (1 - alpha) * contrastive_loss*10)
        loss.backward()
        
        optimizer.step()
        total_loss += loss.item()

        # 1024-way top-1 accuracy
        labels = torch.arange(len(eeg_data)).to(device)
        fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(eeg_features_norm, img_features), labels, k=1).item()
        bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(img_features, eeg_features_norm), labels, k=1).item()

    average_loss = total_loss / (batch_idx+1)
    fwd_percent_correct = fwd_percent_correct / (batch_idx+1)
    bwd_percent_correct = bwd_percent_correct / (batch_idx+1)
    return average_loss, fwd_percent_correct, bwd_percent_correct

def evaluate_model(eegmodel, dataloader, device, text_features_all, img_features_all, k):
    eegmodel.eval()
    text_features_all = text_features_all.to(device).float()
    img_features_all = img_features_all.to(device).float()
    total_loss = 0
    fwd_percent_correct = 0
    bwd_percent_correct = 0
    alpha = 0.9

    mse_loss_fn = nn.MSELoss()
    # ridge_lambda = 0.1
    with torch.no_grad():
        for batch_idx, (eeg_data, text, text_features, img, img_features) in enumerate(dataloader):
            eeg_data = eeg_data.to(device)
            text_features = text_features.to(device).float() # Already normalized
            img_features = img_features.to(device).float() # Already normalized
            
            eeg_features = eegmodel(eeg_data).float()
            eeg_features_norm = nn.functional.normalize(eeg_features.flatten(1), dim=-1)

            logit_scale = eegmodel.logit_scale
            img_loss = eegmodel.loss_func(eeg_features_norm, img_features, logit_scale)
            text_loss = eegmodel.loss_func(eeg_features_norm, text_features, logit_scale)
            contrastive_loss = img_loss
            regress_loss =  mse_loss_fn(eeg_features, img_features)
    
            loss = (alpha * regress_loss *10 + (1 - alpha) * contrastive_loss*10)
            total_loss += loss.item()

            # TODO: Evaluate on average of test images

            # 1024-way top-1 accuracy
            labels = torch.arange(len(eeg_data)).to(device)
            fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(eeg_features_norm, img_features), labels, k=1).item()
            bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(img_features, eeg_features_norm), labels, k=1).item()

            # TODO: k-way top 5 accuracy
                        
        average_loss = total_loss / (batch_idx+1)
        fwd_percent_correct = fwd_percent_correct / (batch_idx+1)
        bwd_percent_correct = bwd_percent_correct / (batch_idx+1)
    
    return average_loss, fwd_percent_correct, bwd_percent_correct

def main_train_loop(sub, eeg_model, train_dataloader, test_dataloader, optimizer, device, text_features_test_all, img_features_test_all, config, logger=None):
    logger = wandb_logger(config) if logger else None
    logger.watch(eeg_model,logger) 
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    v2_accs = []
    v4_accs = []
    v10_accs = []

    best_accuracy = 0.0
    best_model_weights = None
    best_epoch_info = {}
    results = []  # List to store results for each epoch
    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")  
    for epoch in range(config['epochs']):
        train_loss, train_fwd, train_bwd = train_model(eeg_model, train_dataloader, optimizer, device)
        
        if (epoch +1) % 5 == 0:                    
            if config['insubject']==True:       
                os.makedirs(f"./models/contrast/{config['encoder_type']}/{current_time}/{sub}", exist_ok=True)             
                file_path = f"./models/contrast/{config['encoder_type']}/{current_time}/{sub}/{epoch+1}.pth"
                torch.save(eeg_model.state_dict(), file_path)            
            else:                
                os.makedirs(f"./models/contrast/across/{config['encoder_type']}/{current_time}", exist_ok=True)             
                file_path = f"./models/contrast/across/{config['encoder_type']}/{current_time}/{epoch+1}.pth"
                torch.save(eeg_model.state_dict(), file_path)
            print(f"model saved in {file_path}!")

        train_losses.append(train_loss)
        train_accuracies.append(train_fwd)
        
        test_loss, test_fwd, test_bwd = evaluate_model(eeg_model, test_dataloader, device, text_features_test_all, img_features_test_all, k=200)
        test_losses.append(test_loss)
        test_accuracies.append(test_fwd)        
        # Append results for this epoch
        epoch_results = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_fwd": train_fwd,
            "train_bwd": train_bwd,
            "test_loss": test_loss,
            "test_fwd": test_fwd,
            "test_bwd": test_bwd
        }
        results.append(epoch_results)
        
        # if tewt_fwd > best_accuracy:
        #     best_accuracy = tewt_fwd
        #     best_model_weights = eeg_model.state_dict().copy()
        #     best_epoch_info = {
        #         "epoch": epoch + 1,
        #         "train_loss": train_loss,
        #         "train_accuracy": train_accuracy,
        #         "test_loss": test_loss,
        #         "test_accuracy": test_accuracy,
        #     }
        logger.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/fwd_pct_correct": train_fwd,
            "train/bwd_pct_correct": train_bwd,
            "test/loss": test_loss,
            "test/fwd_pct_correct": test_fwd,
            "test/bwd_pct_correct": test_bwd
        })

        print(f"Epoch {epoch + 1}/{config['epochs']} - Train Loss: {train_loss:.4f}, Train fwd: {train_fwd:.4f}, Test Loss: {test_loss:.4f}, Test fwd: {test_fwd:.4f}")        
  
    # model.load_state_dict(best_model_weights)
    # torch.save(model.state_dict(), '{train_pos_img_text}.pth')
    
    fig, axs = plt.subplots(3, 2, figsize=(10, 15))

    
    axs[0, 0].plot(train_losses, label='Train Loss')
    axs[0, 0].plot(test_losses, label='Test Loss')
    axs[0, 0].legend()
    axs[0, 0].set_title("Loss Curve")

    axs[0, 1].plot(train_accuracies, label='Train Accuracy')
    axs[0, 1].plot(test_accuracies, label='Test Accuracy')
    axs[0, 1].legend()
    axs[0, 1].set_title("Accuracy Curve")
    
    info_text = (f"Best Model Info (from Epoch {best_epoch_info['epoch']}):\n"
                f"Train Loss: {best_epoch_info['train_loss']:.4f}\n"
                f"Train Accuracy: {best_epoch_info['train_accuracy']:.4f}\n"
                f"Test Loss: {best_epoch_info['test_loss']:.4f}\n"
                f"Test Accuracy: {best_epoch_info['test_accuracy']:.4f}\n")

    axs[2, 1].axis('off')  
    axs[2, 1].text(0.5, 0.5, info_text, fontsize=10, ha='center', va='center', transform=axs[2, 1].transAxes)

    plt.tight_layout()
    plt.suptitle('pos_img_text', fontsize=16, y=1.05)
    plt.savefig('pos_img_text')
    logger.finish()
    return results


def main():
    subjects = ['sub-02']

    config = {
        "data_path": "/srv/eeg_reconstruction/shared/things_eeg_2/Preprocessed_data_250Hz",
        "project": "Alljoined1_image_generation_pretrain",
        "entity": "alljoined1",
        "name": "lr=3e-3, 1000-way",
        "lr": 3e-3,
        "epochs": 40,
        "batch_size": 1024,
        "logger": True,
        "insubject": True,
        "encoder_type": 'ATM_S_reconstruction_scale_0_1000',
        "img_encoder": 'Proj_img',
        "subjects": subjects
    }

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    for sub in subjects:
        # Instantiate new models for each subject
        eeg_model = globals()[config['encoder_type']](64, 334).to(device)
        
        # Reinitialize the optimizer
        optimizer = torch.optim.AdamW(itertools.chain(eeg_model.parameters()), lr=config['lr'])

        print(f'Processing {sub}: number of parameters:', sum([p.numel() for p in eeg_model.parameters()]))

        train_dataset = EEGDataset(subjects=[sub] if config['insubject'] else [], exclude_subject=sub if not config['insubject'] else None, split="train")
        test_dataset = EEGDataset(subjects=[sub] if config['insubject'] else [], exclude_subject=sub if not config['insubject'] else None, split="test")
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0, drop_last=True)

        text_features_test_all = test_dataset.text_features
        img_features_test_all = test_dataset.img_features

        results = main_train_loop(sub, eeg_model, train_loader, test_loader, optimizer, device, text_features_test_all, img_features_test_all, config, logger=config['logger'])

        # Save results to a CSV file
        results_dir = f"./outputs/{config['encoder_type']}/{sub}/"
        os.makedirs(results_dir, exist_ok=True)
        results_file = f"{results_dir}/{config['encoder_type']}_{('cross_exclude_' if not config['insubject'] else '')}{sub}.csv"

        with open(results_file, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f'Results saved to {results_file}')
            
if __name__ == '__main__':
    main()