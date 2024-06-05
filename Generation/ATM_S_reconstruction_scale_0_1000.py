import os
import sys
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from lavis.models.clip_models.loss import ClipLoss
import random
from utils import create_dataloader, create_dataset, wandb_logger
import csv
import itertools
import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "components"))
from components.models import *


class ATM_S_reconstruction_scale_0_1000(nn.Module):
    def __init__(self, num_channels=63, sequence_length=25, num_subjects=1):
        super(ATM_S_reconstruction_scale_0_1000, self).__init__()
        self.attention_model = EEGAttention(num_channels, num_channels, nhead=1)
        self.subject_wise_linear = nn.ModuleList(
            [nn.Linear(sequence_length, sequence_length) for _ in range(num_subjects)]
        )
        self.enc_eeg = Enc_eeg()
        self.proj_eeg = Proj_eeg()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()

    def forward(self, x):
        x = self.attention_model(x)
        x = self.subject_wise_linear[0](x)
        eeg_embedding = self.enc_eeg(x)
        out = self.proj_eeg(eeg_embedding)
        return out


def train_model(
    eegmodel, dataloader, optimizer, device, text_features_all, img_features_all
):
    eegmodel.train()
    text_features_all = text_features_all.to(device).float()  # (n_cls, d)
    img_features_all = (img_features_all[::10]).to(device).float()
    total_loss, correct, total = 0, 0, 0
    alpha = 0.9
    features_list = []  # List to store features
    mse_loss_fn = nn.MSELoss()
    for batch_idx, (
        eeg_data,
        labels,
        _,
        text_features,
        _,
        img_features,
    ) in enumerate(dataloader):
        eeg_data = eeg_data.to(device)
        text_features = text_features.to(device).float()
        img_features = img_features.to(device).float()
        labels = labels.to(device)

        optimizer.zero_grad()
        eeg_features = eegmodel(eeg_data[:, :, :250]).float()
        features_list.append(eeg_features)
        logit_scale = eegmodel.logit_scale
        img_loss = eegmodel.loss_func(eeg_features, img_features, logit_scale)
        contrastive_loss = img_loss

        regress_loss = mse_loss_fn(eeg_features, img_features)

        loss = alpha * regress_loss * 10 + (1 - alpha) * contrastive_loss * 10
        loss.backward()

        optimizer.step()
        total_loss += loss.item()

        logits_img = logit_scale * eeg_features @ img_features_all.T
        logits_single = logits_img
        predicted = torch.argmax(
            logits_single, dim=1
        )  # (n_batch, ) \in {0, 1, ..., n_cls-1}

        batch_size = predicted.shape[0]
        total += batch_size
        correct += (predicted == labels).sum().item()

    average_loss = total_loss / (batch_idx + 1)
    accuracy = correct / total
    return average_loss, accuracy


def evaluate_model(
    eegmodel, dataloader, device, text_features_all, img_features_all, k
):
    eegmodel.eval()
    text_features_all = text_features_all.to(device).float()
    img_features_all = img_features_all.to(device).float()
    total_loss, correct, total = 0, 0, 0
    alpha = 0.9
    top5_correct_count = 0

    all_labels = set(range(text_features_all.size(0)))
    top5_acc = 0
    mse_loss_fn = nn.MSELoss()
    with torch.no_grad():
        for batch_idx, (
            eeg_data,
            labels,
            _,
            text_features,
            _,
            img_features,
        ) in enumerate(dataloader):
            eeg_data = eeg_data.to(device)
            text_features = text_features.to(device).float()
            labels = labels.to(device)
            img_features = img_features.to(device).float()
            eeg_features = eegmodel(eeg_data[:, :, :250]).float()

            logit_scale = eegmodel.logit_scale

            regress_loss = mse_loss_fn(eeg_features, img_features)
            img_loss = eegmodel.loss_func(eeg_features, img_features, logit_scale)
            _ = eegmodel.loss_func(eeg_features, text_features, logit_scale)
            contrastive_loss = img_loss

            regress_loss = mse_loss_fn(eeg_features, img_features)
            loss = alpha * regress_loss * 10 + (1 - alpha) * contrastive_loss * 10
            total_loss += loss.item()

            for idx, label in enumerate(labels):
                possible_classes = list(all_labels - {label.item()})
                selected_classes = random.sample(possible_classes, k - 1) + [
                    label.item()
                ]
                selected_img_features = img_features_all[selected_classes]
                if k == 200:
                    logits_img = (
                        logit_scale * eeg_features[idx] @ selected_img_features.T
                    )
                    logits_single = logits_img

                    predicted_label = selected_classes[
                        torch.argmax(logits_single).item()
                    ]  # (n_batch, ) \in {0, 1, ..., n_cls-1}
                    if predicted_label == label.item():
                        correct += 1

                    _, top5_indices = torch.topk(logits_single, 5, largest=True)

                    if label.item() in [
                        selected_classes[i] for i in top5_indices.tolist()
                    ]:
                        top5_correct_count += 1
                    total += 1

    print("total_loss", total_loss)
    print("batch_idx+1", batch_idx + 1)

    average_loss = total_loss / (batch_idx + 1)
    accuracy = correct / total
    top5_acc = top5_correct_count / total

    return average_loss, accuracy, top5_acc


def main_train_loop(
    sub,
    eeg_model,
    img_model,
    train_dataloader,
    test_dataloader,
    optimizer,
    device,
    text_features_train_all,
    text_features_test_all,
    img_features_train_all,
    img_features_test_all,
    config,
    logger=None,
):
    logger = wandb_logger(config) if logger else None
    logger.watch(eeg_model, logger)
    logger.watch(img_model, logger)
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    best_accuracy = 0.0
    best_epoch_info = {}
    results = []  # List to store results for each epoch
    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")

    for epoch in range(config["epochs"]):
        train_loss, train_accuracy = train_model(
            eeg_model,
            train_dataloader,
            optimizer,
            device,
            text_features_train_all,
            img_features_train_all,
        )

        if (epoch + 1) % 5 == 0:
            if config["insubject"]:
                os.makedirs(
                    f"./models/contrast/{config['encoder_type']}/{current_time}/{sub}",
                    exist_ok=True,
                )
                file_path = f"./models/contrast/{config['encoder_type']}/{current_time}/{sub}/{epoch+1}.pth"
                torch.save(eeg_model.state_dict(), file_path)
            else:
                os.makedirs(
                    f"./models/contrast/across/{config['encoder_type']}/{current_time}",
                    exist_ok=True,
                )
                file_path = f"./models/contrast/across/{config['encoder_type']}/{current_time}/{epoch+1}.pth"
                torch.save(eeg_model.state_dict(), file_path)
            print(f"model saved in {file_path}!")
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        test_loss, test_accuracy, top5_acc = evaluate_model(
            eeg_model,
            test_dataloader,
            device,
            text_features_test_all,
            img_features_test_all,
            k=200,
        )
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        # Append results for this epoch
        epoch_results = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "top5_acc": top5_acc,
        }
        results.append(epoch_results)

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            _ = eeg_model.state_dict().copy()
            best_epoch_info = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
            }
        logger.log(
            {
                "Train Loss": train_loss,
                "Train Accuracy": train_accuracy,
                "Test Loss": test_loss,
                "Test Accuracy": test_accuracy,
                "Epoch": epoch,
            }
        )

        print(
            f"Epoch {epoch + 1}/{config['epochs']} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Top5 Accuracy: {top5_acc:.4f}"
        )

    _, axs = plt.subplots(3, 2, figsize=(10, 15))

    axs[0, 0].plot(train_losses, label="Train Loss")
    axs[0, 0].plot(test_losses, label="Test Loss")
    axs[0, 0].legend()
    axs[0, 0].set_title("Loss Curve")

    axs[0, 1].plot(train_accuracies, label="Train Accuracy")
    axs[0, 1].plot(test_accuracies, label="Test Accuracy")
    axs[0, 1].legend()
    axs[0, 1].set_title("Accuracy Curve")

    info_text = (
        f"Best Model Info (from Epoch {best_epoch_info['epoch']}):\n"
        f"Train Loss: {best_epoch_info['train_loss']:.4f}\n"
        f"Train Accuracy: {best_epoch_info['train_accuracy']:.4f}\n"
        f"Test Loss: {best_epoch_info['test_loss']:.4f}\n"
        f"Test Accuracy: {best_epoch_info['test_accuracy']:.4f}\n"
    )

    axs[2, 1].axis("off")
    axs[2, 1].text(
        0.5,
        0.5,
        info_text,
        fontsize=10,
        ha="center",
        va="center",
        transform=axs[2, 1].transAxes,
    )

    plt.tight_layout()

    plt.suptitle("pos_img_text", fontsize=16, y=1.05)
    plt.savefig("pos_img_text")
    logger.finish()
    return results


def main():
    Encoder_list = [
        "EEGNetv4_Encoder",
        "ATCNet_Encoder",
        "EEGConformer_Encoder",
        "EEGITNet_Encoder",
        "ShallowFBCSPNet_Encoder",
    ]
    config = {
        "data_path": "/srv/eeg_reconstruction/shared/things_eeg_2/Preprocessed_data_250Hz",
        "project": "EEG_image_generation_pretrain",
        "entity": "alljoined1",
        "name": "lr=3e-4_img_pos_pro_eeg",
        "lr": 3e-4,
        "epochs": 40,
        "batch_size": 1024,
        "logger": True,
        "insubject": True,
        "encoder_type": "ATM_S_reconstruction_scale_0_1000",
        "img_encoder": "Proj_img",
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subjects = ["sub-01"]

    for sub in subjects:
        # Instantiate new models for each subject
        eeg_model = globals()[config["encoder_type"]](63, 1000 // 4).to(device)
        img_model = globals()[config["img_encoder"]]().to(device)

        # Reinitialize the optimizer
        optimizer = torch.optim.AdamW(
            itertools.chain(eeg_model.parameters(), img_model.parameters()),
            lr=config["lr"],
        )

        print(
            f"Processing {sub}: number of parameters:",
            sum([p.numel() for p in eeg_model.parameters()])
            + sum([p.numel() for p in img_model.parameters()]),
        )

        train_dataset = create_dataset(sub, config, train=True)
        test_dataset = create_dataset(sub, config, train=False)

        train_loader = create_dataloader(
            train_dataset, config["batch_size"], shuffle=True, drop_last=True
        )

        test_loader = create_dataloader(
            test_dataset, batch_size=1, shuffle=True, drop_last=True
        )

        text_features_train_all = train_dataset.text_features
        text_features_test_all = test_dataset.text_features
        img_features_train_all = train_dataset.img_features
        img_features_test_all = test_dataset.img_features

        results = main_train_loop(
            sub,
            eeg_model,
            img_model,
            train_loader,
            test_loader,
            optimizer,
            device,
            text_features_train_all,
            text_features_test_all,
            img_features_train_all,
            img_features_test_all,
            config,
            logger=config["logger"],
        )

        # Save results to a CSV file
        results_dir = f"./outputs/{config['encoder_type']}/{sub}/"
        os.makedirs(results_dir, exist_ok=True)
        results_file = f"{results_dir}/{config['encoder_type']}_{('cross_exclude_' if not config['insubject'] else '')}{sub}.csv"

        with open(results_file, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
