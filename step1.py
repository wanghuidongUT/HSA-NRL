import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data.chaoyang import CHAOYANG
from data.miccai import MICCAI
import os
import pickle
from torchvision import models
from torch.optim.lr_scheduler import CosineAnnealingLR


def train(dataloader, epoch, model, optimizer, recorder):
    model.train()
    total, correct = 0, 0
    total_loss = 0.0
    for images, labels, indices in dataloader:
        images = images.cuda()
        labels = labels.cuda()

        outputs = model(images)
        probs = F.softmax(outputs, dim=1)

        for i in range(len(indices)):
            idx = indices[i].item() if isinstance(indices[i], torch.Tensor) else indices[i]
            if idx < len(recorder):
                recorder[idx].append(probs[i][labels[i]].item())

        loss = F.cross_entropy(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    acc = 100.0 * correct / total
    avg_loss = total_loss / total
    print(f"Epoch [{epoch+1}] Train Accuracy: {acc:.2f}%  Loss: {avg_loss:.4f}")
    return avg_loss, acc, acc


def evaluate(dataloader, model):  # updated to return both loss and acc
    model.eval()
    total, correct = 0, 0
    total_loss = 0.0
    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    acc = 100.0 * correct / total
    avg_loss = total_loss / total
    return avg_loss, acc


def main():
    epoch = 100
    patience = 10  # early stopping patience

    # -------- Load Data --------
    image_path = "chaoyang/PicDisease0.npy"
    label_path = "chaoyang/LabelDisease0.npy"

    train_dataset = CHAOYANG(image_path, label_path, train=True)
    val_dataset = CHAOYANG(image_path, label_path, train=False)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)

    def create_model():
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(512, 3)
        return model.cuda()

    # -------- Warm-up Phase --------
    model = create_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

    recorder = [[] for _ in range(len(train_dataset.images))]
    best_val_loss = float('inf')
    counter = 0
    for epoch_idx in range(epoch):
        train(train_loader, epoch_idx, model, optimizer, recorder)
        val_loss, val_acc = evaluate(val_loader, model)
        scheduler.step(val_loss)
        print(f"[Warm-up] Epoch [{epoch_idx+1}] Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%  LR: {optimizer.param_groups[0]['lr']:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered in Warm-up phase.")
                break

    # -------- Easy Sample Selection --------
    scores = np.array([np.mean(r) if len(r) > 0 else 0 for r in recorder])
    threshold = np.percentile(scores, 50)
    clean_idx = np.where(scores >= threshold)[0]

    path_li = np.array(train_dataset.images)[clean_idx]
    label_li = np.array(train_dataset.labels)[clean_idx]

    # -------- Clean Dataset Phase --------
    clean_dataset = MICCAI(train_dataset.images, train_dataset.labels, clean_index=clean_idx, split='easy', noise_type='clean')
    clean_loader = DataLoader(clean_dataset, batch_size=64, shuffle=True, num_workers=0)

    recorder2 = [[] for _ in range(len(train_dataset.images))]
    best_val_loss = float('inf')
    counter = 0
    for epoch_idx in range(epoch):
        train(clean_loader, epoch_idx, model, optimizer, recorder2)
        val_loss, val_acc = evaluate(val_loader, model)
        scheduler.step(val_loss)
        print(f"[Clean Phase] Epoch [{epoch_idx+1}] Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%  LR: {optimizer.param_groups[0]['lr']:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered in Clean Phase.")
                break

    # -------- Correction Phase --------
    correction_model = create_model()
    optimizer_corr = torch.optim.SGD(correction_model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    scheduler_corr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_corr, mode='min', factor=0.5, patience=2, verbose=True)

    e_h_dataset = MICCAI(train_dataset.images, train_dataset.labels, clean_index=clean_idx, split='hard', noise_type='symmetric', noise_rate=0.2)
    e_h_loader = DataLoader(e_h_dataset, batch_size=64, shuffle=True, num_workers=0)

    recorder4 = [[] for _ in range(len(train_dataset.images))]
    best_val_loss = float('inf')
    counter = 0
    for epoch_idx in range(epoch):
        train(e_h_loader, epoch_idx, correction_model, optimizer_corr, recorder4)
        val_loss, val_acc = evaluate(val_loader, correction_model)
        scheduler_corr.step(val_loss)
        print(f"[Correction phase] Epoch [{epoch_idx+1}] Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%  LR: {optimizer.param_groups[0]['lr']:.6f}")


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered in Correction Phase.")
                break

    # -------- Save Files --------
    os.makedirs("record", exist_ok=True)
    np.save("record/clean_image_path.npy", path_li)
    np.save("record/clean_label.npy", label_li)
    np.save("record/clean_index.npy", clean_idx)
    with open("record/recorder4.pkl", "wb") as f:
        pickle.dump(recorder4, f)

    os.makedirs("model", exist_ok=True)
    torch.save(correction_model.state_dict(), "model/correction_model.pth")

if __name__ == '__main__':
    main()
