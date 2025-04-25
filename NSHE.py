import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from data.miccai import MICCAI
import pickle
import os
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate(loader, model1, model2):
    model1.eval()
    # model2 will be updated dynamically by EMA
    correct1, correct2, total = 0, 0, 0
    all_labels = []
    all_preds1 = []
    all_preds2 = []
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.cuda().float()
            labels = labels.cuda()
            outputs1 = model1(images)
            outputs2 = model2(images)
            _, pred1 = torch.max(outputs1, 1)
            _, pred2 = torch.max(outputs2, 1)
            correct1 += (pred1 == labels).sum().item()
            correct2 += (pred2 == labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds1.extend(pred1.cpu().numpy())
            all_preds2.extend(pred2.cpu().numpy())

    # Confusion matrix
    cm1 = confusion_matrix(all_labels, all_preds1)
    cm2 = confusion_matrix(all_labels, all_preds2)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm1, annot=True, fmt='d', ax=axes[0], cmap='Blues')
    axes[0].set_title('Model1 Confusion Matrix')
    sns.heatmap(cm2, annot=True, fmt='d', ax=axes[1], cmap='Greens')
    axes[1].set_title('Model2 Confusion Matrix')
    plt.tight_layout()
    plt.savefig("record/confusion_matrix.png")
    print("[✓] Confusion matrix saved to record/confusion_matrix.png")
    plt.close()

    return 100 * correct1 / total, 100 * correct2 / total

def update_ema(model1, model2, m=0.999):
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        param2.data = m * param2.data + (1 - m) * param1.data

def train(loader, epoch, model1, optimizer, model2, recorder1, recorder2):
    model1.train()
    model2.eval()
    total, correct1, correct2 = 0, 0, 0
    for images, labels, indices in loader:
        images = images.cuda().float()
        labels = labels.cuda().long()

        logits1 = model1(images)
        logits2 = model2(images)

        prob1 = F.softmax(logits1, dim=1)
        prob2 = F.softmax(logits2, dim=1)

        for i in range(len(indices)):
            idx = indices[i].item() if isinstance(indices[i], torch.Tensor) else indices[i]
            if idx < len(recorder1):
                recorder1[idx].append(prob1[i][labels[i]].item())
            if idx < len(recorder2):
                recorder2[idx].append(prob2[i][labels[i]].item())

        loss = F.cross_entropy(logits1, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, pred1 = torch.max(logits1, 1)
        _, pred2 = torch.max(logits2, 1)
        total += labels.size(0)
        correct1 += (pred1 == labels).sum().item()
        correct2 += (pred2 == labels).sum().item()

    acc1 = 100.0 * correct1 / total
    acc2 = 100.0 * correct2 / total
    print(f"Epoch [{epoch+1}] Acc1: {acc1:.2f}%  Acc2: {acc2:.2f}%")
    return acc1, acc2

def make_model():
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(512, 3)
    return model.cuda()

def main():
    os.makedirs("record", exist_ok=True)
    os.makedirs("model", exist_ok=True)

    imgs = np.load("record/clean_image_path.npy", allow_pickle=True)
    labels = np.load("record/clean_label.npy")

    dataset = MICCAI(imgs, labels, clean_index=np.arange(len(imgs)), split='easy', noise_type='clean')
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

    recorder1 = [[] for _ in range(len(imgs))]
    recorder2 = [[] for _ in range(len(imgs))]

    model1 = make_model()
    model2 = make_model()
    import copy
    model2 = copy.deepcopy(model1)
    # model2.load_state_dict(torch.load("model/correction_model.pth"))
    # model2.eval()

    optimizer = torch.optim.SGD(model1.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5, verbose=True)

    for epoch in range(20):
        # train(train_loader, epoch, model1, optimizer, model2, recorder1, recorder2)
        acc1, acc2 = train(train_loader, epoch, model1, optimizer, model2, recorder1, recorder2)
        update_ema(model1, model2, m=0.999)
        scheduler.step(acc2)
        print("ACC1: {:.2f}%, ACC2: {:.2f}%".format(acc1, acc2), "Learning rate:", optimizer.param_groups[0]['lr'])

    test_imgs = np.load("D:/WangHuidong/github/HSA-NRL/chaoyang/PicDiseaseTest0.npy")
    test_labels = np.load("D:/WangHuidong/github/HSA-NRL/chaoyang/LabelDiseaseTest0.npy").squeeze()

    # align test label mapping
    unique_train = np.unique(labels)
    unique_test = np.unique(test_labels)

    if not np.array_equal(unique_train, unique_test):
        print("[Warning] Label mismatch between train and test, remapping test labels.")
        label_map = {v: i for i, v in enumerate(unique_train)}
        test_labels = np.array([label_map[x] for x in test_labels])

    test_dataset = MICCAI(test_imgs, test_labels, clean_index=np.arange(len(test_imgs)), split='easy', noise_type='clean')
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    acc1, acc2 = evaluate(test_loader, model1, model2)
    fused_acc = (acc1 + acc2) / 2
    print(f"\n[✓] Final Test Accuracy Summary:\n  Model1: {acc1:.2f}%\n  Model2: {acc2:.2f}%\n  Fused : {fused_acc:.2f}%")

    with open("record/recorder1.pkl", "wb") as f:
        pickle.dump(recorder1, f)
    with open("record/recorder2.pkl", "wb") as f:
        pickle.dump(recorder2, f)

    torch.save(model1.state_dict(), "model/nshe_model1.pth")
    torch.save(model2.state_dict(), "model/nshe_model2.pth")

if __name__ == '__main__':
    main()
