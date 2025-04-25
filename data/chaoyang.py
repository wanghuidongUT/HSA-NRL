import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class CHAOYANG(Dataset):
    def __init__(self, image_path, label_path, train=True, transform=None, split_ratio=0.9, random_seed=42):
        self.transform = transform
        self.train = train

        # 加载数据
        images = np.load(image_path)           # shape: (N, 67, 67)
        labels = np.load(label_path)           # shape: (N, 1)
        labels = labels.squeeze()              # shape: (N,)

        # 按索引划分 train/val
        indices = np.arange(len(images))
        train_idx, val_idx = train_test_split(indices, train_size=split_ratio, random_state=random_seed, shuffle=True, stratify=labels)

        if train:
            self.images = images[train_idx]
            self.labels = labels[train_idx]
            self.indices = train_idx
        else:
            self.images = images[val_idx]
            self.labels = labels[val_idx]
            self.indices = val_idx

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]                 # shape: (67, 67)
        label = self.labels[index]               # scalar
        idx = self.indices[index]                # global index

        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # shape: (1, 67, 67)
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            img = self.transform(img)

        return img, label, idx
