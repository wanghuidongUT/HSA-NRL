import numpy as np
import torch
from torch.utils.data import Dataset
from data.utils import noisify

class MICCAI(Dataset):
    def __init__(self, image_array, label_array, clean_index=None, split='easy', transform=None,
                 noise_type='symmetric', noise_rate=0.2, random_state=42, nb_classes=3):
        """
        image_array: numpy array of shape (N, 67, 67)
        label_array: numpy array of shape (N,) or (N, 1)
        clean_index: indices of clean/easy samples (used for 'easy' split)
        split: 'easy' or 'hard'
        """
        self.transform = transform
        self.split = split

        self.images = image_array
        self.labels = label_array.squeeze()
        self.nb_classes = nb_classes

        if split == 'easy':
            self.indices = np.array(clean_index)
        elif split == 'hard':
            all_indices = np.arange(len(self.images))
            easy_mask = np.zeros(len(self.images), dtype=bool)
            easy_mask[clean_index] = True
            self.indices = all_indices[~easy_mask]  # not in clean_index
        else:
            raise ValueError("split must be either 'easy' or 'hard'")

        # extract corresponding data
        self.dataset = self.images[self.indices]
        self.train_labels = self.labels[self.indices]

        # apply noise
        if noise_type == 'clean':
            self.train_noisy_labels = self.train_labels.copy()
            self.actual_noise_rate = 0.0
        else:
            self.train_noisy_labels, self.actual_noise_rate = noisify(
                dataset=self.dataset,
                train_labels=self.train_labels,
                noise_type=noise_type,
                noise_rate=noise_rate,
                random_state=random_state,
                nb_classes=self.nb_classes
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img = self.dataset[index]  # (67, 67)
        label = self.train_noisy_labels[index]
        global_index = self.indices[index]  # from original dataset

        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # (1, 67, 67)
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            img = self.transform(img)

        return img, label, global_index