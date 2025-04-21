import torch.utils.data as data
from PIL import Image
import os
import json
import pickle
import numpy as np
import torch
from .utils import noisify

class CHAOYANG(data.Dataset):
    def __init__(self, root, train=True, transform=None):
        self.transform = transform
        self.train = train
        self.dataset = 'chaoyang'
        self.nb_classes = 4

        # 加载 image.npy 和 label.npy
        image_path = os.path.join(root, "PicDisease1.npy")
        label_path = os.path.join(root, "LabelDisease1.npy")
        imgs = np.load(image_path, allow_pickle=True)
        labels = np.load(label_path, allow_pickle=True)

        if self.train:
            self.train_data, self.train_labels = imgs, labels
            self.train_noisy_labels = [i for i in self.train_labels]
            self.noise_or_not = [True for i in range(self.__len__())]
        else:
            self.test_data, self.test_labels = imgs, labels
    
        self.nb_classes=4
        if self.train:
            self.train_data, self.train_labels = imgs,labels
            self.train_noisy_labels=[i for i in self.train_labels]
            self.noise_or_not = [True for i in range(self.__len__())]
        else:
            self.test_data, self.test_labels = imgs,labels

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_noisy_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]
    
        img = Image.open(img)

        if self.transform is not None:
            img = self.transform(img)


        return img, target, index

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)