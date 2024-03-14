import os
import torch
import numpy as np
np.set_printoptions(threshold=np.inf)
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image


class LoadMyDataset(torch.utils.data.Dataset):
    def __init__(self, edge_path, img_path, im_size=256, train=True):
        self.input_path = edge_path
        self.target_path = img_path
        self.input_list = os.listdir(self.input_path)
        self.target_list = os.listdir(self.target_path)

        if train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop((im_size, im_size), scale=(0.9, 1.2)),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((im_size, im_size)),
                transforms.ToTensor()
            ])

    def __getitem__(self, item):
        input_path = os.path.join(self.input_path, self.input_list[item])
        target_path = os.path.join(self.target_path, self.target_list[item])
        sample_input = self.transform(Image.fromarray(np.array(Image.open(input_path).convert('RGB'))))
        sample_target = self.transform(Image.fromarray(np.array(Image.open(target_path).convert('RGB'))))

        return sample_input, sample_target

    def __len__(self):
        return len(self.input_list)

    def load_name(self, index):
        name = self.input_list[index]
        return os.path.basename(name)
