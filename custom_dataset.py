import os,sys,copy,torch,random,cv2,torchvision,json
from torch.utils.data import Dataset,DataLoader,random_split
import torchvision.transforms.functional as TF
import torch.nn.functional as NNF
from torchvision import datasets,transforms

class CustomDataset(Dataset):
    def __init__(self, data:torch.tensor, targets, data_transform=None,target_transform=None):
        self.data = data
        self.targets = targets
        if len(self.data) != len(self.targets):print(f'data:{len(self.data)},targets:{len(self.targets)} not match!')
        self.data_transform = data_transform
        self.target_transform = target_transform

    def __len__(self): return len(self.data)

    def __getitem__(self, index):
        data = self.data[index] if self.data_transform is None else self.data_transform(self.data[index])
        target = self.targets[index] if self.target_transform is None else self.target_transform(self.targets[index])
        return data, target

