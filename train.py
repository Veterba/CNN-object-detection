import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from config import DATASET

import numpy as np

from model import Model


class DetectObjData(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = datasets.ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def classes(self):
        return self.data.classes

def train(dataset, model, epoches):

    data_tr = DataLoader(DetectObjData())
    data_val = Dataset(DetectObjData())
    model = Model()
#    loss_fn = None
    for ep in range(epoches):
        pass

if __name__ == "__main__":
    data_set = DetectObjData(data_dir=DATASET)
    print(data_set[0])
