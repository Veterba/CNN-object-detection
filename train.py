import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

from config import DATASET_ROOT, DATASET_ANNFILE

import numpy as np

from model import Model
from dataset import DetectObjData

def train(dataset, model, epoches):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    data_tr = DataLoader(DetectObjData(data_dir=DATASET_ROOT, ann_file=DATASET_ANNFILE, transform=transform), batch_size=16, shuffle=True)
    data_val = DataLoader(DetectObjData(data_dir=DATASET_ROOT, ann_file=DATASET_ANNFILE, transform=transform), batch_size=16)

    model = Model(num_classes=9)
    loss_fn = nn.CrossEntropyLoss()
#    loss_fn = None
    for ep in range(epoches):
        pass

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    data_set = DetectObjData(data_dir=DATASET_ROOT, ann_file=DATASET_ANNFILE, transform=transform)
    target_classes = list(data_set.loadCats().keys())
    print(target_classes) 
