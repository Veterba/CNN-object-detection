import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.optim as optim

from config import DATASET_ROOT, DATASET_ANNFILE, EPOCHES

import numpy as np

from model import Model
from dataset import DetectObjData
from dataset import collate_function

def train(dataset, model, epoches):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    data_tr = DataLoader(dataset(data_dir=DATASET_ROOT, ann_file=DATASET_ANNFILE, transform=transform), batch_size=16, shuffle=True, collate_fn=collate_function)
    data_val = DataLoader(dataset(data_dir=DATASET_ROOT, ann_file=DATASET_ANNFILE, transform=transform), batch_size=16, collate_fn=collate_function)


    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for ep in range(epoches):
        model.train()
        running_loss = 0.0
        for images, labels in data_tr:
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs.mean(dim=[2, 3]), labels['labels'])
            loss.backward()
            optimizer.step()

            running_loss += loss.train() * images.size(0)

if __name__ == "__main__":
    model = Model()
    epoches = EPOCHES
    dataset = DetectObjData
    train = train(dataset, model, epoches)
    print()

