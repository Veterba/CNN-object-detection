import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.optim as optim

from config import DATASET_ROOT, DATASET_ANNFILE, EPOCHES

import numpy as np

from model import Model
from dataset import DetectObjData
from dataset import collate_function
from loss import yolo_loss

def train(dataset, model, epoches):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    data_tr = DataLoader(dataset(data_dir=DATASET_ROOT, ann_file=DATASET_ANNFILE, transform=transform), batch_size=16, shuffle=True, collate_fn=collate_function)
    data_val = DataLoader(dataset(data_dir=DATASET_ROOT, ann_file=DATASET_ANNFILE, transform=transform), batch_size=16, collate_fn=collate_function)


    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for ep in range(epoches):
        model.train()
        running_loss = 0.0
        for images, targets in data_tr:
            optimizer.zero_grad()
            outputs = model(images)
            loss = yolo_loss(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        print(f"epoch {ep + 1}/{epoches}  loss {running_loss / len(data_tr.dataset):.4f}")

if __name__ == "__main__":
    model = Model()
    epoches = EPOCHES
    dataset = DetectObjData
    train(dataset, model, epoches)

