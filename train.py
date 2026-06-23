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

    data_loader = DataLoader(dataset(data_dir=DATASET_ROOT, ann_file=DATASET_ANNFILE, transform=transform), batch_size=16, shuffle=True, collate_fn=collate_function)


    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for ep in range(epoches):
        model.train()
        running_loss = 0.0
        for images, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = yolo_loss(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        print(f"epoch {ep + 1}/{epoches}  loss {running_loss / len(data_tr.dataset):.4f}")
    torch.save(model.state_dict(), "model.weights.pt")
    

def evaluate_model(dataset, model):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    data_loader = DataLoader(dataset(data_dir=DATASET_ROOT, ann_file=DATASET_ANNFILE, transform=transform), batch_size=16, collate_fn=collate_function)

    model.eval() 
    
    running_loss = 0.0

    
    with torch.no_grad(): 
        for inputs, labels in data_loader:

            outputs = model(inputs)
            
            loss = yolo_loss(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            
    return running_loss / len(data_loader.dataset)

if __name__ == "__main__":
    model = Model()
    epoches = EPOCHES
    dataset = DetectObjData
    model.load_state_dict(torch.load("model.weights.pt"))
    print(evaluate_model(dataset, model))
