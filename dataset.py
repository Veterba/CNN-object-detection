from torchvision import datasets
import torch
from torch.utils.data import DataLoader, Dataset


class DetectObjData(Dataset):
    def __init__(self, data_dir, ann_file, transform=None):
        self.data = datasets.CocoDetection(data_dir, ann_file, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, anns = self.data[idx]
        boxes, labels = [], []

        for a in anns:
            x, y, w, h = a["bbox"]
            labels.append(a["category_id"])
            boxes.append([x, y, x + w, y + h]),

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64)
        }
        return img, target

    def loadCats(self):
        return self.data.coco.cats
