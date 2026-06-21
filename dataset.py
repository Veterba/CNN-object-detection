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
        # boxes come in ORIGINAL-image pixel coords, but img is resized to 128x128.
        # normalize boxes to [0,1] by the original size so they align with the resized tensor.
        info = self.data.coco.loadImgs(self.data.ids[idx])[0]
        W, H = info["width"], info["height"]
        boxes, labels = [], []

        for a in anns:
            x, y, w, h = a["bbox"]
            labels.append(a["category_id"])
            boxes.append([x / W, y / H, (x + w) / W, (y + h) / H])

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64)
        }
        return img, target

    def loadCats(self):
        return self.data.coco.cats

def collate_function(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, 0)
    return images, targets

            

