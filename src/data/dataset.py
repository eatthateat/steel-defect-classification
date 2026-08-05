import torch
import cv2
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms


classes = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']


def class_to_idx(cls):
    for i, cls_name in enumerate(classes):
        if cls == cls_name:
            return i
        
def idx_to_class(idx):
    return classes[idx]


class NEUDataset(Dataset,):
    def __init__(self, root_dir, transform=None):
        self.root = Path(root_dir)
        self.transform = transform
        self.image_paths = sorted(list(Path(self.root / "images").rglob("*.jpg")))
 
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        image = cv2.imread(self.image_paths[idx], cv2.COLOR_BGR2RGB)
        label = class_to_idx(self.image_paths[idx].parent.name)
        
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor(image)
        
        return image, label

