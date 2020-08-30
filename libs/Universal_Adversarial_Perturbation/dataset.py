from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import os


def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset(Dataset):
    def __init__(self, path, txt, transform=None, pert=np.zeros(1), loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.loader = loader
        self.pert = pert
        self.path = path

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.fromarray(np.clip(self.loader(os.path.join(self.path, fn)) + self.pert, 0, 255).astype(np.uint8))
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)
