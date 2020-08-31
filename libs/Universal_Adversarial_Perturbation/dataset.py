from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, path, txt, image_array, transform=None, pert=np.zeros(1), loader=None):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((int(words[0]),int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.loader = loader
        self.pert = pert
        self.image_data = image_array

    def __getitem__(self, index):
        image_index, label = self.imgs[index]
        img = Image.fromarray(np.clip(self.image_data[image_index] + self.pert, 0, 255).astype(np.uint8))
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)
