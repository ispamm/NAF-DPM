import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, RandomAffine, RandomHorizontalFlip, RandomCrop, RandomVerticalFlip
from PIL import Image
import cv2
import numpy as np

def ImageTransform(loadSize):
    return {"train": Compose([
        RandomCrop(loadSize, pad_if_needed=True, padding_mode='constant', fill=255),
        RandomAffine(10, fill=255),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        ToTensor(),
    ]), "test": Compose([
        ToTensor(),
    ]), "train_gt": Compose([
        RandomCrop(loadSize, pad_if_needed=True, padding_mode='constant', fill=255),
        RandomAffine(10, fill=255),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        ToTensor(),
    ])}


class DocData(Dataset):
    def __init__(self, path_img, path_gt, loadSize, mode=1):
        super().__init__()
        self.path_gt = path_gt
        self.path_img = path_img
        self.data_gt = sorted(os.listdir(path_gt))
        self.data_img = sorted(os.listdir(path_img))
        self.mode = mode
        if mode == 1:
            self.ImgTrans = (ImageTransform(loadSize)["train"], ImageTransform(loadSize)["train_gt"])
        else:
            self.ImgTrans = ImageTransform(loadSize)["test"]

    def __len__(self):
        return len(self.data_gt)

    def __getitem__(self, idx):
        

        gt = cv2.imread(os.path.join(self.path_gt, self.data_gt[idx]), 0)
        img = cv2.imread(os.path.join(self.path_img, self.data_img[idx]), 0)
        

        if self.mode == 1:
            gt = Image.fromarray(np.uint8(gt))
            img = Image.fromarray(np.uint8(img))
            
            seed = torch.random.seed()
            torch.random.manual_seed(seed)
            img = self.ImgTrans[0](img)
            torch.random.manual_seed(seed)
            gt = self.ImgTrans[1](gt)
        else:
            img= self.ImgTrans(img)
            gt = self.ImgTrans(gt)
        name = self.data_img[idx]
        #print(f"Processing: {name}\n")
        return img, gt, name
