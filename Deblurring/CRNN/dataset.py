import os
import json
import random
import torch
import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image, ImageOps

from util import get_files
import properties as properties


class PatchDataset(Dataset):

    def __init__(self, img_dir, text_dir, pad=False, include_name=False):
        self.pad = pad
        self.include_name = include_name
        self.files = get_files(img_dir, ['png', 'jpg'])
        self.size = (300, 300)
        self.img_dir = img_dir
        self.text_dir = text_dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        image = Image.open(img_name).convert("RGB")
        img_name = os.path.basename(img_name)
        image = transforms.ToTensor()(image)
        label = self.coord_loader(img_name)
        if self.include_name:
            sample = (image, label, img_name)
        else:
            sample = (image, label)

        return sample

    def coord_loader(self, img_path):
        f = open(os.path.join( self.text_dir, f"{img_path[:-3]}json"), 'r')
        label_list = json.loads(f.read())
        if type(label_list) is dict:
            label_list = [label_list]
        f.close()
        label_list_out = []
        for text_area in label_list:

            label = text_area['Label']
            if label == " " or label == "" :
                continue

            x_min = text_area['x_min'] 
            y_min = text_area['y_min'] 
            x_max = text_area['x_max'] 
            y_max = text_area['y_max'] 

            if len(label) <= properties.max_char_len and x_max - x_min < 128 and y_max - y_min < 32:
                out = {'label': label, 'x_min': x_min, 'y_min': y_min, 'x_max': x_max, 'y_max': y_max}
                label_list_out.append(out)

        if len(label_list_out) == 0:
            label_list_out.append(
                {'label': ' ', 'x_min': 0, 'y_min': 0, 'x_max': 127, 'y_max': 31})

        return label_list_out

    def pad_height(self, image, height=400):
        _, h = image.size
        pad_bottom = height - h
        padding = (0, 0, 0, pad_bottom)
        return ImageOps.expand(image, padding, fill=255)

    def shuffle(self):
        random.shuffle(self.files)

    def collate(data):
        images = []
        labels = []
        if len(data[0]) == 3:
            names = []
            for item in data:
                images.append(item[0])
                labels.append(item[1])
                names.append(item[2])
            return [torch.stack(images), labels, names]
        else:
            for item in data:
                images.append(item[0])
                labels.append(item[1])
            return [torch.stack(images), labels]