import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, RandomAffine, RandomHorizontalFlip, RandomCrop
from PIL import Image
import json
import Deblurring.CRNN.properties as properties
from torchvision.transforms.functional import crop

def ImageTransform():
    return Compose([ToTensor(),])
    

class DocDataFinetuning(Dataset):
    def __init__(self, path_img, path_gt, loadSize, text_dir, crop):
        super().__init__()
        self.path_gt = path_gt
        self.path_img = path_img
        self.text_dir = text_dir
        self.data_gt = sorted(os.listdir(path_gt))
        self.data_img = sorted(os.listdir(path_img))
        self.text_labels = sorted(os.listdir(text_dir))
        self.ImgTrans = ImageTransform()
        self.load_size = loadSize
        self.crop = crop
        self.random_crop = RandomCrop(loadSize, pad_if_needed=True, padding_mode='constant', fill=255)

    def __len__(self):
        return len(self.data_gt)

    def __getitem__(self, idx):

        gt = Image.open(os.path.join(self.path_gt, self.data_gt[idx]))
        img = Image.open(os.path.join(self.path_img, self.data_img[idx]))
        img = img.convert('RGB')
        gt = gt.convert('RGB')

        ### RANDOM CROP AND LABEL ADJUSTMENT
        if self.crop:
            self.crop_indexes = self.random_crop.get_params(img,self.load_size)

            img = crop(img,self.crop_indexes[0],self.crop_indexes[1],self.crop_indexes[2],self.crop_indexes[3])
            gt = crop(gt,self.crop_indexes[0],self.crop_indexes[1],self.crop_indexes[2],self.crop_indexes[3])

        img= self.ImgTrans(img)
        gt = self.ImgTrans(gt)
        name = os.path.basename(self.data_gt[idx])
        label = self.coord_loader(name)

        return img, gt, label, name


    ### Given an image path as input, retrieve de file containing text and boxes
    ### and return a dictionary containing text and corresponding bounding box
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

            if self.crop:
                i,j,h,w = self.crop_indexes
                if x_min < j :
                    continue
                if x_max > j+w:
                    continue
                if y_min < i :
                    continue
                if y_max > i+h:
                    continue

                x_min = x_min - j
                x_max = x_max - j
                y_min = y_min - i
                y_max = y_max - i

            if len(label) <= properties.max_char_len and x_max - x_min < 128 and y_max - y_min < 32:
                out = {'label': label, 'x_min': x_min, 'y_min': y_min, 'x_max': x_max, 'y_max': y_max}
                label_list_out.append(out)


        return label_list_out
    
    def collate(data):
        blur_images = []
        origin_images = []
        labels = []
        names = []
            
        for item in data:
            blur_images.append(item[0])
            origin_images.append(item[1])
            labels.append(item[2])
            names.append(item[3])

        return [torch.stack(blur_images),torch.stack(origin_images), labels, names]


