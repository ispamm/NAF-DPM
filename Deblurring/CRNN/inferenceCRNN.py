import torch
import numpy as np
import argparse

from torch.nn import CTCLoss
import torch.optim as optim

import torchvision.utils as utils
import torchvision.transforms as transforms

from model_crnn import CRNN
#from datasets.ocr_dataset import OCRDataset
#from datasets.img_dataset import ImgDataset
from dataset import PatchDataset
from util import get_char_maps,  PadWhite, AddGaussianNoice,get_text_stack
from util import extract_patches_with_labels, pred_to_string
import properties as properties

from tqdm import tqdm
from PIL import Image, ImageOps
import os


class InferenceCRNN():
    def __init__(self,crnn_model_path) -> None:
        self.char_to_index, self.index_to_char, self.vocab_size = get_char_maps(
            properties.char_set)
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        #self.model = CRNN(self.vocab_size, False).to(self.device)
        self.model = torch.load(crnn_model_path).to(self.device)
        self.model.eval()

        self.input_size = properties.input_size
        self.transform = transforms.Compose([
            PadWhite(self.input_size),
            transforms.ToTensor(),
        ])


    def inference(self, img_path):
        image = Image.open(img_path).convert("L")
        image = self.transform(image).to(self.device)
        image = image.unsqueeze(0)

        scores = self.model(image)

        y = pred_to_string(scores,[],self.index_to_char)


        print(y)




if __name__ == "__main__":
    parser = argparse.ArgumentParser( 
        description='Inference with PreTrained CRNN model')
    
    parser.add_argument('--img_path', type=str,
                        default='', help='input image')
    parser.add_argument('--model_path', type=str,
                        default='', help='input image')


    args = parser.parse_args()
    print(args)

    crnn = InferenceCRNN(args.model_path)

    crnn.inference(args.img_path)
