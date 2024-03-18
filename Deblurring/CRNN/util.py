import os
import torch
import Levenshtein
import numpy as np

import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.utils as utils

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from torchvision.utils import save_image



def get_char_maps(vocabulary=None):
    if vocabulary is None:
        vocab = ['-']+[chr(ord('a')+i) for i in range(26)]+[chr(ord('A')+i)
                                                            for i in range(26)]+[chr(ord('0')+i) for i in range(10)]
    else:
        vocab = vocabulary
    char_to_index = {}
    index_to_char = {}
    cnt = 0
    for c in vocab:
        char_to_index[c] = cnt
        index_to_char[cnt] = c
        cnt += 1
    vocab_size = cnt
    return (char_to_index, index_to_char, vocab_size)


def save_img(images, name, dir, nrow=8):
    img = utils.make_grid(images, nrow=nrow)
    img = transforms.ToPILImage()(img)
    img.save(os.path.join(dir, name + '.png'), 'PNG')


def show_img(images, title="Figure", nrow=8):
    img = utils.make_grid(images, nrow=nrow)
    npimg = img.numpy()
    plt.figure(num=title)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def get_ununicode(text):
    text = text.replace('_', '-')
    text = text.replace('`', "'")
    text = text.replace('©', "c")
    text = text.replace('°', "'")
    text = text.replace('£', "E")
    text = text.replace('§', "S")

    index = text.find('€')
    if index >= 0:
        text = text.replace('€', '<eur>')
    un_unicode = unidecode(text)
    if index >= 0:
        un_unicode = un_unicode.replace('<eur>', '€')
    return un_unicode


def pred_to_string(scores, labels, index_to_char, show_text=False):
    preds = []
    # (seq_len, batch, vocab_size) -> (batch, seq_len, vocab_size)
    scores = scores.cpu().permute(1, 0, 2)
    for i in range(scores.shape[0]):
        interim = []
        for symbol in scores[i, :]:
            index = torch.argmax(symbol).item()
            interim.append(index)
        out = ""
        for j in range(len(interim)):
            if len(out) == 0 and interim[j] != 0:
                out += index_to_char[interim[j]]
            elif interim[j] != 0 and interim[j - 1] != interim[j]:
                out += index_to_char[interim[j]]
        preds.append(out)
        if show_text:
            print(labels[i], " -> ", out)
    return preds


def compare_labels(preds, labels):
    correct_count = 0
    total_cer = 0
    if not isinstance(labels, (list, tuple)):
        labels = [labels]
        print(labels)

    lens = len(labels)
    for i in range(lens):
        if preds[i] == labels[i]:
            correct_count += 1
        distance = Levenshtein.distance(labels[i], preds[i])
        total_cer += distance/len(labels[i])
    return correct_count, total_cer


def set_bn_eval(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()


def padder(crop, h, w):
    _, c_h, c_w = crop.shape
    pad_left = (w - c_w)//2
    pad_right = w - pad_left - c_w
    pad_top = (h - c_h)//2
    pad_bottom = h - pad_top - c_h
    pad = torch.nn.ConstantPad2d((pad_left, pad_right, pad_top, pad_bottom), 1)
    return pad(crop)


def get_text_stack(image, labels, input_size):
    text_crops = []
    labels_out = []
    for lbl in labels:
        label = lbl['label']
        x_min = int(lbl['x_min'])
        y_min = int(lbl['y_min'])
        x_max = int(lbl['x_max'])
        y_max = int(lbl['y_max'])
        text_crop = image[:, y_min:y_max, x_min:x_max]
        text_crop = padder(text_crop, *input_size)
        labels_out.append(label)
        text_crops.append(text_crop)
    if len(labels_out) == 0:
        return [],[]
    return torch.stack(text_crops), labels_out

def extract_patches_with_labels(images, labels, input_size):
    text_crops = []
    labels_out = []
    for image,label_dict in zip(images,labels):
        #i=0
        for lbl in label_dict:
            label = lbl['label']
            x_min = int(lbl['x_min'])
            y_min = int(lbl['y_min'])
            x_max = int(lbl['x_max'])
            y_max = int(lbl['y_max'])
            text_crop = image[:, y_min:y_max, x_min:x_max]
            text_crop = padder(text_crop, *input_size)
            labels_out.append(label)
            text_crops.append(text_crop)

    if len(text_crops)==0:
        return [],[]
    return torch.stack(text_crops), labels_out



def get_dir_list(test_dir):
    dir_list = []
    for root, dirs, _ in os.walk(test_dir):
        if not dirs:
            dir_list.append(root)
    return dir_list


def get_file_list(in_dir, filter):
    files = os.listdir(in_dir)
    processed_list = []
    for fil in files:
        if fil[-3:] in filter:
            processed_list.append(os.path.join(in_dir, fil))
    return processed_list


def get_files(in_dir, filter):
    processed_list = []
    for root, _, filenames in os.walk(in_dir):
        for f_name in filenames:
            if f_name.endswith(tuple(filter)):
                img_path = os.path.join(root, f_name)
                processed_list.append(img_path)
    return processed_list


def get_noisy_image(image, std=0.05, mean=0):
    noise = torch.normal(mean, std, image.shape)
    out_img = image + noise
    out_img.data.clamp_(0, 1)
    return out_img



    


class PadWhite(object):
    def __init__(self, size):
        assert isinstance(size, (int, tuple))
        if isinstance(size, tuple):
            self.height, self.width = size
        elif isinstance(size, int):
            self.height = self.width = size

    def __call__(self, img):
        if img.size[0] > self.width or img.size[1] > self.height:
            img.thumbnail((self.width, self.height))
        delta_width = self.width - img.size[0]
        delta_height = self.height - img.size[1]
        pad_width = delta_width // 2
        pad_height = delta_height // 2
        padding = (pad_width, pad_height, delta_width -
                   pad_width, delta_height-pad_height)
        return ImageOps.expand(img, padding, fill=255)


class AddGaussianNoice(object):
    def __init__(self, std=5, mean=0, is_stochastic=False):
        self.std = std
        self.mean = mean
        self.is_stochastic = is_stochastic

    def __call__(self, image):
        if self.is_stochastic:
            r_std = torch.randint(low=0, high=self.std+1, size=(1,)).item()/100
        else:
            r_std = self.std/100
        noise = torch.normal(self.mean, r_std, image.shape)
        out_img = image + noise
        out_img.data.clamp_(0, 1)
        return out_img