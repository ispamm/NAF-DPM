from PIL import Image
import pytesseract
import numpy as np
from torchvision.transforms import ToPILImage
from torchvision.transforms import Compose, ToTensor
import pytesseract
from pytesseract import Output
import cv2
import json
import os
from tqdm import tqdm as tq

def encode_and_write_to_json(input_dict, output_file_path):
    """
    Encode a dictionary as JSON and write it to a file.

    Parameters:
    - input_dict: The dictionary to be encoded.
    - output_file_path: The file path where the JSON will be written.
    """
    try:
        with open(output_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(input_dict, json_file, indent=2)
        #print(f"JSON encoding of the dictionary written to {output_file_path}")
    except Exception as e:
        print(f"Error: {e}")



def extract_txt_and_boxes(input_folder,output_folder):

    input_file_paths = sorted(os.listdir(input_folder))

    for file in tq(input_file_paths):
        name = file.split('.')[0]
        img = cv2.imread(os.path.join(input_folder,file))
        d = pytesseract.image_to_data(img, output_type=Output.DICT)
        n_boxes = len(d['level'])
        box_and_labels=[]
        for i in range(n_boxes):
            if(d['conf'][i]==-1 or d['conf'][i]<80 ):
                continue

            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            text=d['text'][i].encode("ascii", "ignore").decode()

            extraction = {"Label": text, 'x_min':x,'x_max':x+w,'y_min':y,'y_max':y+h, "conf":d['conf'][i]}
            box_and_labels.append(extraction)
        
        encode_and_write_to_json(box_and_labels, os.path.join(output_folder,f"{name}.json"))




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder',
                         type=str, 
                         default='C:/Users/Giordano/Desktop/dataset/test_orig', 
                         help='path to the input folder, where input images are stored')
    parser.add_argument('--output_folder',
                         type=str, 
                         default='C:/Users/Giordano/Desktop/dataset/test_orig_text', 
                         help='path to the output folder, where the output json will be stored')

    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    extract_txt_and_boxes(input_folder,output_folder)

    print("Text and boxes extracted!")