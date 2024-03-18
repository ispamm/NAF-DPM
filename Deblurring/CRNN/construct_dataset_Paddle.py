import json
import os
from tqdm import tqdm as tq
from paddleocr import PaddleOCR,draw_ocr
import paddle

# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `french`, `german`, `korean`, `japan`
# to switch the language model in order.
ocr = PaddleOCR(lang='en',show_log=False,use_gpu=True)# need to run only once to download and load model into memory
gpu_available  = paddle.device.is_compiled_with_cuda()
print("GPU available:", gpu_available)

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
    """
    Extract text and boxes from each image contained in input folder.
    The extracted information are then encoded using json and stored in a file
     for each image in output folder.

    Parameters:
    - input_folder: The input folder, where input images are stored.
    - output_folder: The output folder, where output images will be stored.
    """
    input_file_paths = sorted(os.listdir(input_folder))
    #For file in input folder
    for file in tq(input_file_paths):
        name = file.split('.')[0]
        img_path = os.path.join(input_folder,file)
        #Use paddle OCR to extract information
        result = ocr.ocr(img_path,cls=False)
        box_and_labels=[]
        #Iterate 
        for idx in range(len(result)):
            res = result[idx]
            if res == None:
                #No output from this image
                box_and_labels= {"Label": "",
                               'x_min':0,
                               'x_max':0,
                               'y_min':0,
                               'y_max':0}
                encode_and_write_to_json(box_and_labels, os.path.join(output_folder,f"{name}.json"))
                continue

            #Iterate over extracted boxes
            for line in res:
                
                #Extract text
                text = line[1][0].encode("ascii", "ignore").decode()
                conf = line[1][1]
                if (conf<0.85):
                    continue

                #Extract box
                box = line[0]
                x1 = box[0][0]
                y1 = box[0][1]

                x2 = box[1][0]
                y2 = box[1][1]

                x3 = box[2][0]
                y3 = box[2][1]

                x4 = box[3][0]
                y4 = box[3][1]

                x_min = min([x1, x2, x3, x4])
                y_min = min([y1, y2, y3, y4])
                x_max = max([x1, x2, x3, x4])
                y_max = max([y1, y2, y3, y4])

                extraction = {"Label": text,
                               'x_min':x_min,
                               'x_max':x_max,
                               'y_min':y_min,
                               'y_max':y_max}
                box_and_labels.append(extraction)
        #Encode and write the output for image i in a dedicated file
        encode_and_write_to_json(box_and_labels, os.path.join(output_folder,f"{name}.json"))



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder',
                         type=str, 
                         default='C:/Users/Giordano/Desktop/dataset/train_orig', 
                         help='path to the input folder, where input images are stored')
    parser.add_argument('--output_folder',
                         type=str, 
                         default='C:/Users/Giordano/Desktop/dataset/train_orig_text_paddle', 
                         help='path to the output folder, where the output json will be stored')

    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    extract_txt_and_boxes(input_folder,output_folder)

    print("Text and boxes extracted!")