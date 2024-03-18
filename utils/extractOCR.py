
import pytesseract
from PIL import Image
from paddleocr import PaddleOCR,draw_ocr
import os

# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `french`, `german`, `korean`, `japan`
# to switch the language model in order.
ocr = PaddleOCR(use_angle_cls=True, lang='en',show_log=False) # need to run only once to download and load model into memory
def list_files(folder_path):
    # Get the list of all files in the folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    return files

folder_path = 'C:/Users/Giordano/Desktop/datiDocDiff/BMVC_OCR_test_data/n_00FinetuneEnd2'
files_in_folder = list_files(folder_path)
output_file = 'C:/Users/Giordano/Desktop/datiDocDiff/BMVC_OCR_test_data/n_00FinetuneEnd2/txt00Fine300Paddle.txt'



for file in files_in_folder:
    with open(output_file,'a',encoding="utf8") as output:
        print(file)
        img_path = os.path.join(folder_path, file)
        result = ocr.ocr(img_path,cls=False)
        for idx in range(len(result)):
            res = result[idx]
            text=[]
            for line in res:
                print(line[1])
                text.append(line[1][0])
        text = " ".join(text)
        output.write(text)
        output.write("\n")
        output.write("\x0C")