train_image_folder= '/mnt/media/giordano/dataset_deblurring/train_orig'
train_text_folder = '/mnt/media/giordano/dataset_deblurring/train_orig_text_paddle'

val_image_folder= '/mnt/media/giordano/dataset_deblurring/test_orig'
val_text_folder = '/mnt/media/giordano/dataset_deblurring/test_orig_text_paddle'


crnn_model_path = "/mnt/media/giordano/CRNN"
crnn_tensor_board = "/mnt/media/giordano/CRNN/Logging"
prep_model_path = "/mnt/media/giordano/CRNN"
prep_tensor_board = "/mnt/media/giordano/CRNN"
img_out_path = "/mnt/media/giordano/CRNN"
param_path = "/mnt/media/giordano/CRNN"

input_size = (32, 128)
num_workers = 4
char_set = ['`', ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N','_',
            'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']', '^', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '~', '€', '}', '\\', '/']

tesseract_path = "/usr/share/tesseract-ocr/4.00/tessdata"
empty_char = ' '
max_char_len = 25

continue_training = False
model_path = '/mnt/media/giordano/CRNN/model_0.pth'
epoch_restart = 1