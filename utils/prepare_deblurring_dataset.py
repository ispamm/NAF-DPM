import os
import shutil
from tqdm import tqdm        
import random
import math
from glob import glob

def divide_images_by_keyword(folder_path, output_folder_origin , output_folder_blur):
    """
    Divide images in a folder based on their names containing keywords into subfolders.

    Parameters:
    folder_path (str): The path to the folder containing the images.
    output_folder (str): The path to the folder where subfolders will be created.

    Returns:
    None
    """
    # Ensure the output folders exists, or create it if not
    if not os.path.exists(output_folder_origin):
        os.mkdir(output_folder_origin)
    if not os.path.exists(output_folder_blur):
        os.mkdir(output_folder_blur)

    
    # List all files in the input folder
    for filename in tqdm(os.listdir(folder_path)):
        # Construct the full path of the image
        image_path = os.path.join(folder_path, filename)

        # Check if the item is a file (not a subdirectory)
        if os.path.isfile(image_path):
            # Extract the name of the image (without extension)
            name, _ = os.path.splitext(filename)

            # Check if the name contains the word "origin" or "blur"
            if "orig" in name:
                keyword = "orig"
                shutil.copy(image_path, os.path.join(output_folder_origin, filename))

            elif "blur" in name:
                keyword = "blur"

                shutil.copy(image_path, os.path.join(output_folder_blur, filename))

            else:
                keyword = "other"
        

def divide_train_test(input_folder_blur,input_folder_origin):

    if len(os.listdir(input_folder_origin))!=len(os.listdir(input_folder_blur)):
        print("Errore dimensioni!")
        return
    
    num_images = len(os.listdir(input_folder_origin))

    print(f"Creating dataset from {num_images} ground-truth images.")
    indices = range(num_images)

    #Creating training indices
    train_count = 30000
    train_sample = random.sample(indices, train_count)

    #Creating validation indices
    validate_count = 10000
    indices_not_train = list(set(indices).difference(set(train_sample)))
    validate_sample = random.sample(indices_not_train, validate_count)

    train_orig = "./train_orig"
    train_blur = "./train_blur"
    test_orig = "./test_orig"
    test_blur = "./test_blur"   
    # Ensure the output folders exists, or create it if not
    if not os.path.exists(train_orig):
        os.mkdir(train_orig)
    if not os.path.exists(train_blur):
        os.mkdir(train_blur)    
    if not os.path.exists(test_orig):
        os.mkdir(test_orig)
    if not os.path.exists(test_blur):
        os.mkdir(test_blur)   

    origin_img_paths = sorted(glob(os.path.join(input_folder_origin,"*.png")))
    blur_img_paths = sorted(glob(os.path.join(input_folder_blur,"*.png")))

    #TRAIN IMAGES
    for i in tqdm(train_sample):

        path_orig = origin_img_paths[i]#os.path.join(input_folder_origin,origin_img_paths[i])
        path_blur = blur_img_paths[i]#os.path.join(input_folder_blur,blur_img_paths[i])
        #print(path_orig)

        filename_orig = origin_img_paths[i].split("\\")[-1]
        filename_blur = blur_img_paths[i].split("\\")[-1]
        #print(filename)

        shutil.copy(path_orig, os.path.join(train_orig, filename_orig))
        shutil.copy(path_blur, os.path.join(train_blur, filename_blur))

    #TEST IMAGES
    for i in tqdm(validate_sample):
        path_orig = origin_img_paths[i]#os.path.join(input_folder_origin,origin_img_paths[i])
        path_blur = blur_img_paths[i]#os.path.join(input_folder_blur,blur_img_paths[i])

        filename_orig = origin_img_paths[i].split("\\")[-1]
        filename_blur = blur_img_paths[i].split("\\")[-1]

        shutil.copy(path_orig, os.path.join(test_orig, filename_orig))
        shutil.copy(path_blur, os.path.join(test_blur, filename_blur))


image_folder_path = "./data"
output_folder_origin = "./origin"
output_folder_blur = "./blur"

divide_images_by_keyword(image_folder_path, output_folder_origin,output_folder_blur)
divide_train_test("./blur","./origin")