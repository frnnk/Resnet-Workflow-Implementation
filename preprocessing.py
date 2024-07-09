import torch
import torchvision.utils as tvu
import torchvision.transforms.v2 as v2 
import csv
import cv2
import pandas as pd
import os
import re
from PIL import Image


def preprocess_images(img_folder: str, herring: bool, greyscale: bool, resize_val: int = 224) -> None:
    """
    Converts a directory of either herring or non-herring images to greyscale 
    and then stores a copy in the specified PATH. Also appends a csv file that assigns
    each image to either 0 (herring) or 1 (non-herring)

    PARAMETERS:
    ---------------------
    img_folder: file path of the image folder containing all the images
    
    herring: either True or False, representing if the image folder is either all herring
    or all non-herring

    greyscale: either True or False, representing if the images should all be converted
    to greyscale
    """
    PATH = "./data/processed/simple_augment"
    rows_to_add = []
    counter = 0
    
    for img_name in os.listdir(img_folder):
        img_path = os.path.join(img_folder, img_name)
        final_path = os.path.join(PATH, img_name)
        
        if not os.path.exists(final_path):
            rows_to_add.append([img_name, 0 if herring else 1])
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE if greyscale else cv2.IMREAD_COLOR)
            # load image using OpenCV

            tensor = convert_cvimg_to_tensor_image(img, greyscale)
            tensor = resize_and_augment_image(tensor, resize_val)
            tvu.save_image(tensor, final_path)

            counter += 1
        # checks for if image is already present

        if counter % 10 == 0:
            print(f"Images added so far: {counter}")
    
    csv_path = os.path.join(PATH, "classes.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, mode="a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "Class"])
    # initializes a csv file with header first

    with open(os.path.join(PATH, "classes.csv"), mode="a", newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows_to_add)
    # appends rows to csv file

    print(f"Added {counter} images to destination directory")


def convert_cvimg_to_tensor_image(cv_image, greyscale):
    """
    Converts an OpenCV image object (numpy array) to an equivalent Pytorch Tensor object
    which contains floats between 0 and 1

    PARAMETERS:
    ---------------------
    cv_image: numpy array representing the image

    greyscale: boolean value indicating whether the input array has 1 channel or 3 channels
    """
    image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB) if not greyscale else cv_image
    tensor = torch.from_numpy(image).float()
    if not greyscale:
        tensor = tensor.permute(2, 0, 1)
    else:
        tensor = tensor.repeat(3, 1, 1)

    tensor /= 255
    tensor = tensor.clamp(0,1)
    return tensor


def resize_and_augment_image(tensor, resize_val):
    """
    Given an input tensor (float values between 0 and 1), resizes the image and
    augments it by performing a series of transformations with a set probability

    PARAMETERS:
    ---------------------
    tensor: Pytorch tensor object
    resize_val: pixel val to resize tensor image to (square)
    """
    resize = v2.Resize((resize_val, resize_val))
    random_flip = v2.RandomHorizontalFlip(0.5)
    return random_flip(resize(tensor))


def remove_empty_images(img_folder):
    label_path = os.path.join(img_folder, "labels")
    image_path = os.path.join(img_folder, "images")
    counter = 0

    nothing = set()
    for name in os.listdir(label_path):
        if os.path.getsize(os.path.join(label_path, name)) == 0:
            file = name.split(".txt")
            nothing.add(file[0])

    for empty in nothing:
        empty_label = empty + ".txt"
        empty_img = empty + ".jpg"
        os.remove(os.path.join(image_path, empty_img))
        os.remove(os.path.join(label_path, empty_label))
        counter += 1
    
    print(f"removed {counter} images from directory")


def otsu_thresholding(img_folder: str, herring: bool) -> None:
    PATH = "./data/processed/" + ("herring" if herring else "non_herring")
    binary = os.path.join(PATH, "binary")
    counter = 0
    # set up directory paths

    for img_name in os.listdir(img_folder):
        img_path = os.path.join(img_folder, img_name)
        final_path = os.path.join(binary, img_name)
        if not os.path.exists(final_path):
            otsu = cv2.threshold(img_path, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cv2.imwrite(final_path, otsu)
# incomplete for now (will work on later)


def crop(test_file: str) -> None:
    """
    Crops all test pictures in a test file according to their annotation box.
    The test file must have "images" and "labels" subfolders. Within the image
    subfolder, each image should have a corresponding annotation file WITH THE 
    SAME NAME in the labels subfolder.

    Input: path file to the test 
    Returns nothing, crops images in-place
    """
    images = os.path.join(test_file, "images")
    labels = os.path.join(test_file, "labels")
    res_path = os.path.join(test_file, "results")
    os.mkdir(res_path)
    count = 1

    for img_name in os.listdir(images): 
        img_path = os.path.join(images, img_name)
        img = Image.open(img_path)
        width, height = img.size
        # find image

        lab_path = os.path.join(labels, re.sub(r"\.[^.]+$", ".txt", img_name))
        # replaces path ending with .txt

        with open(lab_path, "r") as file:
            for box_string in file:
                processed_string = box_string.replace("\n", "")
                info = [float(num) for num in  processed_string.split(" ")]
                xcen, ycen = info[1]*width, info[2]*height
                wbox, hbox = info[3]*width, info[4]*height
                # processes the annotation string into data we can parse

                left = xcen - wbox/2
                up = ycen - hbox/2
                right = xcen + wbox/2
                down = ycen + hbox/2
                positions = (left, up, right, down)

                img_crop = img.crop(positions)
                save_path = os.path.join(res_path, str(count)+".jpg")
                count += 1
                img_crop.save(save_path)
                # saves cropped image to results folder
# outdated


if __name__ == "__main__":
    EX_PATH = "./example_bass.jpg"
    O_PATH = "./test_folder copy/images"
    # fir_path = "./data/raw/herring/filtered/RiverHerring_BlackNWhite_Set3/train"

    # remove_empty_images(fir_path)

    # img = cv2.imread(EX_PATH, cv2.IMREAD_GRAYSCALE)
    # tensor = convert_cvimg_to_tensor_image(img, True)
    # tvu.save_image(resize_and_augment_image(tensor, 224), "./big_bass.jpg")

    # ex_tensor = tvio.read_image("./big_bass.jpg")
    # print(ex_tensor.shape)
    
    # data = "./data/raw/herring/filtered/RiverHerring_BlackNWhite_Set1/test/images"
    # data = "./data/raw/non_herring/SalmonDataset1/train/images"
    # data = "./data/raw/non_herring/BassDataset1/train/images"
    # preprocess_images(data, herring=False, greyscale=True)

    classes = "./data/processed/simple_augment/classes.csv"
    df = pd.read_csv(classes)
    print(len(df))

    # data = pd.read_csv("./data/processed/classes.csv")
    # for index, line in data.iterrows():
    #     print(line["Name"], line["Class"])


    # convert_greyscale(EX_PATH, False)
    # img = cv2.imread(EX_PATH, cv2.IMREAD_GRAYSCALE)
    # blur = cv2.GaussianBlur(img, (5,5), 0)
    # val, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imshow("result2", otsu)

    # kernel = np.ones((5,5), np.uint8)
    # opening = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel, iterations=2)
    # # dil = cv2.dilate(otsu, kernel, iterations=2)
    # cv2.imshow("result3", opening)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    pass