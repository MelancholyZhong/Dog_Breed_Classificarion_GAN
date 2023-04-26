# ================================== #
#        Created by Hui Hu           #
#        Preprocess Images           #
# ================================== #

import imghdr  # check image file
import json
import re
from os import makedirs, walk

import scipy.io  # read mat file
from torchvision import transforms
from torchvision.io import ImageReadMode, read_image
from torchvision.utils import save_image


# extract labels and categories, create folders to save images
def extract_labels():
    folder = "stanford-dogs-dataset/"

    # Read training and test labels
    train_list = scipy.io.loadmat(folder + "lists/train_list.mat")
    test_list = scipy.io.loadmat(folder + "lists/test_list.mat")
    train_file_list = train_list["file_list"]
    train_labels_list = train_list["labels"]
    test_file_list = test_list["file_list"]

    category_dic = {}  # save filename-category pairs, True means it's training data, False means test data
    label_dic = {}  # save the breeds-number pairs
    dir_set = set()  # save folder names of 120 different breeds

    for i in range(len(train_file_list)):
        # file_list[0][0][0] = "n02085620-Chihuahua/n02085620_5927.jpg"
        lst = train_file_list[i][0][0].split("/")
        filename = str(lst[1])
        category_dic[filename] = True  # training data

        dir_set.add(lst[0])

        breed = lst[0].split("-")[1]
        label = str(train_labels_list[i][0])
        if breed not in label_dic:
            label_dic[breed] = label

    for j in range(len(test_file_list)):
        lst = test_file_list[j][0][0].split("/")
        filename = str(lst[1])  # .split(".")[0]
        category_dic[filename] = False  # test data

    # create folder to save json files
    makedirs("data/files/", exist_ok=True)

    # save category to json file
    with open("data/files/category.json", "w") as outfile:
        json.dump(category_dic, outfile)

    # convert to breeds-number pairs and save
    labels = {}
    for breed, label in label_dic.items():
        labels[label] = breed
    with open("data/files/label.json", "w") as outfile:
        json.dump(labels, outfile)

    # create folders to save images
    for dir_name in dir_set:
        makedirs("data/train/" + dir_name, exist_ok=True)
        makedirs("data/test/" + dir_name, exist_ok=True)


# extract bounding box and save to json file
def extract_bounding_box():
    directory = "stanford-dogs-dataset/Annotation"

    bounding_box = {}

    for dir_path, dir_names, file_names in walk(directory):
        for file_name in file_names:
            with open(dir_path + "/" + file_name, "r") as f:
                data = f.read()
            # extract values
            xmin = re.findall("<xmin>(.*?)</xmin>", data)
            xmax = re.findall("<xmax>(.*?)</xmax>", data)
            ymin = re.findall("<ymin>(.*?)</ymin>", data)
            ymax = re.findall("<ymax>(.*?)</ymax>", data)

            height = int(ymax[0]) - int(ymin[0])
            width = int(xmax[0]) - int(xmin[0])
            bounding_box[file_name + ".jpg"] = {
                "top": int(ymin[0]),
                "left": int(xmin[0]),
                "height": height,
                "width": width,
            }

    # save to json file
    makedirs("data/files/", exist_ok=True)
    with open("data/files/bounding_box.json", "w") as outfile:
        json.dump(bounding_box, outfile)


# Read bouding box, crop images and sove them into corresponding folders
def crop_images():
    directory = "stanford-dogs-dataset/Images"
    train_path = "data/train/"
    test_path = "data/test/"

    # get bounding box and category from json file
    with open("data/files/category.json", "r") as f:
        category = json.load(f)
    with open("data/files/bounding_box.json", "r") as f:
        bounding_box = json.load(f)

    # stanford-dogs-dataset/Images/n02089078-black-and-tan_coonhound n02089078_2110.jpg
    index = 1
    for dir_path, dir_names, file_names in walk(directory):
        for file_name in file_names:
            if imghdr.what(dir_path + "/" + file_name):
                # read image and crop it according to boundingn box
                image = read_image(dir_path + "/" + file_name, ImageReadMode.RGB)
                bounding = bounding_box[file_name]
                crop_image = transforms.functional.crop(
                    image, bounding["top"], bounding["left"], bounding["height"], bounding["width"]
                )

                # size = list(crop_image.size())
                # h, w = size[1], size[2]

                # if h > w:
                #     square_image = transforms.functional.pad(crop_image, [(h - w) // 2 + 2, 2])
                # else:
                #     square_image = transforms.functional.pad(crop_image, [2, (w - h) // 2 + 2])

                # affine_image = transforms.functional.affine(
                #     square_image, 0, (0, 0), 70 / min(list(square_image.size())[1:]), 0
                # )
                # final_image = transforms.functional.center_crop(affine_image, (64, 64))

                final_image = transforms.functional.resize(crop_image, [64, 64], antialias=True)

                # save image into corresponding foleder
                folder = dir_path.split("/")[-1]
                if category[file_name]:
                    folder_path = train_path + folder + "/"
                else:
                    folder_path = test_path + folder + "/"

                save_image(final_image.float(), folder_path + file_name, normalize=True)

                # print process
                print(index, folder_path + file_name)
                index += 1


# preprocess images for future algorithm
def preprocess():
    # extract labels and categories
    extract_labels()
    # extract bouding box
    extract_bounding_box()
    # crop images and save them
    crop_images()


if __name__ == "__main__":
    preprocess()
