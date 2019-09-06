# -*- coding: utf-8 -*-
"""
Definition file of the Preprocessing class.

This class is used to process different analyses upon all the images of a directory and store the results in text files.
"""
# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------
import os
import fnmatch
import progressbar
import numpy as np
import csv
import matplotlib.pyplot as plt
import random
import glob
import cv2

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image as ki
from keras.models import load_model

import Class_Image as Ci
from utils_class import list_from_lists, write_list_to_textfile, timeit, safe_folder_creation


# ----------------------------------------------------------------------------------------------------------------------
# Class definition
# ----------------------------------------------------------------------------------------------------------------------
class PreprocessingImages:
    """
    Definition of the Preprocessing class.

    This class is used to process different analyses upon all the images of a directory
    and store the results in text files.
    """

    def __init__(self, img_dir):
        self.img_dir = img_dir

        self.blurry_list = []
        self.overexpose_list = []
        self.exposure_histogram_list = []
        self.under_ratio_list = []
        self.winter_list = []
        self.snowy_list = []

        self.good_list = []
        self.bad_list = []

        self.images_list = fnmatch.filter(os.listdir(self.img_dir), '*.jpg')
        self.green_ratio = "Not computed"

    def __str__(self):
        string = "=== Images folder information === "
        string += "\nPath : " + self.img_dir
        string += "\nTotal number of images : " + str(len(self.images_list))
        string += "\nNumber of overexposed images : " + str(len(self.overexpose_list))
        string += "\nNumber of images with bad exposure : " + str(len(self.exposure_histogram_list))
        string += "\nValue of green ratio : " + str(self.green_ratio)
        string += "\nNumber of images with a green ratio under threshold : " + str(len(self.under_ratio_list))
        string += "\nNumber of blurry images : " + str(len(self.blurry_list))
        string += "\nNumber of snowy images : " + str(len(self.snowy_list))
        return string

    def green_analysis(self, green_seg_model, green_ratio, snow_seg_model):
        """
        Execute all the following analyses at the same time to gain computation time:

            - overexposure
            - exposure from histogram
            - blurriness
            - under green ratio
            - snow detection

        :param green_seg_model: segmentation model for green ratio analysis
        :type green_seg_model: keras.model
        :param green_ratio: threshold value for green ratio
        :type green_ratio: float
        :param snow_seg_model: segmentation model for snow detection
        :type snow_seg_model: keras.model
        """

        # Variable initialization
        self.green_ratio = green_ratio

        # Creation of directories to store results
        list_dir_path = os.path.join(self.img_dir, "Lists")
        list_dir_path = safe_folder_creation(list_dir_path)

        # Complete analysis for each image of the directory
        pbar = progressbar.ProgressBar()
        for img_filename in pbar(self.images_list):
            img_path = os.path.join(self.img_dir, img_filename)
            img = Ci.Image(img_path, build_array=True, build_green_segmentation=True, green_seg_model=green_seg_model,
                           build_snow_segmentation=True, snow_seg_model=snow_seg_model)

            if img.is_overexposed():
                self.overexpose_list.append(img_filename)

            if img.is_wrong_exposed():
                self.exposure_histogram_list.append(img_filename)

            if img.is_under_green_ratio(green_ratio):
                self.under_ratio_list.append(img_filename)

            if img.is_blurry():
                self.blurry_list.append(img_filename)

        # Produce all and bad lists
        self.bad_list = list_from_lists(self.overexpose_list, self.exposure_histogram_list,
                                        self.under_ratio_list, self.blurry_list)
        self.good_list = list(set(self.images_list) - set(self.bad_list))

        # Save in to files
        write_list_to_textfile(self.blurry_list, os.path.join(list_dir_path, "01-Blurry.txt"))
        write_list_to_textfile(self.exposure_histogram_list, os.path.join(list_dir_path, "02-Exposure_histogram.txt"))
        write_list_to_textfile(self.overexpose_list, os.path.join(list_dir_path, "03-Overexposed_images.txt"))
        write_list_to_textfile(self.under_ratio_list,
                               os.path.join(list_dir_path, "04-Under_Ratio_{}.txt".format(green_ratio)))
        write_list_to_textfile(self.bad_list, os.path.join(list_dir_path, "10-All_discarded.txt"))
        write_list_to_textfile(self.good_list, os.path.join(list_dir_path, "11-All_good.txt"))

    def snow_analysis(self, seg_model):
        """
        Complete snow analysis for each image of the directory and store the result in a text file.

        :param seg_model: segmentation model
        :type seg_model: keras.model
        """

        # Variable initialization
        self.snowy_list = []

        # Creation of directory to store results
        list_dir_path = os.path.join(self.img_dir, "Lists")
        list_dir_path = safe_folder_creation(list_dir_path)

        # Complete snow analysis for each image of the directory
        pbar = progressbar.ProgressBar()
        for img_filename in pbar(self.images_list):
            img_path = os.path.join(self.img_dir, img_filename)
            img = Ci.Image(img_path, build_array=True, build_snow_segmentation=True, snow_seg_model=seg_model)

            img.get_global_key_from_filename()
            # img.show_image_snow_segmentation()

            if img.is_snowy():
                self.snowy_list.append(img_filename)

        # Save in to files
        write_list_to_textfile(self.snowy_list, os.path.join(list_dir_path, "21-Snowy_list.txt"))


# ----------------------------------------------------------------------------------------------------------------------
# Functions definition
# ----------------------------------------------------------------------------------------------------------------------
def get_filename_from_key(key, image_folder):
    """
    Get the complete path of an image from its key

    :param key: image key
    :type key: str
    :param image_folder: folder containing images
    :type image_folder: str
    :return: image path
    :rtype: str
    """
    images = os.listdir(image_folder)
    for image in images:
        if key in image:
            return os.path.join(image_folder, image)


def get_label_comparison_from_string(string):
    """
    Create the label probabilities from the string comparison value

    The value being either 'left', 'right' or 'No preference'
    :param string: Comparison value
    :type string: str
    :return: label probabilities
    :rtype: list
    """
    if string == "left":
        return [1, 0]
    elif string == "right":
        return [0, 1]


def get_label_score_from_string(string):
    """
    Create the label probabilities from the string comparison value

    The value being either 'left', 'right' or 'No preference'
    :param string: Comparison value
    :type string: str
    :return: label score
    :rtype: int
    """
    if string == "left":
        return 1
    elif string == "right":
        return 0


def one_hot_it(labels, height, width, n_classes):
    """
    Resize labels array to the output shape of the model and one hot encode it

    :param labels:
    :param height:
    :param width:
    :param n_classes:
    :return:
    """
    x = np.zeros([height, width, n_classes])
    for i in range(height):
        for j in range(width):
            x[i, j, labels[i][j]] = 1
    return x


@timeit
def preprocessing_duels(csv_path, img_size, image_folder, save_folder, test):
    """
    Create the inputs of the comparison network from the comparisons csv and save them as npy

    A csv line has the following format:
        image_key_1, image_key_2, winner, ip_address
        the image keys are 22 character-long string
        winner is one of the 3 following string : left, right, equivalent
    :param csv_path: path of the comparisons csv
    :type csv_path: str
    :param img_size: input image size
    :type img_size: int
    :param image_folder: path of the image folder
    :type image_folder: str
    :param save_folder: path of the folder where npy are saved
    :type save_folder: str
    :param test: value to determine the size of the data dedicated to the test set.
                It can either be a proportion inside [0,1] or the number of comparisons kept aside.
    :type test: float
    :return:
    :rtype:
    """

    # List initialization
    left_images = []
    right_images = []
    labels = []
    labels_score = []

    # Get data from csv
    with open(csv_path, 'r') as csvfileReader:
        reader = csv.reader(csvfileReader, delimiter=',')
        print("Creating inputs from csv ...")
        pbar = progressbar.ProgressBar()
        for line in pbar(reader):
            # Do not include No preference comparisons
            if line != [] and line[2] != 'No preference':
                # Create Image instances
                left_image_path = get_filename_from_key(line[0], image_folder)
                right_image_path = get_filename_from_key(line[1], image_folder)
                left_img = Ci.Image(left_image_path)
                right_img = Ci.Image(right_image_path)

                # Add images to list
                left_images.append(left_img.preprocess_image(img_size))
                right_images.append(right_img.preprocess_image(img_size))

                # Add labels to list
                labels.append(get_label_comparison_from_string(line[2]))
                labels_score.append(get_label_score_from_string(line[2]))

    # Compute number of comparisons kept for test set
    if len(labels) > test > 1:
        nb_test = int(test)
    elif 0 <= test <= 1:
        nb_test = int(test * len(labels))
    else:
        raise ValueError

    print("Done\nSaving test set ...")
    # Create test dataset
    test_left = np.array(left_images[:nb_test])
    test_right = np.array(right_images[:nb_test])
    test_labels = np.array(labels[:nb_test])
    test_labels_score = np.array(labels_score[:nb_test])

    # Save testing dataset as npy
    test_folder = os.path.join(save_folder, "test")
    test_folder = safe_folder_creation(test_folder)
    np.save(os.path.join(test_folder, "test_left_{}".format(img_size)), np.array(test_left))
    np.save(os.path.join(test_folder, "test_right_{}".format(img_size)), np.array(test_right))
    np.save(os.path.join(test_folder, "test_labels_{}".format(img_size)), np.array(test_labels))
    np.save(os.path.join(test_folder, "test_labels_score_{}".format(img_size)), np.array(test_labels_score))

    print("Done\nSaving train set ...")
    # Create training dataset
    train_left = np.array(left_images[nb_test:])
    train_right = np.array(right_images[nb_test:])
    train_labels = np.array(labels[nb_test:])
    train_labels_score = np.array(labels_score[nb_test:])

    # Save training dataset as npy
    train_folder = os.path.join(save_folder, "train")
    train_folder = safe_folder_creation(train_folder)
    np.save(os.path.join(train_folder, "train_left_{}".format(img_size)), np.array(train_left))
    np.save(os.path.join(train_folder, "train_right_{}".format(img_size)), np.array(train_right))
    np.save(os.path.join(train_folder, "train_labels_{}".format(img_size)), np.array(train_labels))
    np.save(os.path.join(train_folder, "train_labels_score_{}".format(img_size)), np.array(train_labels_score))
    print("Done")

    return train_left, train_right, train_labels, train_labels_score


@timeit
def data_aug(train_left, train_right, train_label, train_label_score, nb, save_folder):
    """
    Create nb augmented comparisons from one comparison of the set. Save results in a folder.

    :param train_left: left images array
    :type train_left: np.array
    :param train_right: right images array
    :type train_right: np.array
    :param train_label: comparisons labels array
    :type train_label: np.array
    :param train_label_score: ranking labels array
    :type train_label_score: np.array
    :param nb: number of augmented comparisons created for each original one
    :type nb: int
    :param save_folder: path of the folder where the augmented dataset folder is stored
    :type save_folder: str
    :return: augmented dataset
    :rtype: tuple(np.array)
    """

    # Create saving folder
    aug_folder = safe_folder_creation(os.path.join(save_folder))

    # Specify data generator parameters
    datagenargs = {
        'rotation_range': 2, 'width_shift_range': 0.2, 'height_shift_range': 0.2,
        'shear_range': 0.1,
        'zoom_range': 0.25, 'horizontal_flip': True, 'fill_mode': 'nearest'
    }

    #  Create generators
    left_datagen = ImageDataGenerator(**datagenargs)
    right_datagen = ImageDataGenerator(**datagenargs)

    # Initialization of data
    train_left_aug = list(train_left)
    train_right_aug = list(train_right)
    train_label_aug = list(train_label)
    train_label_score_aug = list(train_label_score)
    img_size = train_left[0].shape[0]

    # Display processing advancement
    print("Creating new inputs...")
    pbar = progressbar.ProgressBar()
    # Create nb augmented images from an original one
    for duel in pbar(range(len(train_label_score))):
        for _ in range(nb):
            # Create one augmented image from the left one
            ori_left_img = train_left[duel]
            left_img = ori_left_img.reshape((1,) + ori_left_img.shape)
            aug_img = left_datagen.flow(left_img, batch_size=1)
            left_aug_img = aug_img[0].reshape(ori_left_img.shape)

            # Create one augmented image from the right one
            ori_right_img = train_right[duel]
            right_img = ori_right_img.reshape((1,) + ori_right_img.shape)
            aug_img = right_datagen.flow(right_img, batch_size=1)
            right_aug_img = aug_img[0].reshape(ori_right_img.shape)

            # Add to list
            train_left_aug.append(left_aug_img)
            train_right_aug.append(right_aug_img)
            train_label_aug.append(train_label[duel])
            train_label_score_aug.append(train_label_score[duel])

    # Convert to array
    train_left_aug = np.array(train_left_aug)
    train_right_aug = np.array(train_right_aug)
    train_label_aug = np.array(train_label_aug)
    train_label_score_aug = np.array(train_label_score_aug)
    train_data_aug = [train_left_aug, train_right_aug]

    print("Done\nSaving data ...")
    # Save augmented training dataset as npy
    np.save(os.path.join(aug_folder, "train_left_{}".format(img_size)), np.array(train_left_aug))
    np.save(os.path.join(aug_folder, "train_right_{}".format(img_size)), np.array(train_right_aug))
    np.save(os.path.join(aug_folder, "train_labels{}".format(img_size)), np.array(train_label_aug))
    np.save(os.path.join(aug_folder, "train_labels_score{}".format(img_size)), np.array(train_label_score_aug))
    print("Done")

    return train_data_aug, train_label_aug, train_label_score_aug


def shows_aug_img(aug_array, nb_aug):
    """
    Shows how the augmented images look like.

    :param aug_array: array of augmented images
    :type aug_array: np.array
    :param nb_aug: number of augmented images for one original
    :type nb_aug: int
    :return:
    :rtype:
    """
    # Variable initialization
    nrows = nb_aug + 1
    nb_ori_comp = aug_array.shape[0] // nrows
    ncols = nb_ori_comp
    figsize = [nrows, ncols]
    fig = plt.figure(figsize=figsize)
    ax = []

    for i in range(nrows * ncols):
        # Set original images on the top line
        if i < ncols:
            ax.append(fig.add_subplot(nrows, ncols, i + 1))
            b, g, r = cv2.split(aug_array[i].astype(int))
            rgb_img = cv2.merge([r, g, b])
            plt.imshow(rgb_img)
            ax[-1].set_xticks([])
            ax[-1].set_yticks([])
        # Set augmented photo in a column below their original one
        else:
            line = (i - ncols) % nb_aug
            if line == 0:
                line = nb_aug
            col = (i - ncols) // nb_aug
            pos = line * ncols + col + 1
            ax.append(fig.add_subplot(nrows, ncols, pos))
            b, g, r = cv2.split(aug_array[i].astype(int))
            rgb_img = cv2.merge([r, g, b])
            plt.imshow(rgb_img)
            ax[-1].set_xticks([])
            ax[-1].set_yticks([])
    # Show image
    plt.show()
    # Save figure
    plt.savefig("{}_augmented_images_from_{}_comparisons.jpg".format(nb_aug, nb_ori_comp))


def create_input_from_comparisons_predictions(model_path, img_dir, nb_creation, save_folder):
    """
    Create new data input for ranking model by predicting them with the comparisons model.

    :param model_path: path of the trained comparisons model
    :type model_path: str
    :param img_dir: images directory path
    :type img_dir: str
    :param nb_creation: number of comparisons created
    :type nb_creation: int
    :param save_folder: path of the folder where the predictions dataset folder is stored
    :type save_folder: str
    :return: augmented dataset
    :rtype: tuple(np.array)
    """

    # Variable initialization
    images_path = glob.glob(img_dir + "/*jpg")
    nb_img = len(images_path) - 1
    train_left_pred = []
    train_right_pred = []
    train_label_pred = []
    train_label_score_pred = []

    # Create output folder
    pred_folder = safe_folder_creation(save_folder)

    # Load comparisons model
    comp_model = load_model(model_path)
    img_size = comp_model.input_shape[0][1]

    for _ in range(nb_creation):
        # Select randomly two images
        img_left = Ci.Image(images_path[random.randint(0, nb_img)], build_array=True)
        img_right = Ci.Image(images_path[random.randint(0, nb_img)], build_array=True)

        input_img_left = img_left.preprocess_image(img_size)
        input_img_right = img_right.preprocess_image(img_size)

        # Compute labels
        proba = comp_model.predict([np.array([input_img_left]), np.array([input_img_right])])
        index_max = np.argmax(proba)
        jump_next_pred = False
        if index_max == 0:
            winner = "left"
        elif index_max == 1:
            winner = "right"
        else:
            jump_next_pred = True
            winner = "Draw"

        if not jump_next_pred:
            # Add labels to list
            train_label_pred.append(get_label_comparison_from_string(winner))
            train_label_score_pred.append(get_label_score_from_string(winner))

            # Add images to list
            train_left_pred.append(img_left.array)
            train_right_pred.append(img_right.array)

    # Convert to array
    train_left_pred = np.array(train_left_pred)
    train_right_pred = np.array(train_right_pred)
    train_label_pred = np.array(train_label_pred)
    train_label_score_pred = np.array(train_label_score_pred)
    train_data_pred = [train_left_pred, train_right_pred]

    # Save augmented training dataset as npy
    np.save(os.path.join(pred_folder, "train_left_{}".format(img_size)), train_left_pred)
    np.save(os.path.join(pred_folder, "train_right_{}".format(img_size)), train_right_pred)
    np.save(os.path.join(pred_folder, "train_labels{}".format(img_size)), train_label_pred)
    np.save(os.path.join(pred_folder, "train_labels_score{}".format(img_size)), train_label_score_pred)

    return train_data_pred, train_label_pred, train_label_score_pred
