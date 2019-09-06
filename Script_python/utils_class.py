# -*- coding: utf-8 -*-
"""
List of functions used by the others classes
"""
# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np
import time
from random import shuffle
import os
import glob
from shutil import copy
import random
from colorama import Back, Style
from progressbar import ProgressBar

# ============= Timer decorator ========================================================================================
def timeit(function):
    """
    Decorator who prints the execution time of a function
    :param function: function to be executed
    :type function: function
    :return: function's return
    :rtype:function's return type
    """
    def timed(*args, **kw):
        ts = time.time()
        print('\nExecuting %r ' % function.__name__)
        result = function(*args, **kw)
        te = time.time()
        print('\n%r executed in %2.2f s' % (function.__name__, (te - ts)))
        return result
    return timed


def list_from_lists(*args):
    """
    Combine lists into an unique one and save it into a text file.
    The different lists are stored in text files.
    :param args: name of text files containing a list
    :type args: tuple(list)
    :return: combined list
    :rtype: list
    """
    final_list = args[0]
    for l_name in args[1:]:
        for name in l_name:
            if name not in final_list:
                final_list.append(name)
    return final_list


def write_list_to_textfile(simple_list, filename):
    """
    Write the elements of a simple list in a text file with one element per line.
    :param simple_list: list to be written in a file
    :type simple_list: list
    :param filename: filename or complete path of the created file
    :type filename: str
    """
    with open(filename, 'w+') as f:
        for elem in simple_list:
            f.write(elem + "\n")


def shuffle_unison_arrays(array_list):
    """
    Shuffle the arrays of the list accordingly. Arrays must have same first dimension

    :param array_list: list of arrays
    :type array_list: list
    :return: list of shuffled arrays
    :rtype: list
    """
    # Variable initialization
    shuffled_arrays = []

    # Check arrays first dimension
    dim = array_list[0].shape[0]
    for array in array_list:
        assert array.shape[0] == dim

    # Create indices list and shuffle it
    indices = np.arange(0, dim)
    shuffle(indices)

    # Create shuffled arrays
    for array in array_list:
        shuffled_array = array[indices]
        shuffled_arrays.append(shuffled_array)

    return shuffled_arrays


def mapillary_to_sql_date(date):
    """
    Converts a Mapillary format date to a SQL format date.

    :param date: Mapillary format date
    :type date: str
    :return: SQL format date
    :rtype: str
    """
    date_sql = date[:10] + " " + date[11:13] + ":" + date[14:16] + ":" + date[17:]
    return date_sql


def season_from_date(date):
    """
    Returns the season for a given date.

    Summer starts on the 1st of May and finishes on the 31st of October.
    Winter starts on the 1st of November and finishes on the 30th of April.
    :param date: date to be checked
    :type date: str
    :return: season of the date
    :rtype: str
    """
    month = int(date[5:7])
    if 4 < month < 11:
        return "summer"
    else:
        return "winter"


def bytes_convert(size):
    """
    Converts size from bits to bytes.

    :param size: size in bits
    :type size: float
    :return: size in bytes
    :rtype: float
    """
    if size > 2 ** 30:
        return "{:.2f}GB".format(size / 2 ** 30)
    else:
        if size > 2 ** 20:
            return "{:.2f}MB".format(size / 2 ** 20)
        else:
            if size > 2 ** 10:
                return "{:.2f}kB".format(size / 2 ** 10)
            else:
                return size


def copy_random_images(img_dir, out_dir, nb_max=10000, size_max=1e9, size_min=30000):
    """
    Randomly select and copy to a new directory images of a directory.

    The number of selected images can be limited by a threshold or by the total size of the selected images.
    A minimum image size is required to remove images that don't have useful information.

    :param img_dir: image directory path
    :type img_dir: str
    :param out_dir: path of the directory where selected images are saved
    :type out_dir: str
    :param nb_max: maximum number of selected images
    :type nb_max: int
    :param size_max: maximum size in bits
    :type size_max: float
    :param size_min: minimum image size in bits
    :type size_min: float
    """

    # Variable initialization
    images = glob.glob(os.path.join(img_dir, "*.jpg"))
    nb_images = 0
    size = 0
    nb_iter = 0
    nb_iter_max = len(images) + 5

    # Create out directory if need be
    out_dir = safe_folder_creation(out_dir)

    while size < size_max and nb_images < nb_max and images and nb_iter < nb_iter_max:
        # Print processing
        if nb_images % 3 == 0:
            print("Processing .", end="\r", flush=True)
        elif nb_images % 3 == 1:
            print("Processing ..", end="\r", flush=True)
        else:
            print("Processing ...", end="\r", flush=True)

        img_path = random.choice(images)
        img_basename = os.path.basename(img_path)
        img_size = os.path.getsize(img_path)
        if img_size > size_min and not os.path.exists(os.path.join(out_dir, img_basename)):
            copy(img_path, out_dir)
            nb_images += 1
            size += img_size
            images.remove(img_path)
            nb_iter_max -= 1
        else:
            nb_iter += 1

    # Choose stopping message to print
    if size >= size_max:
        print("Size max reached !")
    elif nb_images >= nb_max:
        print("Number maximum of images reached !")
    elif not images:
        print("All the images of the data folder have been copied !")
    elif nb_iter >= nb_iter_max:
        print("Too many tries without copying images !")
    print("Number of copied images : {}".format(nb_images))
    print("Total size of copied images : {}".format(bytes_convert(size)))


def copy_img_from_filelist(file_list, image_folder, new_folder):
    """
    Copy the images listed in the file_list to the new_folder that will be created.

    The images listed in the file_list are stored in the image_folder.
    :param file_list: file with one image filename per line
    :type file_list: str
    :param image_folder: folder containing the images
    :type image_folder: str
    :param new_folder: folder where the images are copied
    :type new_folder: str
    """

    # Create output directory
    new_folder = safe_folder_creation(new_folder)

    # Read file
    with open(file_list) as f:
        image_list = f.read().splitlines()

    # Copy images
    pbar = ProgressBar()
    for img in pbar(image_list):
        copy(os.path.join(image_folder, img), new_folder)


def safe_folder_creation(folder_path):
    """
    Safely create folder and return the new path value if a change occurred.

    :param folder_path: proposed path for the folder that must be created
    :type folder_path: str
    :return: path of the created folder
    :rtype: str
    """
    # Boolean initialization
    folder = True

    while folder:
        # Check path validity
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            folder = False
        else:
            folder_path = input(Back.RED + 'Folder already exists : {}\n Please enter a new path !'.format(folder_path)
                                + Style.RESET_ALL)
    return folder_path
