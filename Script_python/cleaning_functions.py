# -*- coding: utf-8 -*-
"""
Lists of intermediate functions used by others scripts
"""
# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------
import os
import csv
import glob
import fnmatch
import re

import progressbar
import mysql.connector

from utils_class import write_list_to_textfile
from Class_Database import connection_database, commit_query, end_connection, insert_into


# ----------------------------------------------------------------------------------------------------------------------
# Functions definitions
# ----------------------------------------------------------------------------------------------------------------------
def list_from_textfile(textfile_path):
    """
    Create a list of strings from the information in the text file. Removes the \n at the end of a line.

    :param textfile_path: path of the text file
    :type textfile_path: str
    :return: list of information in text file
    :rtype: list
    """
    with open(textfile_path) as f:
        file_list = f.read().splitlines()
    return file_list


def remove_comparison_bad_image(csv_path, list_wrong_image):
    """
    Create a new csv without comparisons including a bad image

    Csv has the following format : key_image_1, key_image_2, winner, ip_address
    :param csv_path: csv path
    :type csv_path: str
    :param list_wrong_image: list of excluded images paths
    :type list_wrong_image: list
    :return: number of comparisons removed
    :rtype: int
    """
    removed_images = 0
    nb_comp = 0
    list_comp_removed = []
    with open(csv_path, 'r') as f:
        lines = csv.reader(f, delimiter=',')
        pbar = progressbar.ProgressBar()
        with open(csv_path[:-4] + "_clean.csv", 'w') as new_f:
            new_f_writer = csv.writer(new_f, delimiter=",", lineterminator="\n")
            for line in pbar(lines):
                if line[0] not in list_wrong_image and line[1] not in list_wrong_image:
                    new_f_writer.writerow(line)
                else:
                    removed_images += 1
                    list_comp_removed.append(nb_comp)
                nb_comp += 1
    print(removed_images, nb_comp)


def clean_comparison_csv(csv_path, list_file_path):
    """
    Remove the comparisons including one image that has been discarded.

    :param csv_path: csv path
    :type csv_path: str
    :param list_file_path: path to the file containing the list of the discarded images
    :type list_file_path: str
    """
    ls = list_from_textfile(list_file_path)
    ls_key = []
    for filename in ls:
        key = filename[21:43]
        ls_key.append(key)
    remove_comparison_bad_image(csv_path, ls_key)


def check_images_presences(csv_path, image_folder):
    """
    List all the image path used in the comparisons file

    :param csv_path: comparison csv path
    :type csv_path: str
    :param image_folder: path of the image folder
    :type image_folder: str
    :return: list of image path
    :rtype: list
    """
    # Variable initialization
    all_img = []

    # Loop through each comparison
    with open(csv_path, 'r') as csvfileReader:
        reader = csv.reader(csvfileReader, delimiter=',')
        pbar = progressbar.ProgressBar()
        for line in pbar(reader):
            if line != [] and line[2] != 'No preference':
                left_image_path = get_filename_from_key(line[0], image_folder)
                right_image_path = get_filename_from_key(line[1], image_folder)
                # Add image path to the list
                all_img.append(left_image_path)
                all_img.append(right_image_path)
    return all_img


# ------------------------- Database cleaning --------------------------------------------------------------------------
def add_comp(csv_path, db_name, host, user, password, local, ssl_ca):
    with open(csv_path, 'r') as f:
        lines = f.read().splitlines()

    # Connection to database
    conn, cursor = connection_database(db_name, host, user, password, local, ssl_ca)

    for line in lines:
        key1, key2, winner, ip = line.split(',')
        columns = ("key1", "key2", "winner", "ip_address")
        values = (key1, key2, winner, ip)

        try:
            insert_into(cursor, "duels", columns, values)
        except mysql.connector.errors.IntegrityError:
            print("Comparison already in table", key1, key2, winner, ip)
            continue

            # Validate queries
    commit_query(conn)

    # End connection to database
    end_connection(conn, cursor)


def update_users(csv_path, db_name, host, user, password, local, ssl_ca):
    with open(csv_path, 'r') as f:
        lines = f.read().splitlines()

    list_ip = []
    for line in lines:
        _, _, _, ip = line.split(',')
        if ip not in list_ip:
            list_ip.append(ip)

    # Connection to database
    conn, cursor = connection_database(db_name, host, user, password, local, ssl_ca)
    for ip in list_ip:
        try:
            # Check if users iss inside table:
            query = "SELECT * FROM users WHERE ip_address='{}';".format(ip)
            cursor.execute(query)
            result = cursor.fetchall()
            if not result:
                # Add users to data
                columns = ("comparisons", "ip_address")
                values = (0, ip)
                insert_into(cursor, "users", columns, values)
                # commit_query(conn)

            # Update number
            query = "UPDATE users SET comparisons=(SELECT COUNT(*) from duels WHERE ip_address='{}') WHERE ip_address='{}';".format(ip, ip)
            cursor.execute(query)
        except mysql.connector.errors.IntegrityError as e:
            print(e)
            continue

    # Validate queries
    commit_query(conn)
    # End connection to database
    end_connection(conn, cursor)


# ------------------------- Images filename cleaning -------------------------------------------------------------------
def remove_heading_(map_dir):
    img_list = glob.glob(map_dir + "/*.jpg")
    pbar = progressbar.ProgressBar()
    for img in pbar(img_list):
        path = os.path.normpath(img)
        parts = path.split(os.sep)
        img = img[:43] + img[47:]
        parts[0] = parts[0] + os.sep
        new_name = os.path.join(*parts)
        os.rename(img, new_name)


def check_len_71(img_dir):
    images_list = fnmatch.filter(os.listdir(img_dir), '*.jpg')
    lengths = []
    nb71 = 0
    nbnot71 = 0
    for img in images_list:
        if len(img) == 71:
            nb71 += 1
        else:
            nbnot71 += 1
            if len(img) not in lengths:
                lengths.append(len(img))
    print("Length different of 71", nbnot71)
    print("Length equal to 71", nb71)
    print("List lengths: ", lengths)


def add_heading_(map_dir, pano=False):
    img_list = fnmatch.filter(os.listdir(map_dir), '*.jpg')
    pbar = progressbar.ProgressBar()
    for img in pbar(img_list):
        if len(img) == 67:
            if pano:
                new_img = img[:43] + "_999" + img[43:]
            else:
                new_img = img[:43] + "_000" + img[43:]
            new_name = os.path.join(map_dir, new_img)
            old_name = os.path.join(map_dir, img)
            os.rename(old_name, new_name)


def add_heading_textfile(textpath, pano=False):
    ln = list_from_textfile(textpath)
    nl = []
    for old_name in ln:
        if len(old_name) == 67:
            if pano:
                new_name = old_name[:43] + "_999" + old_name[43:]
            else:
                new_name = old_name[:43] + "_000" + old_name[43:]
            nl.append(new_name)
        else:
            print(len(old_name), old_name)

    path, base = os.path.split(textpath)
    new_base = base[:-4] + "_heading" + base[-4:]
    new_file = os.path.join(path, new_base)
    write_list_to_textfile(nl, new_file)


def add_heading_folder_text(folder):
    txt_list = glob.glob(os.path.join(folder, '*.txt'))
    for txt in txt_list:
        add_heading_textfile(txt)


def check_double_image(filename):
    if re.search("\(1\)", filename):
        return True
    else:
        return False


def check_double_image_folder(img_dir):
    images_list = glob.glob(os.path.join(img_dir, '*.jpg'))
    l_double = []
    for img in images_list:
        if check_double_image(img):
            l_double.append(img)
    print(len(l_double))
    return l_double


def correct_heading(old_basename):
    new_basename = old_basename
    if re.search("_0_", old_basename):
        for e in re.finditer("_0_", old_basename):
            if e.start() == 43:
                new_basename = old_basename.replace("_0_", "_000_")
    elif re.search("_90_", old_basename) and re.search("_90_", old_basename).start() == 43:
        for e in re.finditer("_90_", old_basename):
            if e.start() == 43:
                new_basename = old_basename.replace("_90_", "_090_")
    return new_basename


def correct_heading_file(file_path):
    ln = list_from_textfile(file_path)
    nl = []
    for old_name in ln:
        nl.append(correct_heading(old_name))
    path, base = os.path.split(file_path)
    new_base = base[:-4] + "_heading" + base[-4:]
    new_file = os.path.join(path, new_base)
    write_list_to_textfile(nl, new_file)


def correct_heading_file_folder(folder_path):
    txt_list = glob.glob(os.path.join(folder_path, '*.txt'))
    for txt in txt_list:
        correct_heading_file(txt)


def check_not_71(filename):
    if len(filename) == 71:
        return False
    else:
        return True


def global_check_filename(filename):
    if check_not_71(filename) or check_double_image(filename):
        return False
    else:
        return True
