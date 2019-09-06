# -*- coding: utf-8 -*-
"""
This script is used to recap all the main functions and steps of the production of data sets used to train
a siamese model with two images as inputs and a comparison as label.
"""
# ======================================================================================================================
# ------------------------------------- Test Image class ---------------------------------------------------------------
# ======================================================================================================================
from keras.models import load_model
from Class_Image import Image
from os.path import abspath

SEG_GREEN_MODEL = load_model(abspath("Examples/Models/trained_green_sky_4_classes_128.h5"))
SEG_SNOW_MODEL = load_model(abspath("Examples/Models/trained_snow_2_classes_256.h5"))
IMG_SNOW = abspath("Examples/Test_images/-75.444059_45.466568_aEfJTEUjC46NYCfS69xmGA_000_2016-10-15T21-13-22.jpg")

image_snow = Image(IMG_SNOW, build_array=True, build_green_segmentation=True, green_seg_model=SEG_GREEN_MODEL,
            build_snow_segmentation=True, snow_seg_model=SEG_SNOW_MODEL)
print(image_snow)
image_snow.is_blurry()
image_snow.is_overexposed()
image_snow.is_under_green_ratio(0.83)
image_snow.is_wrong_exposed()
image_snow.is_snowy()
print(image_snow)


# ======================================================================================================================
# ------------------------------------- Test Panorama class ------------------------------------------------------------
# ======================================================================================================================
from Class_Image import Panorama
from os.path import abspath

SAVE_DIR = abspath('Examples/Test_images')
PANO = abspath("Examples/Test_images/-75.497094_45.457925__m3YDpWSlFIhCXx922pZLw_999_2019-03-08T13-06-51.jpg")

pano = Panorama(PANO, build_array=True)
pano.create_4_images(SAVE_DIR)
pano.plot_created_images("Examples/Test_images/4.jpg")


# ======================================================================================================================
# ------------------------------------- Test Download class ------------------------------------------------------------
# ======================================================================================================================
from Class_Download import Download
from os.path import abspath

PHOTO_DIR = abspath("Examples/Photos")
LAYER_PATH = abspath("Examples/Shapefile/test.shp")
MAP_KEY = ""
GSV_KEY = ""

d = Download(PHOTO_DIR, LAYER_PATH)
d.download_mapillary(MAP_KEY)
d.download_gsv(GSV_KEY, end=5)
d.create_images_from_mapillary_panos()


# ======================================================================================================================
# ------------------------------------- Test Preprocessing_images class ------------------------------------------------
# ======================================================================================================================
from keras.models import load_model
from Class_Preprocessing import PreprocessingImages
from os.path import abspath

SEG_GREEN_MODEL = load_model(abspath("Examples/Models/trained_green_sky_4_classes_128.h5"))
SEG_SNOW_MODEL = load_model(abspath("Examples/Models/trained_snow_2_classes_256.h5"))
MAPILLARY_JPG_DIR = abspath("Examples/Photos/Mapillary_images")

p = PreprocessingImages(MAPILLARY_JPG_DIR)
p.snow_analysis(SEG_SNOW_MODEL)
p.green_analysis(SEG_GREEN_MODEL, 0.83, SEG_SNOW_MODEL)


# ======================================================================================================================
# ------------------------------------- Create web images directory ----------------------------------------------------
# ======================================================================================================================
from utils_class import copy_random_images, copy_img_from_filelist
from os.path import abspath, join

MAPILLARY_JPG_DIR = abspath("Examples/Photos/Mapillary_images")
GOOD_IMG_DIR = abspath("Examples/Photos/Good_images")
WEB_IMG_DIR = abspath("Examples/Photos/Web_images")

copy_img_from_filelist(join(MAPILLARY_JPG_DIR, "Listsgreen", "11-All_good.txt"), MAPILLARY_JPG_DIR, GOOD_IMG_DIR)
copy_random_images(GOOD_IMG_DIR, WEB_IMG_DIR, nb_max=10, size_max=10e6)


# ======================================================================================================================
# ------------------------------------- Test Database class ------------------------------------------------------------
# ======================================================================================================================
from Class_Database import Database
from os.path import abspath, join


DB_NAME = ""
SERVER = ""
USERNAME = ""
PASSWORD = ""
LOCAL  = False
SSL_CA = abspath("Examples/SSL/BaltimoreCyberTrustRoot.crt.pem")
DATA_FOLDER = abspath("Examples/Comparisons_npy")
COMP_CSV = join(DATA_FOLDER, "duels_date.csv")
WEB_IMG_DIR = abspath("Examples/Photos/Web_Images")

da = Database(DB_NAME, SERVER, USERNAME, PASSWORD, LOCAL, SSL_CA, WEB_IMG_DIR)
da.complete_images_table("images_test")
da.get_duels(COMP_CSV, "duels_question_1")


# ======================================================================================================================
# ------------------------------------- Create input data from csv -----------------------------------------------------
# ======================================================================================================================
from Class_Preprocessing import preprocessing_duels
from os.path import abspath, join

WEB_IMG_DIR = abspath("Examples/Photos/Web_Images")
DATA_FOLDER = abspath("Examples/Comparisons_npy")
COMP_CSV = join(DATA_FOLDER, "duels_date.csv")

left, right, labels, labels_score = preprocessing_duels(COMP_CSV, 224, WEB_IMG_DIR, DATA_FOLDER, 0.2)


# ======================================================================================================================
# ------------------------------------- Create augmented data from input -----------------------------------------------
# ======================================================================================================================
from Class_Preprocessing import data_aug
from os.path import abspath, join
import numpy as np

DATA_FOLDER = abspath("Examples/Comparisons_npy")
DATA_FOLDER_AUG = abspath("Examples/Comparisons_npy/augmented_1")

left = np.load(join(DATA_FOLDER, "train", "train_left_224.npy"))
right = np.load(join(DATA_FOLDER, "train", "train_right_224.npy"))
labels = np.load(join(DATA_FOLDER, "train", "train_labels_224.npy"))
labels_score = np.load(join(DATA_FOLDER, "train", "train_labels_score_224.npy"))

train_data_aug, train_label_aug, train_label_score_aug = data_aug(left, right, labels, labels_score, 1, DATA_FOLDER_AUG)


# ======================================================================================================================
# ------------------------------------- Simple training ----------------------------------------------------------------
# ======================================================================================================================
from Class_training import simple_training
from Model_comparisons import comparisons_model

DATA_FOLDER = abspath("Examples/Comparisons_npy")
TRAIN_RESULTS = abspath("Examples/Training_results/Simple")

left = np.load(join(DATA_FOLDER, "train", "train_left_224.npy"))
right = np.load(join(DATA_FOLDER, "train", "train_right_224.npy"))
labels = np.load(join(DATA_FOLDER, "train", "train_labels_224.npy"))
labels_score = np.load(join(DATA_FOLDER, "train", "train_labels_score_224.npy"))

simple_training(left, right, labels, comparisons_model, [224], TRAIN_RESULTS, 0.2, 5, 8)


# ======================================================================================================================
# ------------------------------------- K-fold training ----------------------------------------------------------------
# ======================================================================================================================
from Class_training import k_fold
from Model_ranking import create_meta_network

DATA_FOLDER = abspath("Examples/Comparisons_npy/augmented_1")
TRAIN_RESULTS = abspath("Examples/Training_results/Kfold")

left = np.load(join(DATA_FOLDER, "train", "train_left_224.npy"))
right = np.load(join(DATA_FOLDER, "train", "train_right_224.npy"))
labels = np.load(join(DATA_FOLDER, "train", "train_labels_224.npy"))
labels_score = np.load(join(DATA_FOLDER, "train", "train_labels_score_224.npy"))

k_fold(left, right, labels, 3, create_meta_network, [224], TRAIN_RESULTS, 0.2, 5, 8)


# ======================================================================================================================
# ------------------------------------- Predictions Ranking score ------------------------------------------------------
# ======================================================================================================================
from Class_training import ranking_layer
from os.path import abspath

MODEL_PATH = abspath('Examples/Training_results/Kfold/fitted_ranking_model.h5')
IMG_DIR = abspath('Examples/Photos/Good_images')
CSV_PATH = abspath('Examples/Shapefile/ranking_score_layer.csv')

ranking_layer(MODEL_PATH, IMG_DIR, CSV_PATH)

# ======================================================================================================================
# ------------------------------------- Predictions Trueskill score ----------------------------------------------------
# ======================================================================================================================
from Class_training import trueskill_layer
from os.path import abspath

MODEL_PATH = abspath('Examples/Training_results/Simple/fitted_model.h5')
IMG_DIR = abspath('Examples/Photos/Good_images')
CSV_PATH = abspath('Examples/Shapefile/trueskill_score_layer.csv')

trueskill_layer(MODEL_PATH, IMG_DIR, CSV_PATH)
