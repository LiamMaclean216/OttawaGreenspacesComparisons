# -*- coding: utf-8 -*-
"""
This script is used to recap all the main functions and steps of the production of data sets used to train
a siamese model with two images as inputs and a comparison as label.
"""

# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------
from os.path import abspath, join
import numpy as np
from keras.models import load_model
from Class_Download import Download
from Class_Database import Database
from Class_Image import Image, Panorama
from Class_Preprocessing import PreprocessingImages, preprocessing_duels, data_aug
from Class_training import trueskill_layer, ranking_layer, simple_training, k_fold
from Model_comparisons import comparisons_model
from Model_ranking import create_meta_network
from utils_class import copy_random_images, copy_img_from_filelist


# ----------------------------------------------------------------------------------------------------------------------
# Variables initialization
# ----------------------------------------------------------------------------------------------------------------------
# Models
SEG_GREEN_MODEL = load_model(abspath("Examples/Models/trained_green_sky_4_classes_128.h5"))
SEG_SNOW_MODEL = load_model(abspath("Examples/Models/trained_snow_2_classes_256.h5"))
MODEL_COMPARISONS = abspath('Examples/Training_results/Simple/fitted_model.h5')
MODEL_RANKING = abspath('Examples/Training_results/Kfold/fitted_ranking_model.h5')

# Images
IMG_SNOW = abspath("Examples/Test_images/-75.444059_45.466568_aEfJTEUjC46NYCfS69xmGA_000_2016-10-15T21-13-22.jpg")
PANO = abspath("Examples/Test_images/-75.497094_45.457925__m3YDpWSlFIhCXx922pZLw_999_2019-03-08T13-06-51.jpg")

# Images folder
SAVE_DIR = abspath('Examples/Photos/Mapillary_jpg_from_pano')
PHOTO_DIR = abspath("Examples/Photos/Mapillary_jpg")
WEB_IMG_DIR = abspath("Examples/Photos/Web_Images")
MAPILLARY_JPG_DIR = abspath("Photos/Mapillary_images")
GOOD_IMG_DIR = abspath("Examples/Photos/Good_images")

# Layers
LAYER_PATH = abspath("Examples/Shapefile/test.shp")
CSV_RANKING = abspath('Examples/Shapefile/ranking_score_layer.csv')
CSV_TRUESKILL = abspath('Examples/Shapefile/trueskill_score_layer.csv')

# Keys
MAP_KEY = ""
GSV_KEY = ""

# Database
DB_NAME = ""
SERVER = ""
USERNAME = ""
PASSWORD = ""
LOCAL = False
SSL_CA = abspath("SSL/BaltimoreCyberTrustRoot.crt.pem")

#  Data
DATA_FOLDER_AUG = abspath("Comparisons_npy/augmented_1")
DATA_FOLDER = abspath("Comparisons_npy")
COMP_CSV = join(DATA_FOLDER, "duels_date.csv")

# Results
TRAIN_RESULTS_KFOLD = abspath("Training_results/Kfold")
TRAIN_RESULTS_SIMPLE = abspath("Training_results/Simple")

# ----------------------------------------------------------------------------------------------------------------------
# Run functions
# ----------------------------------------------------------------------------------------------------------------------

# ------------------------------------- Test Image class ---------------------------------------------------------------
image_snow = Image(IMG_SNOW, build_array=True, build_green_segmentation=True, green_seg_model=SEG_GREEN_MODEL,
                   build_snow_segmentation=True, snow_seg_model=SEG_SNOW_MODEL)
print(image_snow)
image_snow.is_blurry()
image_snow.is_overexposed()
image_snow.is_under_green_ratio(0.83)
image_snow.is_wrong_exposed()
image_snow.is_snowy()
print(image_snow)

# ------------------------------------- Test Panorama class ------------------------------------------------------------
pano = Panorama(PANO, build_array=True)
pano.create_4_images(SAVE_DIR)
pano.plot_created_images()

# ------------------------------------- Test Download class ------------------------------------------------------------
d = Download(PHOTO_DIR, LAYER_PATH)
d.download_mapillary(MAP_KEY)
d.download_gsv(GSV_KEY)
d.create_images_from_mapillary_panos()

# ------------------------------------- Test Preprocessing_images class ------------------------------------------------
p = PreprocessingImages(MAPILLARY_JPG_DIR)
p.snow_analysis(SEG_SNOW_MODEL)
p.green_analysis(SEG_GREEN_MODEL, 0.83, SEG_SNOW_MODEL)

# ------------------------------------- Create web images directory ----------------------------------------------------
copy_img_from_filelist(join(MAPILLARY_JPG_DIR, "Listsgreen", "11-All_good.txt"), MAPILLARY_JPG_DIR, GOOD_IMG_DIR)
copy_random_images(GOOD_IMG_DIR, WEB_IMG_DIR, nb_max=10, size_max=10e6)

# ------------------------------------- Test Database class ------------------------------------------------------------
da = Database(DB_NAME, SERVER, USERNAME, PASSWORD, LOCAL, SSL_CA, WEB_IMG_DIR)
da.complete_images_table("images_test")
da.get_duels(COMP_CSV, "duels_question_1")

# ------------------------------------- Create input data from csv -----------------------------------------------------
left, right, labels, labels_score = preprocessing_duels(COMP_CSV, 224, WEB_IMG_DIR, DATA_FOLDER, 0.2)

# ------------------------------------- Create augmented data from input -----------------------------------------------
train_data_aug, train_label_aug, train_label_score_aug = data_aug(left, right, labels, labels_score, 1, DATA_FOLDER_AUG)

# ------------------------------------- Simple training ----------------------------------------------------------------
left = np.load(join(DATA_FOLDER, "train", "train_left_224.npy"))
right = np.load(join(DATA_FOLDER, "train", "train_right_224.npy"))
labels = np.load(join(DATA_FOLDER, "train", "train_labels_224.npy"))

simple_training(left, right, labels, comparisons_model, [224], TRAIN_RESULTS_SIMPLE, 0.2, 5, 8)

# ------------------------------------- K-fold training ----------------------------------------------------------------
left = np.load(join(DATA_FOLDER, "train", "train_left_224.npy"))
right = np.load(join(DATA_FOLDER, "train", "train_right_224.npy"))
labels_score = np.load(join(DATA_FOLDER, "train", "train_labels_score_224.npy"))

k_fold(left, right, labels, 3, create_meta_network, [224], TRAIN_RESULTS_KFOLD, 0.2, 5, 8)

# ------------------------------------- Predictions Ranking score ------------------------------------------------------
ranking_layer(MODEL_COMPARISONS, WEB_IMG_DIR, CSV_RANKING)

# ------------------------------------- Predictions Trueskill score ----------------------------------------------------
trueskill_layer(MODEL_RANKING, WEB_IMG_DIR, CSV_TRUESKILL)
