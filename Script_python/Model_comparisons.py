# -*- coding: utf-8 -*-
"""
Script used to define the comparisons model and trained it.
"""
# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------
import os
import numpy as np

from keras import Input, Model
from keras.applications import VGG19
from keras.layers import concatenate, Conv2D, Dropout, Flatten, Dense, BatchNormalization
from keras.optimizers import SGD

from Class_training import simple_training
from utils_class import shuffle_unison_arrays


# ----------------------------------------------------------------------------------------------------------------------
# Functions definitions
# ----------------------------------------------------------------------------------------------------------------------
def comparisons_model(img_size, weights=None):
    """
    Create comparisons network which reproduce the choice in an images duel.

    :param img_size: size of input images during training
    :type img_size: tuple(int)
    :param weights: path to the weights use for initialization
    :type weights: str
    :return: ranking comparisons model
    :rtype: keras.Model
    """
    vgg_feature_extractor = VGG19(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))

    # Fine tuning by freezing the last 4 convolutional layers of VGG19 (last block)
    for layer in vgg_feature_extractor.layers[:-4]:
        layer.trainable = False

    # Definition of the 2 inputs
    img_a = Input(shape=(img_size, img_size, 3), name="left_image")
    img_b = Input(shape=(img_size, img_size, 3), name="right_image")
    out_a = vgg_feature_extractor(img_a)
    out_b = vgg_feature_extractor(img_b)

    # Concatenation of the inputs
    concat = concatenate([out_a, out_b])

    # Add convolution layers on top
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name="Conv_1")(concat)
    x = Dropout(0.43, name="Drop_1")(x)
    # x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name="Conv_2")(x)
    x = Dropout(0.49, name="Drop_2")(x)
    # x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(2, activation='softmax', name="Final_dense")(x)

    classification_model = Model([img_a, img_b], x)

    if weights:
        classification_model.load_weights(weights)

    sgd = SGD(lr=1e-5, decay=1e-6, momentum=0.695, nesterov=True)
    classification_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return classification_model


if __name__ == "__main__":
    # ------------------------------------------------------------------------------------------------------------------
    # Variables initialization
    # ------------------------------------------------------------------------------------------------------------------
    width = 224
    save_folder = r"D:\Guillaume\Ottawa\Data\Comparisons_npy\09_04"
    data_left = np.load(os.path.join(save_folder, "train", "train_left_224.npy"))
    data_right = np.load(os.path.join(save_folder, "train", "train_right_224.npy"))
    data_label = np.load(os.path.join(save_folder, "train", "train_labels_224.npy"))
    # ------------------------------------------------------------------------------------------------------------------
    # Run functions
    # ------------------------------------------------------------------------------------------------------------------
    # Build npy
    # trainLeft, trainRight, train_label = preprocessing_duels(csv_path, width, height, img_dir, save_folder, 0.1)

    # Shuffling data
    data_left_shuffled, data_right_shuffled, data_label_shuffled = shuffle_unison_arrays([data_left, data_right, data_label])

    # Simple training
    folder_path = r"D:\Guillaume\Ottawa\Data\Training_Models_Results\Comparisons_Trueskill/Simple_retrain"
    simple_training(data_left_shuffled, data_right_shuffled, data_label_shuffled, comparisons_model, [width], folder_path, val_split=0.2, epochs=300, batch_size=32)

    # K fold training
    # folder_path = r"D:\Guillaume\Ottawa\Data\All_Results_to_sort\kfold-drop-out_normal"
    # k_fold(data_left, data_right, data_label, 5, comparisons_model, [width], folder_path)

    # Evaluation on test data
    # save_folder = "D:/Guillaume/Ottawa/Script_python/Results/npy_13_08/"
    # model_path = os.path.join(save_folder, "k-fold", "fitted_comparisons_k5_clear_session.h5")
    # evaluation_test_set(model_path, save_folder, mode="comparisons")
