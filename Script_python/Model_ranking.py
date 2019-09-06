# -*- coding: utf-8 -*-
"""
Script used to define the ranking model and the meta meta network used to trained it.
"""
# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------
import os
import numpy as np

from keras import Model
from keras.applications import VGG19
from keras.layers import Dropout, Flatten, Dense, Subtract, Activation, BatchNormalization, Input
from keras.optimizers import SGD
from keras.models import load_model, model_from_json

from Class_training import simple_training, k_fold, evaluation_test_set
from utils_class import shuffle_unison_arrays


# ----------------------------------------------------------------------------------------------------------------------
# Functions definitions
# ----------------------------------------------------------------------------------------------------------------------
def create_ranking_network(img_size):
    """
    Create ranking network which give a score to an image.

    :param img_size: size of input images during training
    :type img_size: tuple(int)
    :return: ranking network model
    :rtype: keras.Model
    """
    # Create feature extractor from VGG19
    feature_extractor = VGG19(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))
    # for layer in feature_extractor.layers[:-4]:
    #     layer.trainable = False

    # Add dense layers on top of the feature extractor
    inp = Input(shape=(img_size, img_size, 3), name='input_image')
    base = feature_extractor(inp)
    base = Flatten(name='Flatten')(base)

    # Block 1
    base = Dense(32, activation='relu', name='Dense_1')(base)
    base = BatchNormalization(name='BN1')(base)
    base = Dropout(0.490, name='Drop_1')(base)

    # # Block 2
    # base = Dense(128, activation='relu', name='Dense_2')(base)
    # base = BatchNormalization(name='BN2')(base)
    # base = Dropout(0.368, name='Drop_2')(base)

    # Final dense
    base = Dense(1, name="Dense_Output")(base)
    base_network = Model(inp, base, name='Scoring_model')
    return base_network


def create_meta_network(img_size, weights=None):
    """
    Create meta network which is used to to teach the ranking network.

    :param img_size: dimension of input images during training.
    :type img_size: tuple(int)
    :param weights: path to the weights use for initialization
    :type weights: str
    :return: meta network model
    :rtype: keras.Model
    """

    # Create the two input branches
    input_left = Input(shape=(img_size, img_size, 3), name='left_input')
    input_right = Input(shape=(img_size, img_size, 3), name='right_input')
    base_network = create_ranking_network(img_size)
    left_score = base_network(input_left)
    right_score = base_network(input_right)

    # Subtract scores
    diff = Subtract()([left_score, right_score])

    # Pass difference through sigmoid function.
    prob = Activation("sigmoid", name="Activation_sigmoid")(diff)
    model = Model(inputs=[input_left, input_right], outputs=prob, name="Meta_Model")

    if weights:
        print('Loading weights ...')
        model.load_weights(weights)


    sgd = SGD(lr=1e-6, decay=1e-6, momentum=0.393, nesterov=True)
    model.compile(optimizer=sgd, loss="binary_crossentropy", metrics=['accuracy'])

    return model


if __name__ == "__main__":
    # ------------------------------------------------------------------------------------------------------------------
    # Variables initialization
    # ------------------------------------------------------------------------------------------------------------------
    # Define input image size
    IMG_SIZE = 224
    WEIGTHS = r'D:\Guillaume\Ottawa\Data\Training_Models_Results\Ranking\Kfold_08_30_1_Dense_32_Drop_49/weights_k1.h5'
    aaa = create_meta_network(IMG_SIZE, WEIGTHS)

    # Created folder where training results are stored
    folder_path = r"D:\Guillaume\Ottawa\Data\Training_Models_Results\Ranking\Simple_09_03"

    # Load training data
    data_folder = r"D:\Guillaume\Ottawa\Data\Comparisons_npy\08_13"
    train_left = np.load(os.path.join(data_folder, "augmented_2", "train_left_224.npy"))
    train_right = np.load(os.path.join(data_folder, "augmented_2", "train_right_224.npy"))
    train_label = np.load(os.path.join(data_folder, "augmented_2", "train_labels_score_224.npy"))

    # ------------------------------------------------------------------------------------------------------------------
    # Run functions
    # ------------------------------------------------------------------------------------------------------------------
    # Shuffle data
    data_left, data_right, data_label = shuffle_unison_arrays([train_left, train_right, train_label])

    # Simple training
    simple_training(data_left, data_right, data_label, create_meta_network, [IMG_SIZE, WEIGTHS],
                    folder_path, val_split=0.2, epochs=150, batch_size=32)

    # Kfold training
    # k_fold(data_left, data_right, data_label, 3, create_meta_network, [IMG_SIZE],
    #                 folder_path, epochs=80, batch_size=16)

    # Evaluation on test set
    # evaluation_test_set(os.path.join(folder_path, 'fitted_model.h5'), data_folder)
