# -*- coding: utf-8 -*-
"""
Script used to make a hyperparameters optimization of the ranking model.

The load_data and model functions are in the format asked by the Hyperas library.
Finally, the results and the inputs are save in the result folder that will be created.
"""
# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------
import os
import numpy as np

from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperopt import STATUS_OK, tpe, Trials
from keras import Input, Model, Sequential
from keras.applications import VGG19
from keras.layers import Dropout, Flatten, Dense, Subtract, Activation, BatchNormalization
from keras.optimizers import SGD

from Class_training import save_hyperas_results


# ----------------------------------------------------------------------------------------------------------------------
# Functions definitions
# ----------------------------------------------------------------------------------------------------------------------
def load_data():
    """
    Load the data used to train the ranking model.

    :return: training data and labels
    :rtype : tuple(np.array)
    """

    save_folder = r"D:\Guillaume\Ottawa\Data\Comparisons_npy\08_13"
    print("Loading left images ...")
    data_left = np.load(os.path.join(save_folder, "train", "train_left_224.npy"))
    print("Done\nLoading right images ...")
    data_right = np.load(os.path.join(save_folder, "train", "train_right_224.npy"))
    print("Done\nLoading labels ...")
    data_label = np.load(os.path.join(save_folder, "train", "train_labels_score_224.npy"))
    print('Data download finished !')
    data = [data_left, data_right]

    return data, data_label


def model(data, data_label):
    """
    Defines the ranking model, all hyperparameters in double brackets will be optimize by Hyperas.

    :return: a dictionary with following keys :
                - loss : the metrics function to be minimized by Hyperopt.
                - status : a boolean that tells if everything went fine.
                - model : the model on which hyperparameters optimization occurs.
    """
    img_size = 224
    input_dim = (img_size, img_size, 3)
    img_a = Input(shape=input_dim, name="left_image")
    img_b = Input(shape=input_dim, name="right_image")

    vgg_feature_extractor = VGG19(weights='imagenet', include_top=False, input_shape=input_dim)
    # for layer in vgg_feature_extractor.layers[:-4]:
    #     layer.trainable = False

    base_network = Sequential()
    base_network.add(vgg_feature_extractor)
    base_network.add(Flatten())

    base_network.add(Dense({{choice([32, 64, 128])}}, activation='relu', name="Dense1"))
    base_network.add(BatchNormalization(name='BN1'))
    base_network.add(Dropout({{uniform(0, 0.5)}}, name="Drop1"))

    if {{choice(['one_dense', 'two_dense'])}} == 'two_dense':
        base_network.add(Dense({{choice([32, 64, 128])}}, activation='relu', name="Dense2"))
        base_network.add(BatchNormalization(name='BN2'))
        base_network.add(Dropout({{uniform(0, 0.5)}}, name="Drop2"))

    base_network.add(Dense(1, name="Final_dense"))
    left_score = base_network(img_a)
    right_score = base_network(img_b)

    diff = Subtract()([left_score, right_score])
    prob = Activation("sigmoid", name="Activ_sigmoid")(diff)

    ranking_model = Model([img_a, img_b], prob)

    sgd = SGD(lr={{choice([1e-4, 1e-5, 1e-6])}}, decay={{choice([1e-4, 1e-5, 1e-6])}},
              momentum={{uniform(0, 0.9)}}, nesterov=True)
    ranking_model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    result = ranking_model.fit(
        [data[0], data[1]],
        data_label,
        batch_size=32,
        epochs=30,
        validation_split=0.2)

    validation_acc = np.amax(result.history['val_acc'])
    print('Best validation acc of epoch:', validation_acc)

    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': ranking_model}


if __name__ == "__main__":
    # ------------------------------------------------------------------------------------------------------------------
    # Variables initialization
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    result_folder = r'D:\Guillaume\Ottawa\Data\Training_Models_Results\Ranking\Hyperas_08_30'

    # ------------------------------------------------------------------------------------------------------------------
    # Run functions
    # ------------------------------------------------------------------------------------------------------------------
    best_run, best_model = optim.minimize(model=model,
                                          data=load_data,
                                          algo=tpe.suggest,
                                          max_evals=1,
                                          trials=Trials())

    # Print results
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    # Save results
    save_hyperas_results(best_model, best_run, result_folder, load_data, model)
