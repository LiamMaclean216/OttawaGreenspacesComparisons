# -*- coding: utf-8 -*-
"""
Script used to make a hyperparameters optimization of the comparisons model.

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
from keras import Input, Model
from keras.applications import VGG19
from keras.layers import concatenate, Conv2D, Dropout, Flatten, Dense, BatchNormalization
from keras.optimizers import SGD

from Class_training import save_hyperas_results


# ----------------------------------------------------------------------------------------------------------------------
# Functions definitions
# ----------------------------------------------------------------------------------------------------------------------
def load_data():
    """
    Load the data used to train the comparisons model.

    :return: training data and labels
    :rtype : tuple(np.array)
    """
    save_folder = r"D:\Guillaume\Ottawa\Data\Comparisons_npy\08_14"
    data_left = np.load(os.path.join(save_folder, "train", "train_left_224.npy"))
    data_right = np.load(os.path.join(save_folder, "train", "train_right_224.npy"))
    data_label = np.load(os.path.join(save_folder, "train", "train_labels_224.npy"))

    data = [data_left, data_right]

    return data, data_label


def model(data, data_label):
    """
    Defines the comparisons model, all hyperparameters in double brackets will be optimize by Hyperas.
    :return: a dictionary with following keys :
                - loss : the metrics function to be minimized by Hyperopt.
                - status : a boolean that tells if everything went fine.
                - model : the model on which hyperparameters optimization occurs.
    """
    img_size = 224
    vgg_feature_extractor = VGG19(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
    for layer in vgg_feature_extractor.layers[:-4]:
        layer.trainable = False

    img_a = Input(shape=(img_size, img_size, 3), name="left_image")
    img_b = Input(shape=(img_size, img_size, 3), name="right_image")

    out_a = vgg_feature_extractor(img_a)
    out_b = vgg_feature_extractor(img_b)

    concat = concatenate([out_a, out_b])

    x = Conv2D({{choice([64, 128, 256, 512])}}, (3, 3), activation='relu', padding='same', name="Conv_1")(concat)
    x = Dropout({{uniform(0, 0.5)}}, name="Drop_1")(x)
    x = Conv2D({{choice([64, 128, 256, 512])}}, (3, 3), activation='relu', padding='same', name="Conv_2")(x)
    x = Dropout({{uniform(0, 0.5)}}, name="Drop_2")(x)
    x = Conv2D({{choice([64, 128, 256, 512])}}, (3, 3), activation='relu', padding='same', name="Conv_3")(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(2, activation='softmax', name="Dense_Final")(x)

    comparisons_model = Model([img_a, img_b], x)

    sgd = SGD(lr={{choice([1e-4, 1e-5, 1e-6])}}, decay={{choice([1e-4, 1e-5, 1e-6])}}, momentum={{uniform(0, 0.9)}},
              nesterov=True)
    comparisons_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    result = comparisons_model.fit(
        [data[0], data[1]],
        data_label,
        batch_size=16,
        epochs=30,
        validation_split=0.2)

    validation_acc = np.amax(result.history['val_acc'])
    print('Best validation acc of epoch:', validation_acc)

    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': comparisons_model}


if __name__ == "__main__":
    # ------------------------------------------------------------------------------------------------------------------
    # Variables initialization
    # ------------------------------------------------------------------------------------------------------------------
    result_folder = r'D:\Guillaume\Ottawa\Data\Training_Models_Results\Comparisons_Trueskill\08_27_hyperas'

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
