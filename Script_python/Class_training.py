# -*- coding: utf-8 -*-
"""
File containing the definition of functions used to trained model and store results.

"""
# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------
import os
import numpy as np
import inspect
import json
import csv
import random as rd
import matplotlib.pyplot as plt
import glob

import trueskill as tr
from progressbar import ProgressBar
from keras.models import load_model, model_from_json
from keras import backend as bk
#from keras.optimizers import SGD
from tensorflow.keras.applications import VGG19
from tensorflow.keras.optimizers import SGD

from utils_class import timeit, safe_folder_creation
from Class_Image import Image

# ----------------------------------------------------------------------------------------------------------------------
# Functions definition
# ----------------------------------------------------------------------------------------------------------------------

# ---------------- Training functions ----------------------------------------------------------------------------------

def batched_forward_pass(data, model, batch_size=20):
    len_data = data.shape[0]
    a = vgg(data_left[0:batch_size])
    
    for i in range(batch_size, len_data, batch_size):
        v = vgg(data_left[i:i+batch_size])
        a = np.concatenate((a, v), 0)
        
    return a

@timeit
def simple_training(data_left, data_right, data_label,
                    model_function, model_function_args, folder_path, val_split, epochs, batch_size):
    """
    Train a siamese model and store results in a folder.

    :param data_left: left images array
    :type data_left: np.array
    :param data_right: right images array
    :type data_right: np.array
    :param data_label: labels array
    :type data_label: np.array
    :param model_function: function which build the model to train with k fold
    :type model_function: function
    :param model_function_args: arguments of the model building function
    :type model_function_args: list
    :param folder_path: path of the folder where results are stored
    :type folder_path: str
    :param val_split: set proportion dedicated to the validation set
    :type val_split: float
    :param epochs: number of training epochs
    :type epochs: int
    :param batch_size: batch size
    :type batch_size: int
    :return:
    :rtype:
    """
    # Create folder to store results
    folder_path = safe_folder_creation(folder_path)

    # Build model
    conv_model = model_function(*model_function_args)
    #train_data = [data_left, data_right]
    
    vgg_trainable = 'block3_pool'
    vgg_include_until = 'block4_pool'
    
    vgg = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    vgg_no_train = Model(inputs=vgg.input, outputs=vgg.get_layer(vgg_trainable).output)
    
    try:
        embeddings_left = np.load(folder_path + "/embeddings_left_{}.npy".format(vgg_trainable))
        embeddings_right = np.load(folder_path + "/embeddings_right_{}.npy".format(vgg_trainable))
        print("Embeddings successfully loaded from files")
        
    except FileNotFoundError:
        print("Creating and saving embeddings...")
        embeddings_left = batched_forward_pass(data_left, vgg_no_train, batch_size)
        embeddings_right = batched_forward_pass(data_right, vgg_no_train, batch_size)
        
        np.save(os.path.join(folder_path, "/embeddings_left_{}.npy".format(vgg_trainable)), embeddings_left)
        np.save(os.path.join(folder_path, "/embeddings_right_{}.npy".format(vgg_trainable)), embeddings_right)
        
    return
    train_data[embeddings_left, embeddings_right]
    
    # Train model
    model_fitted = conv_model.fit(train_data, data_label, vgg_trainable=vgg_trainable,
                                  epochs=epochs, batch_size=batch_size, validation_split=val_split)

    # Save model and plots
    model_path = os.path.join(folder_path, 'fitted_model.h5')
    conv_model.save(model_path)
    plot_validation_info(model_fitted, folder_path)

    # Save also weights and structure for backup
    weights_path = os.path.join(folder_path, 'weights.h5')
    conv_model.save_weights(weights_path)
    json_path = os.path.join(folder_path, 'structure.json')
    save_structure_in_json(conv_model, json_path)


@timeit
def k_fold(data_left, data_right, data_label, k, model_function, model_function_args, folder_path, epochs,
           batch_size):
    """
    Execute a K-fold cross validation for a model.

    :param data_left: left images array
    :type data_left: np.array
    :param data_right: right images array
    :type data_right: np.array
    :param data_label: labels array
    :type data_label: np.array
    :param k: number of fold
    :type k: int
    :param model_function: function which build the model to train with k fold
    :type model_function: function
    :param model_function_args: arguments of the model building function
    :type model_function_args: list
    :param folder_path: path of the folder where results are stored
    :type folder_path: str
    :param epochs: number of training epochs
    :type epochs: int
    :param batch_size: batch size
    :type batch_size: int
    """

    # Variable initialization
    nb_comp = len(data_label)
    num_val_samples = nb_comp // k
    all_scores = []

    # Create folder to store results
    folder_path = safe_folder_creation(folder_path)

    # Train model for each fold
    for i in range(k):
        print('Processing fold #', i)

        # Select validation set
        val_left = data_left[i * num_val_samples: (i + 1) * num_val_samples]
        val_right = data_right[i * num_val_samples: (i + 1) * num_val_samples]
        val_data = [val_left, val_right]
        val_label = data_label[i * num_val_samples: (i + 1) * num_val_samples]

        # Get the remaining data to create the training set
        partial_train_left = np.concatenate(
            [data_left[:i * num_val_samples], data_left[(i + 1) * num_val_samples:]])
        partial_train_right = np.concatenate(
            [data_right[:i * num_val_samples], data_right[(i + 1) * num_val_samples:]])
        partial_train_data = [partial_train_left, partial_train_right]
        partial_train_label = np.concatenate(
            [data_label[:i * num_val_samples], data_label[(i + 1) * num_val_samples:]])

        # Build model
        bk.clear_session()
        model = model_function(*model_function_args)

        # Train model
        model_fitted = model.fit(partial_train_data, partial_train_label, validation_data=[val_data, val_label],
                                 epochs=epochs, batch_size=batch_size, verbose=1)

        # Evaluate model and store result
        val_mse, val_mae = model.evaluate(val_data, val_label, verbose=1)
        all_scores.append(val_mae)

        # Save model and plots
        model_path = os.path.join(folder_path, 'fitted_model_k{}.h5'.format(i + 1))
        model.save(model_path)
        plot_validation_info_kfold(model_fitted, i + 1, folder_path)

        # Save also weights and structure for backup
        weights_path = os.path.join(folder_path, 'weights_k{}.h5'.format(i + 1))
        model.save_weights(weights_path)
        json_path = os.path.join(folder_path, 'structure_k{}.json'.format(i + 1))
        save_structure_in_json(model, json_path)

    # Save K-fold summary in a text file
    filename = os.path.join(folder_path, "k_fold_report.txt")
    summary = k_fold_summary(k, model_function, model_function_args, nb_comp, num_val_samples, all_scores, filename)
    print(summary)


# ---------------- Training monitoring functions -----------------------------------------------------------------------
def plot_validation_info_kfold(trained_model, i, save_folder):
    """
    Save graphs of validation accuracy and loss during a k-fold training.

    :param trained_model: history of a model training
    :type trained_model: keras.history
    :param i: fold number
    :type i: int
    :param save_folder: folder path  where plots are saved
    :type save_folder: str
    """

    # Getting statistics values
    acc = trained_model.history['accuracy']
    val_acc = trained_model.history['val_accuracy']
    loss = trained_model.history['loss']
    val_loss = trained_model.history['val_loss']
    epochs = range(1, len(acc) + 1)

    # Validation accuracy plot
    plt.figure()
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_folder, "val_acc_k{}.png".format(i)))

    # Validation loss plot
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(os.path.join(save_folder, "val_loss_k{}.png".format(i)))


def plot_validation_info(trained_model, save_folder):
    """
    Save graphs of validation accuracy and loss during a simple training.

    :param trained_model: history of a model training
    :type trained_model: keras.history
    :param save_folder: folder path  where plots are saved
    :type save_folder: str
    """

    # Getting statistics values
    acc = trained_model.history['accuracy']
    val_acc = trained_model.history['val_accuracy']
    loss = trained_model.history['loss']
    val_loss = trained_model.history['val_loss']
    epochs = range(1, len(acc) + 1)

    # Validation accuracy
    plt.figure()
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_folder, "val_acc.png"))

    # Validation loss
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(os.path.join(save_folder, "val_loss.png"))


def k_fold_summary(k, model_function, model_function_args, nb_comp, nb_val, metrics, filename):
    """
    Create and save summary string of the k-fold training.

    :param k: number of fold
    :type k: int
    :param model_function: function which build the model to train with k fold
    :type model_function: function
    :param model_function_args: arguments of the model building function
    :type model_function_args: list
    :param nb_comp: number of comparison used for training
    :type nb_comp: int
    :param nb_val: size of a validation set
    :type nb_val: int
    :param metrics: list of accuracy results of each fold
    :type metrics: list
    :param filename: path of the summary file
    :type filename: str
    :return: summary string
    :rtype: str
    """

    # Stringify results
    best_acc = np.max(metrics)
    best_model = np.argmax(metrics) + 1
    s = "K-fold summary :\n================"
    s += "\nBest accuracy : {}".format(best_acc)
    s += "\nAchieved during fold : {}\n".format(best_model)
    for i in range(len(metrics)):
        s += "\nAccuracy for fold {} : {}".format(i + 1, metrics[i])

    # Stringify K-fold parameters
    s += "\n\nParameters\n=========="
    s += "\nNumber of folds k : {}".format(k)
    s += "\nTotal number of comparisons in data: {}".format(nb_comp)
    s += "\nNumber of comparisons in a fold: {}".format(nb_val)

    # Stringify building model function
    s += "\n\nModel building function\n======================="
    s += "\nFunction parameters : {}".format(str(model_function_args))
    s += "\nFunction source code:\n" + inspect.getsource(model_function)

    # Save in a file
    with open(filename, "w+") as f:
        f.write(s)
    return s


def save_structure_in_json(model, json_path):
    """
    Save the model structure in a json file.

    :param model: keras model
    :type model: kerqs.Model
    :param json_path: path of the jsn file
    :type json_path: str
    """

    json_string = model.to_json()
    with open(json_path, "w+") as f:
        f.write(json_string)


def save_hyperas_results(best_model, best_run, result_folder, data, model):
    """
    Create folder and file containing results of the Hyperas hyperparameters optimization.

    :param best_model: best model found
    :type best_model: keras.Model
    :param best_run: dictionary of parameters of the best model
    :type best_run: dict
    :param result_folder:
    :type result_folder:
    :param data: loading data function
    :type data: function
    :param model: creation model function
    :type model: function
    """

    # Create result folder
    result_folder = safe_folder_creation(result_folder)

    # Save best model
    best_model.save(os.path.join(result_folder, "best_model.h5"))

    # Save also weights and structure for backup
    weights_path = os.path.join(result_folder, 'weights.h5')
    best_model.save_weights(weights_path)
    json_path = os.path.join(result_folder, 'structure.json')
    save_structure_in_json(best_model, json_path)

    # Save model function an used data in a text file
    s = "Grid search parameters Hyperas \n"
    s += "\nData function code:\n\n" + inspect.getsource(data)
    s += "\nModel source code:\n\n" + inspect.getsource(model)
    s += "\nHyperas results:\n\n" + json.dumps(best_run)

    # Save in a file
    with open(os.path.join(result_folder, 'Hyperas_params.txt'), "w+") as f:
        f.write(s)


def evaluation_test_set(trained_model_path, data_folder, mode="ranking"):
    """
    Evaluate a trained model on a test set.

    :param trained_model_path: path of the trained model saved as an .h5 file
    :type trained_model_path: str
    :param data_folder: main data folder path
    :type data_folder: str
    :param mode: model type. Value can be either "ranking" or "comparisons"
    :type mode: str
    :return: test loss and accuracy
    :rtype: tuple(float)
    """

    # Load data set
    print("Loading left images ...")
    test_left = np.load(os.path.join(data_folder, "test", "test_left_224.npy"))
    print("Done\nLoading right images...")
    test_right = np.load(os.path.join(data_folder, "test", "test_right_224.npy"))
    test_data = [test_left, test_right]

    # Load labels depending on the model type
    print("Done\nLoading labels ...")
    if mode == "ranking":
        test_label = np.load(os.path.join(data_folder, "test", "test_labels_score_224.npy"))
    elif mode == "comparisons":
        test_label = np.load(os.path.join(data_folder, "test", "test_labels_224.npy"))
    else:
        raise ValueError

    print("Done\nLoading model ...")
    # Load model
    best_model = load_model(trained_model_path)
    print("Done\nEvaluation ...")

    # Model evaluation on test set
    loss_acc = best_model.evaluate(test_data, test_label)

    # Save results to text file
    result_path = os.path.join(os.path.split(trained_model_path)[0], 'Evaluation_results.txt')
    with open(result_path, 'w+') as f:
        s = "Model evaluation on test set results\n==================================\n"
        s += '\nModel path : {}'.format(trained_model_path)
        s += '\nTest set folder : {}'.format(data_folder)
        s += '\nTest loss: {}'.format(loss_acc[0])
        s += '\nTest accuracy : {}'.format(loss_acc[1])
        f.write(s)

    return loss_acc


# ---------------- Get models functions --------------------------------------------------------------------------------
def get_ranking_from_meta(meta_path):
    """
    Get ranking model part from the meta network and save it in the same folder.

    :param meta_path: path of the trained meta model.
    :type meta_path:str
    """
    print("Loading meta model ...")
    meta_model = load_model(meta_path)

    print("Done\nSaving ranking model")
    ranking_path = os.path.join(os.path.split(meta_path)[0], "fitted_ranking_model.h5")
    ranking_model = meta_model.layers[2]
    ranking_model.save(ranking_path)


def compiled_model_from_json(json_path, lr, decay, momentum, weights=None):
    """
    Create a compiled model ready for training from a json structure.

    :param json_path: path of the json structure file
    :type json_path: str
    :param lr: learning rate value for the SGD optimizer
    :type lr: float
    :param decay: decay value for the SGD optimizer
    :type decay: float
    :param momentum: momentum value for the SGD optimizer
    :type momentum: float
    :param weights: path to the weights use for initialization
    :type weights: str
    :return: compiled model
    :rtype: keras.model
    """
    # Load model structure
    with open(json_path, 'r') as f:
        jj = f.read()
    m = model_from_json(jj)

    # Lod weights if need be
    if weights:
        m.load_weights(weights)

    sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    m.compile(optimizer=sgd, loss="binary_crossentropy", metrics=['accuracy'])
    return m


# ---------------- Producing output layers functions -------------------------------------------------------------------
@timeit
def trueskill_layer(model_path, images_dir, csv_path):
    """
    Compute a trueskill rating for each image of the directory from 100 comparisons predictions.
    Save the results in a csv file.

    :param model_path: path of the trained comparisons model
    :type model_path: str
    :param images_dir: images directory path
    :type images_dir: str
    :param csv_path: path of csv where results are stored
    :type csv_path: str
    """

    # Variable initialization
    print("Loading model ...")
    model = load_model(model_path)
    images_path = glob.glob(images_dir + "/*jpg")
    pictures = []
    images = []

    # Build list of Image objects
    print("Done\nLoading images...")
    pbar = ProgressBar()
    for img_path in pbar(images_path):
        img = Image(img_path, build_array=True)
        img.get_info_from_filename()
        images.append(img)
        pictures.append(img.preprocess_image(224))
    pictures = np.array(pictures)

    pbar2 = ProgressBar()
    print("Done\nComputing trueskill rating for each image ...")
    for i in pbar2(range(len(images))):
        # Create 100 contenders and predict a comparison for each
        contenders = [rd.randint(0, len(images) - 1) for _ in range(100)]
        predictions = model.predict(
            [np.array([pictures[i] for _ in range(100)]), np.array([pictures[contenders[k]] for k in range(100)])])

        # Update image trueskill rating for each comparison
        for j in range(predictions.shape[0]):
            if predictions[j][0] > predictions[j][1]:
                images[i].trueskill_rating, images[contenders[j]].trueskill_rating = tr.rate_1vs1(
                    images[i].trueskill_rating, images[contenders[j]].trueskill_rating)
            else:
                images[contenders[j]].trueskill_rating, images[i].trueskill_rating = tr.rate_1vs1(
                    images[contenders[j]].trueskill_rating, images[i].trueskill_rating)

    # Save results in csv
    print("Done\nSaving results in csv ...")
    trueskill_csv(images, csv_path)
    print("Done")


def trueskill_csv(images, csv_path):
    """
    Save image ratings in csv file.

    :param images: list of Image instances.
    :type images: list
    :param csv_path: path of csv where results are stored
    :type csv_path: str
    """

    # Create csv
    with open(csv_path, 'w+') as layer:
        csv_writer = csv.writer(layer, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')

        # Write header
        csv_writer.writerow(["Filename", "Global_key", "Latitude", "Longitude", "Trueskill_mu", "Trueskill_sigma"])

        # Write a line per image
        for img in images:
            filename = img.basename
            key = img.global_key
            lat = img.lat
            lon = img.lon
            mu = img.trueskill_rating.mu
            sigma = img.trueskill_rating.sigma
            csv_writer.writerow([filename, key, lat, lon, mu, sigma])


@timeit
def ranking_layer(model_path, images_dir, csv_path):
    """
    Predict the score of all the images of the directory with the ranking model. Save the results in a csv file.

    :param model_path: path of the trained ranking model
    :type model_path: str
    :param images_dir: images directory path
    :type images_dir: str
    :param csv_path: path of csv where results are stored
    :type csv_path: str
    """

    # Variable initialization
    images_path = glob.glob(images_dir + "/*jpg")
    images = []

    # Load scoring model
    print("Loading model ...")
    model = load_model(model_path)
    
    # Build list of Image objects
    print("Done\nMake score prediction for each image ...")
    pbar = ProgressBar()
    for img_path in pbar(images_path):
        img = Image(img_path, build_array=True)
        img.get_info_from_filename()
        images.append(img)
        img.ranking_score = model.predict(np.array([img.preprocess_image(224)]))[0][0]

    # Save results in csv
    print("Done\nSaving results in csv ...")
    ranking_csv(images, csv_path)


def ranking_csv(images, csv_path):
    """
    Save image ratings in csv file.

    :param images: list of Image instances.
    :type images: list
    :param csv_path: path of csv where results are stored
    :type csv_path: str
    """

    # Create csv
    with open(csv_path, 'w+') as layer:
        csv_writer = csv.writer(layer, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')

        # Write header
        csv_writer.writerow(["Filename", "Global_key", "Latitude", "Longitude", "Ranking_score"])

        # Write a line per image
        for img in images:
            filename = img.basename
            key = img.global_key
            lat = img.lat
            lon = img.lon
            r_score = img.ranking_score
            csv_writer.writerow([filename, key, lat, lon, r_score])
