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
from keras.applications import VGG19, ResNet152V2, ResNet50V2, VGG16, Xception
from keras.layers import concatenate, Conv2D, Dropout, Flatten, Dense, BatchNormalization, AveragePooling2D, Reshape
from keras.optimizers import SGD, Adam

from Class_training import simple_training
from utils_class import shuffle_unison_arrays

from attention_module import cbam_block
from keras.layers.experimental.preprocessing import RandomTranslation, RandomFlip, RandomRotation, RandomZoom, RandomHeight, RandomWidth,Rescaling 

from keras.applications.imagenet_utils import preprocess_input
# ----------------------------------------------------------------------------------------------------------------------
# Functions definitions
# ----------------------------------------------------------------------------------------------------------------------

def preprocessing_layers(a):
    #a = preprocess_input(a)#Rescaling(1/255)(a)
    a = RandomTranslation( 0.2, 0.2, fill_mode="reflect",interpolation="bilinear",)(a)
    a = RandomFlip()(a)
    a = RandomZoom(0.25)(a)
    a = RandomRotation(2)(a)
    #a = RandomHeight(0.2)(a)
    #a = RandomWidth(0.2)(a)
    
    return a
    
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
    
    #vgg_feature_extractor = VGG19(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
    
    #vgg_include_until='block4_pool'
    #feature_extractor = VGG19(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
    #feature_extractor = Model(inputs=vgg_feature_extractor.input, outputs=feature_extractor.get_layer(vgg_include_until).output)
    
    feature_extractor = VGG19(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
    
    
    #feature_extractor = ResNet152V2(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
    # Fine tuning by freezing the last 4 convolutional layers of VGG19 (last block)
    #for layer in feature_extractor.layers[:-8]:
    #    layer.trainable = False
    #feature_extractor.trainable = False

    # Definition of the 2 inputs
    img_a = Input(shape=(224*224*3), name="data_left")
    out_a = Reshape((224,224,3), input_shape=(224*224*3,))(img_a)
    out_a = preprocessing_layers(out_a)
    
    
    img_b = Input(shape=(224*224*3), name="data_right")
    out_b = Reshape((224,224,3), input_shape=(224*224*3,))(img_b)
    out_b = preprocessing_layers(out_b)
    
    
    out_a = feature_extractor(out_a)
    out_b = feature_extractor(out_b)

    #out_a = AveragePooling2D(pool_size=(7, 7))(out_a)
    #out_b = AveragePooling2D(pool_size=(7, 7))(out_b)
    # Concatenation of the inputs
    
    concat = concatenate([out_a, out_b])
    x = concat
    
   # x = cbam_block(x)
    # Add convolution layers on top
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name="Conv_1")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.75, name="Drop_1")(x)
    
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name="Conv_2")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.75, name="Drop_2")(x)
    x = Flatten()(x)
    #Dropout(0.75, name="Drop_1")(x)
    #x = Dense(4096, activation='relu', kernel_initializer='glorot_normal', name="Dense1")(x)
    
    #x = AveragePooling2D(pool_size=(7, 7))
    #x = Flatten()(x)
    
    #x = Dense(512, activation='relu', kernel_initializer='glorot_normal', name="Dense1")(x)
    #Dropout(0.5, name="Drop_1")(x)
    
    x = Dense(256, activation='relu', kernel_initializer='glorot_normal', name="Dense2")(x)
    Dropout(0.5, name="Drop_2")(x)
    
    #Dropout(0.75, name="Drop_3")(x)
    #x = Dense(1000, activation='relu', kernel_initializer='glorot_normal', name="Dense1")(x)
    
    x = Dense(2, activation='softmax', name="Final_dense")(x)
    classification_model = Model([img_a, img_b], x)

    if weights:
        classification_model.load_weights(weights)

    #sgd = SGD(lr=1e-5, decay=1e-6, momentum=0.695, nesterov=True)
    
    classification_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-5), metrics=['accuracy'])
    return classification_model

def ranking_model(img_size, weights=None):
    """
    Create comparisons network which reproduce the choice in an images duel.

    :param img_size: size of input images during training
    :type img_size: tuple(int)
    :param weights: path to the weights use for initialization
    :type weights: str
    :return: ranking comparisons model
    :rtype: keras.Model
    """
    
    #vgg_feature_extractor = VGG19(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
    
    #vgg_include_until='block4_pool'
    #feature_extractor = VGG19(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
    #feature_extractor = Model(inputs=vgg_feature_extractor.input, outputs=feature_extractor.get_layer(vgg_include_until).output)
    
    feature_extractor = VGG19(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
    
    
    #feature_extractor = ResNet152V2(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
    # Fine tuning by freezing the last 4 convolutional layers of VGG19 (last block)
    #for layer in feature_extractor.layers[:-8]:
    #    layer.trainable = False
    #feature_extractor.trainable = False

    # Definition of the 2 inputs
    img = Input(shape=(224*224*3), name="data")
    x = Reshape((224,224,3), input_shape=(224*224*3,))(img)
    x = preprocessing_layers(x)
    
    
    x = feature_extractor(x)
   # x = cbam_block(x)
    # Add convolution layers on top
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name="Conv_1")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.75, name="Drop_1")(x)
    
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name="Conv_2")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.75, name="Drop_2")(x)
    x = Flatten()(x)
    #Dropout(0.75, name="Drop_1")(x)
    #x = Dense(4096, activation='relu', kernel_initializer='glorot_normal', name="Dense1")(x)
    
    #x = AveragePooling2D(pool_size=(7, 7))
    #x = Flatten()(x)
    
    #x = Dense(512, activation='relu', kernel_initializer='glorot_normal', name="Dense1")(x)
    #Dropout(0.5, name="Drop_1")(x)
    
    x = Dense(256, activation='relu', kernel_initializer='glorot_normal', name="Dense2")(x)
    Dropout(0.75, name="Drop_2")(x)
    
    #Dropout(0.75, name="Drop_3")(x)
    #x = Dense(1000, activation='relu', kernel_initializer='glorot_normal', name="Dense1")(x)
    
    x = Dense(1, activation=None, name="Final_dense")(x)
    classification_model = Model(img, x)

    if weights:
        classification_model.load_weights(weights)

    #sgd = SGD(lr=1e-5, decay=1e-6, momentum=0.695, nesterov=True)
    
    classification_model.compile(loss='mse', optimizer=Adam(learning_rate=1e-5), metrics=['mae'])
    return classification_model
