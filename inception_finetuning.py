#! usr/bin/env python
import pandas as pd
import numpy as np
import pickle
import os
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import ImageDataGenerator, array_to_img
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.inception_v3 import InceptionV3
inception_model = InceptionV3(weights='imagenet')
# change the last for three classes
x = inception_model.get_layer('avg_pool').output
predictions = Dense(3, activation='softmax')(x)
model_inception_finetune = Model(inputs=inception_model.input, outputs=predictions)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 172 layers and unfreeze the rest:
def build_model():
    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(3, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 172 layers and unfreeze the rest:
    for layer in model.layers[:172]:
        layer.trainable = False
    for layer in model.layers[172:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


#set up generators
def build_generators():
    """Image generators with data augmentation"""
    train_datagen_inception = ImageDataGenerator(
                                       rotation_range=80,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')
    train_generator_inception = train_datagen_inception.flow_from_directory(directory='train/',
                                                        target_size=[229, 229],
                                                        batch_size=16,
                                                       classes=['type_1','type_2','type_3'])
    validation_datagen_inception = ImageDataGenerator(rotation_range=40,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')
    validation_generator_inception = validation_datagen_inception.flow_from_directory(directory='validation/',
                                          target_size=[229, 229],
                                      batch_size=16,
                                     classes=['type_1','type_2','type_3'])
    return train_generator_inception, validation_generator_inception

def trainer(model, num_epochs,train_gen, val_gen,
            load_weights=None,
            save_weights = 'inception_v3_bottleneck_last_two.h5'):
    """Train the inception model and load weights if provided"""
    if save_weights != None:
        model_bottleneck_cnn.load_weights("weights/{}".format(load_weights))
    else:
        pass
    loss = []
    # train the model
    for i in range(num_epochs):
        print(i,'iteration number')
        if i % 10 == 0: # save the weights
            # serialize model to YAML
            model_yaml_bottleneck = model.to_yaml()
            with open("weights/inception_model.yaml", "w") as yaml_file:
                yaml_file.write(model_yaml_bottleneck)
            # serialize weights to HDF5
            model.save_weights("weights/{}".format(save_weights))

        else:
            # train the model on the new data for a few epochs
            l = model.fit_generator(train_gen,
                        steps_per_epoch=8,
                        epochs=1,
                        validation_data=val_gen,
                        validation_steps=8)
            loss.append(l)



if __name__ =='__main__':

    # PARAMETERS
    num_epochs = 1000
    # Save weights
    save_weights = 'inception_v3_bottleneck_last_two.h5'
    # Load weights
    load_weights = None
    #create generators
    train_gen, val_gen = build_generators()
    #Create he model
    model = build_model()

    # train the model
    loss = trainer(model, num_epochs,train_gen, val_gen, save_weights, load_weights)
    #save the loss
    with open('loss/loss_inception_1000epochs','wb') as fp:
        pickle.dump(loss,fp)
