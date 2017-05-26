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
from keras.optimizers import SGD
inception_model = InceptionV3(weights='imagenet')
# deal with truncated images
from PIL import ImageFile
from PIL import Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.preprocessing import StandardScaler

def preprocessing(x):
# remove the mean and divide by standard deviation
    standardscaler = StandardScaler()

    new_image = np.zeros((x.shape))

    for i in range(3): # for each channel
        new_image[:,:,i] = np.array([standardscaler.fit_transform(x[:,:,i])])



    return new_image


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
    # and a logistic layer -- let's say we have 3 classes
    predictions = Dense(3, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)




    for layer in model.layers[:172]:
        layer.trainable = True
    for layer in model.layers[172:]:
        layer.trainable = True

    sgd = SGD(lr=.001)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


#set up generators
def build_generators():
    """Image generators with data augmentation"""
    train_datagen_inception = ImageDataGenerator(preprocessing_function=preprocessing,
                                   rotation_range=90,
                                   width_shift_range=0.05,
                                   height_shift_range=0.05,
                                   shear_range=0.05,
                                   zoom_range=0.05,
                                   horizontal_flip=True,
                                    vertical_flip=True,
                                   fill_mode='nearest')
    train_generator_inception = train_datagen_inception.flow_from_directory(directory='train/',
                                                        target_size=[229, 229],
                                                        batch_size=16,
                                                       classes=['type_1','type_2','type_3'])
    validation_datagen_inception = ImageDataGenerator(preprocessing_function=preprocessing,
        rotation_range=5,
                                    width_shift_range=0.05,
                                    height_shift_range=0.05,
                                    shear_range=0.05,
                                    zoom_range=0.05,
                                    horizontal_flip=True,
                                    vertical_flip=True,
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
    if load_weights != None:
        print('loading weights')
        model.load_weights("weights/{}".format(load_weights))
        print('weights loaded')
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
                        steps_per_epoch=16,
                        epochs=1,
                        validation_data=val_gen,
                        validation_steps=16)
            loss.append(l)
    # finished training
    # serialize model to YAML
    model_yaml_bottleneck = model.to_yaml()
    with open("weights/inception_model.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml_bottleneck)
    # serialize weights to HDF5
    model.save_weights("weights/{}".format(save_weights))



if __name__ =='__main__':

    # PARAMETERS
    num_epochs = 2000
    # Save weights
    save_weights = 'inception_adam_all_layers_normalizedPixels.h5'
    # Load weights
    load_weights = None
    #create generators
    train_gen, val_gen = build_generators()
    #Create he model
    model = build_model()

    # train the model
    loss = trainer(model, num_epochs,train_gen, val_gen,load_weights, save_weights)
    #save the loss
    with open('loss/loss_inception_adam_1000epochs','wb') as fp:
        pickle.dump(loss,fp)
