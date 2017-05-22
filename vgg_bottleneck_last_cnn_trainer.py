
#! usr/bin/env python
import pandas as pd
import numpy as np
import pickle
import os
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import ImageDataGenerator, array_to_img
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense

# deal with truncated images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def create_generators():
    """Create some image genreators with image augmentation"""
    train_datagen = ImageDataGenerator(
                                       rotation_range=80,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')
    train_generator = train_datagen.flow_from_directory(directory='train/',
                                                        target_size=[224, 224],
                                                        batch_size=16,
                                                       classes=['type_1','type_2','type_3'])


    validation_datagen = ImageDataGenerator(rotation_range=10,
                                    width_shift_range=0.05,
                                    height_shift_range=0.05,
                                    shear_range=0.05,
                                    zoom_range=0.05,
                                    horizontal_flip=True,
                                    fill_mode='nearest')
    validation_generator = validation_datagen.flow_from_directory(directory='validation/',
                                                                  target_size=[224, 224],
                                                                  batch_size=16,
                                                                 classes=['type_1','type_2','type_3'])
    return train_generator, validation_generator

def build_model(layer_to_finetune=8):
    """Build a VGG16 model to finetune"""
    vgg16 = VGG16(weights='imagenet')

    # repalce last layer with 3 for 3 classes
    fc2 = vgg16.get_layer('fc2').output
    p = Dense(output_dim=3, activation='softmax', name='logit')(fc2)
    model_bottleneck_cnn = Model(input=vgg16.input, output=p)
    model_bottleneck_cnn.compile(
        optimizer='adam',metrics=['accuracy'],loss='categorical_crossentropy')

    # train bottleneck features plus last convolution layers
    for k,v in model_bottleneck_cnn.layers_by_depth.items():
        print(k,v[0].trainable,v)
        if k < 8: # train these layers
            pass
        else:
            v[0].trainable = False
    return model_bottleneck_cnn


def trainer(model_bottleneck_cnn,num_epochs,train_gen, val_gen,
            save_weights = 'vgg16_vgg_model_bottleneck_first_cnn100.h5',
            load_weights = None):
    """Train the model"""

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
            model_yaml_bottleneck = model_bottleneck_cnn.to_yaml()
            with open("weights/vgg16_model_bottleneck_first_cnn.yaml", "w") as yaml_file:
                yaml_file.write(model_yaml_bottleneck)
            # serialize weights to HDF5
            model_bottleneck_cnn.save_weights("weights/{}".format(save_weights))

        else:
            l = model_bottleneck_cnn.fit_generator(train_gen,
                                steps_per_epoch=16,
                                epochs=1,
                                validation_data=val_gen,
                                validation_steps=16);
            loss.append(l)
    return loss


if __name__ == '__main__':
    # PARAMETERS
    num_epochs = 500
    # Save weights
    save_weights = 'vgg16_vgg_model_bottleneck_first_cnn100.h5')
    # Load weights
    load_weights = None
    #create generators
    train_gen, val_gen = create_generators()
    #Create he model
    model = build_model()

    # train the model
    loss = trainer(model, num_epochs,train_gen, val_gen, save_weights, load_weights)
    #save the loss
    with open('loss/loss_vgg_100epochs','wb') as fp:
        pickle.dump(loss,fp)
