import numpy as np
from resnetBioHashModel import BioHashFace
from keras.preprocessing import image
from keras_vggface import utils
import keras
import unittest
from keras.models import Model
from keras import models
from keras import layers
from keras import optimizers
from keras.layers import *
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import ipfsapi

keras.backend.set_image_dim_ordering('tf')

hidden_dim=4096
nb_class=2

apiIPFS = ipfsapi.connect('127.0.0.1', 5001)  #LOCAL IPFS SERVER, PREVOUSLY INSTALLED



def createIPFSHash(ImageDir="images/train/beto"):  #CREATE A cryptographic HASHID FROM IPFS SERVER

    dictionaryIPFS = apiIPFS.add(ImageDir, recursive=True)

    return dictionaryIPFS["Hash"]





def train_model():
    model = BioHashFace(include_top=False, input_shape=(224, 224, 3))

    #for layer in model.layers[:-7]:
    #    layer.trainable = False

    last_layer = model.get_layer('avg_pool').output

    fully = Flatten(name='flatten')(last_layer)
    fully = Dense(hidden_dim, activation='relu', name='fc6')(fully)
    fully = Dense(hidden_dim, activation='relu', name='fc7')(fully)
    out = Dense(nb_class, activation='softmax', name='fc8')(fully)
    new_bio_model = Model(model.input, out)

    new_bio_model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])

    train_data_path = 'image/train'
    validation_data_path = 'image/validation'

    img_width, img_height = 224, 224

    train_datagen = ImageDataGenerator(
          rescale=1./255,
          rotation_range=10,
          width_shift_range=0.2,
          height_shift_range=0.2,
          horizontal_flip=True,
          fill_mode='nearest')

    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_batchsize = 16
    val_batchsize = 16

    train_generator = train_datagen.flow_from_directory(
            train_data_path,
            save_to_dir="image/data",
            target_size=(img_width, img_height),
            batch_size=train_batchsize,
            class_mode='categorical')

    validation_generator = validation_datagen.flow_from_directory(
            validation_data_path,
            target_size=(img_width, img_height),
            batch_size=val_batchsize,
            class_mode='categorical',
            shuffle=True)

    dict_class = train_generator.class_indices

    print(new_bio_model.summary())

    #new_bio_model.load_weights('latest_v1.h5')

    history = new_bio_model.fit_generator(
      train_generator,
      steps_per_epoch=train_generator.samples/train_generator.batch_size ,
      epochs=50,
      validation_data=validation_generator,
      validation_steps=validation_generator.samples/validation_generator.batch_size,
      verbose=1)

    new_bio_model.save('latest_v1.h5')

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


def predict_model():

    model = BioHashFace(include_top=False, input_shape=(224, 224, 3))
    last_layer = model.get_layer('avg_pool').output
    fully = Flatten(name='flatten')(last_layer)
    fully = Dense(hidden_dim, activation='relu', name='fc6')(fully)
    fully = Dense(hidden_dim, activation='relu', name='fc7')(fully)
    out = Dense(nb_class, activation='softmax', name='fc8')(fully)
    new_bio_model = Model(model.input, out)

    new_bio_model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])

    new_bio_model.load_weights('latest_v1.h5')


    print(new_bio_model.summary())


    img = image.load_img('image/beto1.jpg', target_size=(224, 224))
    img2 = image.load_img('image/beto2.jpg', target_size=(224, 224))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = utils.preprocess_input(x, version=2)
    x2 = image.img_to_array(img2)
    x2 = np.expand_dims(x2, axis=0)
    x2 = utils.preprocess_input(x2, version=2)

    preds = new_bio_model.predict(x)
    preds2 = new_bio_model.predict(x2)

    print(np.argmax(preds, axis=1)[0])


