import numpy as np
from resnetBioHashModel import BioHashFaceResnetClassifier, preprocess_input
from keras.preprocessing import image
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
import sys

keras.backend.set_image_dim_ordering('tf')

hidden_dim=4096
nb_class=2
plot_training=False
IPFS_Host = "127.0.0.1"
IPFS_Port = 5001

apiIPFS=None


def connect2IPFS():

    if not ipfsapi:
        print("Error######: IPFS Client is Not instaled ")
        exit(-1)

    try:
        apiIPFS = ipfsapi.connect(IPFS_Host, IPFS_Port)  #LOCAL IPFS SERVER, PREVOUSLY INSTALLED

        print("Successful connected to IPFS nectwork")
        print(apiIPFS)

    except Exception as e:
        print("Unexpected Error, on IPFS::")
        print(e)
        exit(-1)


def createIPFSHash(ImageDir="images/train/beto"):  #CREATE A cryptographic HASHID FROM IPFS SERVER

    if not apiIPFS:
        print("Error: not connected to IPFS Server")
        return None

    try:
        dictionaryIPFS = apiIPFS.add(ImageDir, recursive=True)
        print(dictionaryIPFS)
        if dictionaryIPFS["Hash"]:
            return dictionaryIPFS["Hash"]
        else:
            return None

    except Exception as e:
        print("Unexpected Error::")
        print(e)
        return None


def retriveIPFSHashContent(hashid=None):

    if not apiIPFS:
        print("Error: not connected to IPFS Server")
        return None

    try:
        content = apiIPFS.cat(hashid)
        print(content)
        return content

    except Exception as e:
        print("Unexpected Error::")
        print(e)
        return None


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

    if plot_training:
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


def loadSavedModel():

    model = BioHashFaceResnetClassifier(hidden_dim, nb_class)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])

    model.load_weights('latest_v1.h5')
    print(model.summary())

    return model

def compare_biohash(biohash1, biohash2):

    if not biohash1 and not biohash2:
        print("Error, you must provide a valid biohash1 & biohash2")
        return -1

    model = loadSavedModel()
    if not model:
        print("Error, loading the model")
        return -1

    biometric1 = retriveIPFSHashContent(biohash1)
    biometric2 = retriveIPFSHashContent(biohash2)

    if not biometric1 and not biometric2:
        print("Error, Invalid IPFS HASH")
        return -1

    with open('image/' + biohash1, 'w') as file:
        file.write(biometric1)
    with open('image/' + biohash2, 'w') as file:
        file.write(biometric2)


    img1 = image.load_img('image/' + biohash1, target_size=(224, 224))
    img2 = image.load_img('image/' + biohash2, target_size=(224, 224))


    x = image.img_to_array(img1)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x, version=2)
    x2 = image.img_to_array(img2)
    x2 = np.expand_dims(x2, axis=0)
    x2 = preprocess_input(x2, version=2)

    prediction1 = model.predict(x)
    prediction2 = model.predict(x2)
    print(np.argmax(prediction1, axis=1)[0],np.argmax(prediction2, axis=1)[0] )

    if np.argmax(prediction1, axis=1)[0] == np.argmax(prediction2, axis=1)[0]:
        print("Succesful Match ")
        return 0
    else:
        print("Did not Match ")
        return 1



connect2IPFS()

if __name__ == '__main__':
    globals()[sys.argv[1]]()
