import numpy as np
from resnetBioHashModel import BioHashFaceResnetClassifier, preprocess_input, BioHashFace
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
import hashlib

keras.backend.set_image_dim_ordering('tf')
hidden_dim=4096
nb_class=2
plot_training=False

IPFS_Host = "127.0.0.1"
IPFS_Port = 5001

apiIpfs = None
isConnectedIPFS = False

def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

def connect2IPFS(host, port):

    if host and port:
        IPFS_Host = host
        IPFS_Port = port

    if not ipfsapi:
        print("Error######: IPFS Client is Not instaled ")
        exit(-1)

    global isConnectedIPFS
    global apiIpfs

    if not isConnectedIPFS:
        try:
            apiIpfs = ipfsapi.connect(IPFS_Host, IPFS_Port)  #LOCAL IPFS SERVER, PREVOUSLY INSTALLED

            print("Successful connected to IPFS nectwork")
            print(apiIpfs)
            isConnectedIPFS = True
        except Exception as e:
            print("Unexpected Error, on IPFS::")
            print(e)
            exit(-1)
    #else:
        #print("Already connected")


def createIPFSHash(ImagePath=""):  #CREATE A cryptographic HASHID FROM IPFS SERVER
    #connect2IPFS()
    if not apiIpfs:
        connect2IPFS(IPFS_Host, IPFS_Port )

    if not apiIpfs:
        print("Error: not connected to IPFS Server")
        return None

    try:
        dictionaryIPFS = apiIpfs.add(ImagePath)
        print(dictionaryIPFS)
        if dictionaryIPFS["Hash"]:
            return dictionaryIPFS["Hash"]
        else:
            return None

    except Exception as e:
        print("Unexpected Error::")
        print(e)
        return None


def getIPFSHashFile(hashid=None):
    if not apiIpfs:
        connect2IPFS(IPFS_Host, IPFS_Port )

    if not apiIpfs:
        print("Error: not connected to IPFS Server")
        return None

    try:
        content = apiIpfs.get(hashid)
        return content

    except Exception as e:
        print("Unexpected Error::")
        print(e)
        return None

def retriveIPFSHashContent(hashid=None):
    if not apiIpfs:
        connect2IPFS(IPFS_Host, IPFS_Port )

    if not apiIpfs:
        print("Error: not connected to IPFS Server")
        return None

    try:
        content = apiIpfs.cat(hashid)
        #print(content)
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
    x = Flatten(name='flatten')(last_layer)
    out = Dense(nb_class, activation='softmax', name='classifier')(x)
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

def train_fingerprint_model():
    model = BioHashFace(include_top=False, input_shape=(224, 224, 3))

    #for layer in model.layers[:-7]:
    #    layer.trainable = False
    last_layer = model.get_layer('avg_pool').output
    x = Flatten(name='flatten')(last_layer)
    out = Dense(nb_class, activation='softmax', name='classifier')(x)
    new_bio_model = Model(model.input, out)

    new_bio_model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])

    train_data_path = 'fingerprint/train'
    validation_data_path = 'fingerprint/validation'

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
            save_to_dir="fingerprint/data",
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

    history = new_bio_model.fit_generator(
      train_generator,
      steps_per_epoch=train_generator.samples/train_generator.batch_size ,
      epochs=50,
      validation_data=validation_generator,
      validation_steps=validation_generator.samples/validation_generator.batch_size,
      verbose=1)

    new_bio_model.save('latest_fingerprint_v1.h5')

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

    model = BioHashFace(include_top=False, input_shape=(224, 224, 3))
    last_layer = model.get_layer('avg_pool').output
    x = Flatten(name='flatten')(last_layer)
    out = Dense(nb_class, activation='softmax', name='classifier')(x)
    new_bio_model = Model(model.input, out)

    new_bio_model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])

    new_bio_model.load_weights('latest_v1.h5')
    #print(new_bio_model.summary())

    return new_bio_model


def loadFaceModelFeaturesOnly():

    model = BioHashFace(include_top=False, input_shape=(224, 224, 3))

    new_bio_model = Model(inputs=model.layers[0].input, outputs=model.get_layer('avg_pool').output)
    #print(new_bio_model.summary())

    #last_layer = model.get_layer('avg_pool').output
    #x = Flatten(name='flatten')(last_layer)
    #out = Dense(nb_class, activation='softmax', name='classifier')(x)
    #new_bio_model = Model(model.input, out)
    #new_bio_model.compile(loss='categorical_crossentropy',
    #              optimizer=optimizers.RMSprop(lr=1e-4),
    #              metrics=['acc'])

    #new_bio_model.load_weights('latest_v1.h5')
    #print(new_bio_model.summary())
    #print(model.summary())
    #return new_bio_model
    return new_bio_model



def loadFingerprintSavedModel():

    model = BioHashFace(include_top=False, input_shape=(224, 224, 3))
    last_layer = model.get_layer('avg_pool').output
    x = Flatten(name='flatten')(last_layer)
    out = Dense(nb_class, activation='softmax', name='classifier')(x)
    new_bio_model = Model(model.input, out)

    new_bio_model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])

    new_bio_model.load_weights('latest_fingerprint_v1.h5')
    #print(new_bio_model.summary())

    return new_bio_model


def compare_biohash_multimodal(biohash1, biohash2):
    print("Coming soon...")


def save_vectors_ipfs(picture):
    try:
        idhash = hashlib.md5(picture).hexdigest()
        np.save('image/' + idhash, picture)
        biohash = createIPFSHash('image/' + idhash + ".npy")
        return biohash
    except Exception as e:
        print("Unexpected Error::")
        print(e)
        return None

def generate_biohash(picture):

    if not picture:
        print("Error, you must provide a valid picture")
        return None
    try:
        idhash = hashlib.md5(picture).hexdigest()
        print(idhash)
        with open('image/' + idhash, 'wb') as file:
            file.write(picture)
        biohash = createIPFSHash('image/' + idhash)
        return biohash
    except Exception as e:
        print("Unexpected Error::")
        print(e)
        return None



def compareVectorFeatures(vector1, vector2):
    epsilon = 0.41 #cosine similarity
    #epsilon = 120  #euclidean distance
    cosine_similarity = findCosineSimilarity(vector1, vector2)
    print("cosine_similarity: ", cosine_similarity)
    if(cosine_similarity < epsilon):
        print("Succesful Match ")
        return 0
    else:
        print("Did not Match ")
        return 1

def compare_biohashEuclidianDistance(biohash1, biohash2, isFingerprint):

    if not biohash1 and not biohash2:
        print("Error, you must provide a valid biohash1 & biohash2")
        return -1

    model = loadFaceModelFeaturesOnly()
    if isFingerprint:
        model = loadFingerprintSavedModel()

    if not model:
        print("Error, loading the model")
        return -1

    biometric1 = retriveIPFSHashContent(biohash1)
    biometric2 = retriveIPFSHashContent(biohash2)

    if not biometric1 and not biometric2:
        print("Error, Invalid IPFS HASH")
        return -1

    with open('image/' + biohash1, 'wb') as file:
        file.write(biometric1)
    with open('image/' + biohash2, 'wb') as file:
        file.write(biometric2)

    img1 = image.load_img('image/' + biohash1, target_size=(224, 224))
    img2 = image.load_img('image/' + biohash2, target_size=(224, 224))

    x = image.img_to_array(img1)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x, version=2)
    x2 = image.img_to_array(img2)
    x2 = np.expand_dims(x2, axis=0)
    x2 = preprocess_input(x2, version=2)

    img1_representation = model.predict(x).ravel()
    img2_representation = model.predict(x2).ravel()

    print(img1_representation.shape)

    epsilon = 0.41 #cosine similarity
    #epsilon = 120  #euclidean distance

    cosine_similarity = findCosineSimilarity(img1_representation.ravel(), img2_representation)
    euclidean_distance = findEuclideanDistance(img1_representation, img2_representation)

    print("cosine_similarity: ", cosine_similarity)
    print("euclidean_distance: ", euclidean_distance)
    if(cosine_similarity < epsilon):
        print("Succesful Match ")
        return 0
    else:
        print("Did not Match ")
        return 1


def compare_biohash(biohash1, biohash2, isFingerprint):

    if not biohash1 and not biohash2:
        print("Error, you must provide a valid biohash1 & biohash2")
        return -1

    model = loadSavedModel()
    if isFingerprint:
        model = loadFingerprintSavedModel()

    if not model:
        print("Error, loading the model")
        return -1

    biometric1 = retriveIPFSHashContent(biohash1)
    biometric2 = retriveIPFSHashContent(biohash2)

    if not biometric1 and not biometric2:
        print("Error, Invalid IPFS HASH")
        return -1

    with open('image/' + biohash1, 'wb') as file:
        file.write(biometric1)
    with open('image/' + biohash2, 'wb') as file:
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

#if __name__ == '__main__':
#    globals()[sys.argv[1]]()
