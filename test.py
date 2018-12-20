import keras
from keras.preprocessing import image

from BioHash import connect2IPFS, train_model, generate_biohash, compare_biohash, train_fingerprint_model

print("train_model")
#train_model()


#Uncomment to Train the Model

#train_fingerprint_model()


print("end")


connect2IPFS()

######   FACE

image1 = open('image/beto1.jpg', 'rb').read()
image2 = open('image/beto2.jpg', 'rb').read()


print("generating biohash 1")
biohash1 = generate_biohash(image1)
print(biohash1)


print("generating biohash 2")
biohash2 = generate_biohash(image2)
print(biohash2)



print("comparing biohash 1 vs biohash 2  FACE")
print(compare_biohash(biohash1, biohash2, False))


### FINGERPRINT

image1 = open('fingerprint/person1-1.png', 'rb').read()
image2 = open('fingerprint/person1-2.png', 'rb').read()

print("generating biohash 1")
biohash1 = generate_biohash(image1)
print(biohash1)


print("generating biohash 2")
biohash2 = generate_biohash(image2)
print(biohash2)

print("comparing biohash 1 vs biohash 2  FINGERPRINT")
print(compare_biohash(biohash1, biohash2, True))

