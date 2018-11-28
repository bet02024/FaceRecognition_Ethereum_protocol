import keras
from keras.preprocessing import image

from BioHash import connect2IPFS, train_model, generate_biohash, compare_biohash

print("train_model")
#train_model()
print("end")
 

connect2IPFS()
image1 = open('image/beto1.jpg', 'rb').read()
image2 = open('image/beto2.jpg', 'rb').read()


print("generating biohash 1")
biohash1 = generate_biohash(image1)
print(biohash1)



print("generating biohash 2")
biohash2 = generate_biohash(image2)
print(biohash2)


print("comparing biohash 1 vs biohash 2")
print(compare_biohash(biohash1, biohash2))


