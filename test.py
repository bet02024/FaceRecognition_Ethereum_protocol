import keras
from keras.preprocessing import image

from BioHash import connect2IPFS, train_model, generate_biohash, compare_biohash, train_fingerprint_model, compare_biohashEuclidianDistance

#print("train_model")
#train_model()


#Uncomment to Train the Model

#train_fingerprint_model()


print("end")


connect2IPFS()

######   FACE

image1 = open('image/beto1.jpg', 'rb').read()
image2 = open('image/beto2.jpg', 'rb').read()
image3 = open('image/beto3.png', 'rb').read()
image4 = open('image/angel1.png', 'rb').read()
image5 = open('image/angel2.png', 'rb').read()


print("generating biohash 1")
biohash1 = generate_biohash(image1)
print(biohash1)
print("generating biohash 2")
biohash2 = generate_biohash(image2)
print(biohash2)
print("generating biohash 3")
biohash3 = generate_biohash(image3)
print(biohash3)

print("generating biohash 4")
biohash4 = generate_biohash(image4)
print(biohash4)
print("generating biohash 5")
biohash5 = generate_biohash(image5)
print(biohash5)

#print("comparing biohash 1 vs biohash 2  FACE")
#print(compare_biohash(biohash1, biohash2, False))
print("comparing biohash 1 vs biohash 2")
print(compare_biohashEuclidianDistance(biohash1, biohash2, False))

print("comparing biohash 1 vs biohash 3")
print(compare_biohashEuclidianDistance(biohash1, biohash3, False))

print("comparing biohash 2 vs biohash 3")
print(compare_biohashEuclidianDistance(biohash2, biohash3, False))


print("comparing biohash 5 vs biohash 4")
print(compare_biohashEuclidianDistance(biohash5, biohash4, False))

print("comparing biohash 5 vs biohash 1")
print(compare_biohashEuclidianDistance(biohash1, biohash5, False))

print("comparing biohash 5 vs biohash 3")
print(compare_biohashEuclidianDistance(biohash2, biohash5, False))

exit(0)
### FINGERPRINT

#image1 = open('fingerprint/person1-1.png', 'rb').read()
#image2 = open('fingerprint/person1-2.png', 'rb').read()

#print("generating biohash 1")
#biohash1 = generate_biohash(image1)
#print(biohash1)


#print("generating biohash 2")
#biohash2 = generate_biohash(image2)
#print(biohash2)

#print("comparing biohash 1 vs biohash 2  FINGERPRINT")
#print(compare_biohash(biohash1, biohash2, True))

