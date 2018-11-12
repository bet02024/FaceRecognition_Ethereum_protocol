


BioHahs, Blockchain Biometric Validation using Hash onchain to perform a Biometric Face Recognition

IPFS & ML For Face Recognition

# Requirements for ML Models

Python 3.6

Keras six>=1.9.0

TensorFlow > 1

numpy >=1.9.1

scipy >=0.14

h5py

pillow

pyyaml

matplotlib

# Requirements for IPFS

IPFS Server running on the local Enviroment

Host **127.0.0.1**

Port **5001**

# Requirements for SmartContract Example

Solidity 0.4.16

NodeJS > 6.0

Web3JS

# How to train the Model?

**BioHash.py**

Method  train(pathDir=**"Folder path of the biometric data set**)

Inside pathDir, will be storage all the Biometric dataset for the classifier.

Example structure:

pathDir/train/biohash{1 ... N}/picture{1 ... N}.jpg

pathDir/validation/biohash{1 ... N}/picture{1 ... N}.jpg

*** Pre-Trained Weights ***

model/vggface_tf_resnet50.h5



# Smart Contract Example

### BioHashComparation.sol
 Solidity Example,
 How to use the hash to request a comparison using the BioHash

