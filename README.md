
# BIOHASH

Face Re-Identification Protocol on Ethereum Blockchain


    Blockchain Biometric match platform
    Generate a BioHash and store on-chain to perform a Face Recognition, using
    Smart Contracts & Machine Learning Models.



Solidity, Web3, IPFS & TensorFlow & Keras for Face Recognition protocol.

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

You must need a IPFS Server running on the local Enviroment

### How to install IPFS?
    https://docs.ipfs.io/introduction/install/

Host **127.0.0.1**  (Your Host IP)

Port **5001**       (Your Port)

# Requirements for SmartContract Example

Solidity 0.4.16

NodeJS > 6.0

Web3JS

# How to train the Model?

**Call BioHash.py from Miner Node**

Method **train_model()**

Method  train(pathDir=**"Folder path of the biometric data set**)

Inside pathDir, will be storage all the Biometric dataset for the classifier.

Example structure:

pathDir/train/biohash{1 ... N}/picture{1 ... N}.jpg

pathDir/validation/biohash{1 ... N}/picture{1 ... N}.jpg

*** Pre-Trained Weights ***

model/vggface_tf_resnet50.h5


# How to compare 2 Bio-Hashes ? V1 (Face Only)


**Call BioHash.py from Miner Node**

Method **compare_biohash( hash1, hash2)**

where hash1 & hash2 are the biohashes to compare



# How to compare 2 Bio-Hashes ? V2 (Fingerprint & Face) "Comming Soon ..."

Method **compare_biohash_multimodal( hash1, hash2)**

where hash1 & hash2 are the biohashes to compare



# Smart Contract Example

### BioHashComparation.sol

###### Solidity implementation, How to call the Smart Contract to request a comparison using the BioHash



 **function validateHash(bytes32 _biohashOriginal, bytes32 _biohashToCompare)**

    Request a comparation on the Blockchainm,
    The smart contract write OnChain the Request

 **updateBioHashComparationResult(bytes32 _biohashOriginal, bytes32 _biohashToCompare, bool isMatch)**

    The Miner update the result of the bio match on the Blockchain,
    the smart contract write OnChain the result.



# WEB3 JS Example, Call to the SmartContract using  Web3JS

### SmartContractAPI.js

Web3 example in how to request a call to the smart contract
