pragma solidity ^0.4.21;

contract BioHashComparation {

    // Current state of the biohash transaction.
    address public caller;
    bytes32 public biohashOriginal;
    bytes32 public biohashToCompare;
    enum State { Created, Working, Done }
    State public status;
    bool isBiometricMatch;

    // Events that will be fired on changes.
    event HashMatchingRequested( bytes32 biohashOriginal, bytes32 biohashToCompare);
    event HashMatchingDone(bytes32 biohashOriginal, bytes32 biohashToCompare, bool resultComparation);

    function BioHash(
        address _caller
    ) public {
        caller = _caller;
    }

    // REQUEST A BIOMETRIC VALIDATION
    function validateHash(bytes32 _biohashOriginal, bytes32 _biohashToCompare) public  {

        caller = msg.sender;
        biohashOriginal = _biohashOriginal;
        biohashToCompare = _biohashToCompare;
        status = State.Created;
        //The MINER RUN A VALIDATION WITH SIGNAL, PULL THE BIOHASH FROM BLOCKCHAIN
        emit HashMatchingRequested(biohashOriginal, biohashToCompare);
    }

    function startMinningEvent(bytes32 _biohashOriginal, bytes32 _biohashToCompare) public  {
        if (_biohashOriginal == biohashOriginal && _biohashToCompare == biohashToCompare) {

            status = State.Working;
        }
    }


    /// THE MINER UPDATES THE RESULT, WRITE ONCHAIN THE BIO HASH RESULT
    function updateBioHashComparationResult(bytes32 _biohashOriginal, bytes32 _biohashToCompare, bool isMatch) public returns (bool) {

        status = State.Done;

        if (_biohashOriginal == biohashOriginal && _biohashToCompare == biohashToCompare) {

            isBiometricMatch = isMatch;
            emit HashMatchingDone( biohashOriginal, biohashToCompare, isMatch);
        }
        return true;
    }
}
