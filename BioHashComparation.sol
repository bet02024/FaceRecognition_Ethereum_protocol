pragma solidity ^0.4.21;

contract BioHashComparation {

    enum State { Created, Working, Done }

    struct BioHahs{
        address owner;
        string biohashOriginal;
        string biohashToCompare;
        uint idTransaction;
        State status;
        bool isBiometricMatch;
        bool isFingerprint;
    }

    uint public transaction_count;

    mapping(uint=>BioHahs) public biohash_info;

    event HashMatchingRequested(string biohashOriginal, string biohashToCompare,bool isFingerprint, uint idTransaction);
    event HashMatchingDone(uint idTransaction, bool resultComparation);



    function getTotalNumberOfTransactions() public view returns (uint){
      return transaction_count;
    }

    function storeBioHashRequest(string _biohashOriginal, string _biohashToCompare, bool _isFingerprint) public  {
        transaction_count++;
        biohash_info[transaction_count]=BioHahs(
          {
                owner: msg.sender,
                biohashOriginal: _biohashOriginal,
                biohashToCompare: _biohashToCompare,
                status: State.Created,
                idTransaction: transaction_count,
                isBiometricMatch: false,
                isFingerprint: _isFingerprint
          }
       );
        //The MINER RUN A VALIDATION WITH SIGNAL, PULL THE BIOHASH FROM BLOCKCHAIN
        emit HashMatchingRequested(_biohashOriginal, _biohashToCompare, _isFingerprint, transaction_count);
    }

    function startMinningEvent(uint transactionId) public  {
        biohash_info[transactionId].status = State.Working;
    }

    /// THE MINER UPDATES THE RESULT, WRITE ONCHAIN THE BIO HASH RESULT
    function updateBioHashComparationResult(uint transactionId, bool isMatch) public returns (bool) {

        biohash_info[transactionId].status = State.Done;
        biohash_info[transactionId].isBiometricMatch = isMatch;
        emit HashMatchingDone( transactionId, isMatch);

    }


}
