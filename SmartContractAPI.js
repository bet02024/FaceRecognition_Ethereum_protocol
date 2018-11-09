var Web3 = require('web3');
var web3Provider = null;
var BioHashComparation;

var contract = require('./SmartContract/BioHashComparation.sol');

function init() {
  //initializing web3 to access blockchain
  initweb3();
}

//########################## CHANGE THIS ADDRESS
var SmartContractAddress = "0x6c68d153b9709283e3900e944f1c6677273987c1";

var BioHashComparation;  //SmartContract Intance

function initweb3() {

  if (typeof web3 !== 'undefined') {
    web3 = new Web3(web3.currentProvider);

  } else {

    web3 = new Web3(new Web3.providers.HttpProvider("http://localhost:7545"));

  }

  web3.eth.defaultAccount = web3.eth.accounts[1];

  var BioHashComparationContractAddress = SmartContractAddress;

  BioHashComparation = new web3.eth.Contract(contract,BioHashComparationContractAddress );


}


#### Biometric CALL To SMART CONTRACT

function performBiometricValidation(String Hash1, String Hash2){

 //Compare BIOHASH 1 vs BIOHASH 2

  BioHashComparation.validateHash(Hash1, Hash2);

}


####  CALL To CHECK THE RESULT

function getBiometricValidation(String Hash1, String Hash2){

 //Compare BIOHASH 1

  if ( BioHashComparation.status == "0x2"){
    return BioHashComparation.isBiometricMatch;
  } else  {
    return null;
  }

}





