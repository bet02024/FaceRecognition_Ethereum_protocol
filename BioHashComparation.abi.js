[
	{
		"constant": false,
		"inputs": [
			{
				"name": "transactionId",
				"type": "uint256"
			}
		],
		"name": "startMinningEvent",
		"outputs": [],
		"payable": false,
		"stateMutability": "nonpayable",
		"type": "function"
	},
	{
		"constant": true,
		"inputs": [
			{
				"name": "",
				"type": "uint256"
			}
		],
		"name": "biohash_info",
		"outputs": [
			{
				"name": "owner",
				"type": "address"
			},
			{
				"name": "biohashOriginal",
				"type": "string"
			},
			{
				"name": "biohashToCompare",
				"type": "string"
			},
			{
				"name": "idTransaction",
				"type": "uint256"
			},
			{
				"name": "status",
				"type": "uint8"
			},
			{
				"name": "isBiometricMatch",
				"type": "bool"
			},
			{
				"name": "isFingerprint",
				"type": "bool"
			}
		],
		"payable": false,
		"stateMutability": "view",
		"type": "function"
	},
	{
		"constant": false,
		"inputs": [
			{
				"name": "_biohashOriginal",
				"type": "string"
			},
			{
				"name": "_biohashToCompare",
				"type": "string"
			},
			{
				"name": "_isFingerprint",
				"type": "bool"
			}
		],
		"name": "storeBioHashRequest",
		"outputs": [],
		"payable": false,
		"stateMutability": "nonpayable",
		"type": "function"
	},
	{
		"constant": true,
		"inputs": [],
		"name": "getTotalNumberOfTransactions",
		"outputs": [
			{
				"name": "",
				"type": "uint256"
			}
		],
		"payable": false,
		"stateMutability": "view",
		"type": "function"
	},
	{
		"constant": false,
		"inputs": [
			{
				"name": "transactionId",
				"type": "uint256"
			},
			{
				"name": "isMatch",
				"type": "bool"
			}
		],
		"name": "updateBioHashComparationResult",
		"outputs": [
			{
				"name": "",
				"type": "bool"
			}
		],
		"payable": false,
		"stateMutability": "nonpayable",
		"type": "function"
	},
	{
		"constant": true,
		"inputs": [],
		"name": "transaction_count",
		"outputs": [
			{
				"name": "",
				"type": "uint256"
			}
		],
		"payable": false,
		"stateMutability": "view",
		"type": "function"
	},
	{
		"anonymous": false,
		"inputs": [
			{
				"indexed": false,
				"name": "biohashOriginal",
				"type": "string"
			},
			{
				"indexed": false,
				"name": "biohashToCompare",
				"type": "string"
			},
			{
				"indexed": false,
				"name": "isFingerprint",
				"type": "bool"
			},
			{
				"indexed": false,
				"name": "idTransaction",
				"type": "uint256"
			}
		],
		"name": "HashMatchingRequested",
		"type": "event"
	},
	{
		"anonymous": false,
		"inputs": [
			{
				"indexed": false,
				"name": "idTransaction",
				"type": "uint256"
			},
			{
				"indexed": false,
				"name": "resultComparation",
				"type": "bool"
			}
		],
		"name": "HashMatchingDone",
		"type": "event"
	}
]
