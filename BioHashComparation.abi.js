[
	{
		"constant": true,
		"inputs": [],
		"name": "biohashToCompare",
		"outputs": [
			{
				"name": "",
				"type": "bytes32"
			}
		],
		"payable": false,
		"stateMutability": "view",
		"type": "function"
	},
	{
		"constant": true,
		"inputs": [],
		"name": "status",
		"outputs": [
			{
				"name": "",
				"type": "uint8"
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
				"name": "_caller",
				"type": "address"
			}
		],
		"name": "BioHash",
		"outputs": [],
		"payable": false,
		"stateMutability": "nonpayable",
		"type": "function"
	},
	{
		"constant": false,
		"inputs": [
			{
				"name": "_biohashOriginal",
				"type": "bytes32"
			},
			{
				"name": "_biohashToCompare",
				"type": "bytes32"
			}
		],
		"name": "validateHash",
		"outputs": [],
		"payable": false,
		"stateMutability": "nonpayable",
		"type": "function"
	},
	{
		"constant": true,
		"inputs": [],
		"name": "biohashOriginal",
		"outputs": [
			{
				"name": "",
				"type": "bytes32"
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
				"type": "bytes32"
			},
			{
				"name": "_biohashToCompare",
				"type": "bytes32"
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
		"constant": false,
		"inputs": [
			{
				"name": "_biohashOriginal",
				"type": "bytes32"
			},
			{
				"name": "_biohashToCompare",
				"type": "bytes32"
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
		"inputs": [],
		"name": "caller",
		"outputs": [
			{
				"name": "",
				"type": "address"
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
				"type": "bytes32"
			},
			{
				"indexed": false,
				"name": "biohashToCompare",
				"type": "bytes32"
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
				"name": "biohashOriginal",
				"type": "bytes32"
			},
			{
				"indexed": false,
				"name": "biohashToCompare",
				"type": "bytes32"
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
