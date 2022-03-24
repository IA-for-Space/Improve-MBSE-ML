# Improve-MBSE-ML


## Project structure

``` BASH
polaris/                   					- Polaris original code

src/

	python/                					- Code that has been modified from the original source
	
		polaris/           					- Polaris original and modified code
		
		polaris-own-code/  					- Only Polaris modified code
		
	datasets/              					- Non-processed original datasets of LightSail2 and Mars Express
	
	polaris_workspace/     					- Polaris programming and testing space
	
		datasets/          					- Modified Mars Express and LightSail2 preprocessed datasets
		
		lightsail_graphs/  					- Generated graphs.json by Polaris for LightSail2 datasets
		
		marsexpress_graphs/ 				- Generated graphs.json by Polaris for Mars Express datasets
		
		polaris/            				- Polaris original code
		
		polaris_mod_project/				- Polaris modified code programming and testing
		
			polaris/                    	- Polaris modified code
			
			polaris_mars_express_graphs/	- Generated graphs.json by Polaris for Mars Express datasets using Polaris modified code
			
		pre-process/           				- Pre-processed LightSail2 and Mars Express datasets
		
	causalnex_workspace/     				- Causalnex programming and testing space
	
		datasets/          					- Modified Mars Express and LightSail2 preprocessed datasets
		
		causalnex_develop/  				- Causalnex original code
		
		causalnex_guide_code/ 				- Testing notebooks following Causalnex beginner\'s guide
		
		causalnex_mod_project/            	- New Causalnex class and notebooks to work with it.	
		
		preprocess_mars_express/           	- Pre-processed Mars Express dataset

```

## Polaris modified code

The files that have been modified in Polaris are:

	polaris/polaris/learn/analysis.py
	
	polaris/polaris/learn/predictor/cross_correlation.py
	

## Causalnex modified code

No original causalnex source code has been changed, just one new class has been created: CausalnexDataset.py




