# Integrating Model-Based System Engenieering and Machine Learning with telemetric data

<div align=center>
  <a href="https://www.universidadviu.com/es/"><img src="https://user-images.githubusercontent.com/15159632/155946766-9bf49086-a07f-473c-a703-65c1cc739c9c.png" alt="VIU" title="VIU" hspace="30" height="96px" /></a>
  <a href="https://iac.es/"><img src="https://user-images.githubusercontent.com/15159632/155946637-70e34166-80c8-407b-a719-c5d1ad421118.png" alt="IAC" title="IAC" hspace="30" height="96px" /></a>
<center><a href="http://visionspace.com/"><img src="https://user-images.githubusercontent.com/15159632/117484138-f7920900-af66-11eb-8def-6e9880860c4a.png" alt="VisionSpace" title="VisionSpace" height="100px" /></a></center>

</div>

<br/>

Author: Moras Acosta, Manuel David ([@br0ly23](https://github.com/br0ly23)) (manomoras@gmail.com)

Advisor: Guzman-Alvarez, Cesar Augusto ([@cguz](https://github.com/cguz)) 

Co-Advisor: Garcia Lorenzo, Begona (-) 

---

## Table of Content

- [Overview](#overview)
- [Requirements](#requirements)
- [How it works](#how-it-works)
- [Source code](#source-code)
- [Selected Results](#selected-results)
- [Publications](#publications)
- [References](#references)

## Overview




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




