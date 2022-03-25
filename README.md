# Integrating Model-Based System Engenieering and Machine Learning with telemetric data

<div align=center>
  <a href="https://www.universidadviu.com/es/"><img src="https://user-images.githubusercontent.com/15159632/155946766-9bf49086-a07f-473c-a703-65c1cc739c9c.png" alt="VIU" title="VIU" hspace="30" height="96px" /></a>
<a href="http://visionspace.com/"><img src="https://user-images.githubusercontent.com/15159632/117484138-f7920900-af66-11eb-8def-6e9880860c4a.png" alt="VisionSpace" title="VisionSpace" height="100px" /></a>
</div>

<br/>

Author: Romero Martínez, Manuel ([@ManRom19](https://github.com/ManRom19)) (manuel_rm1993@hotmail.com)

Advisor: Guzman-Alvarez, Cesar Augusto ([@cguz](https://github.com/cguz)) 

---

## Table of Content

- [Integrating Model-Based System Engenieering and Machine Learning with telemetric data](#integrating-model-based-system-engenieering-and-machine-learning-with-telemetric-data)
	- [Table of Content](#table-of-content)
	- [Overview](#overview)
	- [Requirements](#requirements)
  
## Overview

This project has been made following two porpuses:

- Using more than one regressor in the graph generation performed by Polaris, in order to compare and be able to choose the one whose results are closest to the expected solution.

	For achieving that purpose, we will create a new Python class which will allow us to work with one data set, and with multiple graphs, one per
	each chosen regressor. The chosen regressors must be among the 5 available regressors: XGBoosting, RandomForest, AdaBoost, ExtraTrees and GradientBoosting.
	We will also modify some of the Polaris' functions to make the code functional for data sets beside LightSail's. 

- Be able to visualize, study, and accept or discard causal relationships between variables of a data set using Causalnex.
  
	We will create a new Python class, that will store the graph with the relationships generated by Causalnex, allow us to visualize it, get its data, and add or remove relationships to the final result. 




## Requirements

This work is implemented in python and for both the Dataset engineering and the Sequencer algorithm, requires the following:

- python 3.8.5
- numpy 1.19.2
- pandas 1.1.3
- jupyter 1.0.0
- polaris-ml 0.13.5
- pygraphviz 1.8
- ipython 7.19.0
- fets 0.5.3
- mlflow 1.22.0
- causalnex 0.11.0

