# Integrating Model-Based System Engenieering and Machine Learning with telemetric data

<div align=center>
  <a href="https://www.universidadviu.com/es/"><img src="https://user-images.githubusercontent.com/15159632/155946766-9bf49086-a07f-473c-a703-65c1cc739c9c.png" alt="VIU" title="VIU" hspace="30" height="96px" /></a>
<a href="https://polarisml.space/"><img src="https://user-images.githubusercontent.com/15159632/160088399-417c7b0d-d09c-42cd-869f-94be6f7cb019.png" alt="Polaris" title="Polaris" height="100px" /></a>
</div>

<br/>

Author: Romero Martínez, Manuel ([@ManRom19](https://github.com/ManRom19)) (manuel_rm1993@hotmail.com)

Advisor: Guzman-Alvarez, Cesar Augusto ([@cguz](https://github.com/cguz)) 

---

## Table of Content

- [Overview](#overview)
- [Requirements](#requirements)
  
## Overview

Develop a tool to improve Model-Based Systems Engineering employing Machine Learning techniques using space telemetric data.

We have two main porpuses:

- Using more than one regressor in the graph generation performed by Polaris, in order to compare and be able to choose the one whose results are closest to the expected solution.

	For achieving that purpose, we developed new Python code that allow us to work with 5 available regressors: XGBoosting, RandomForest, AdaBoost, ExtraTrees and GradientBoosting.
	We also modified the Polaris' functions to work with other data sets beside LightSail's. 

- Be able to visualize, study, and accept or discard causal relationships between variables of a data set using Causalnex.
  
	We created a new Python class, that stores the graph with the relationships generated by Causalnex, allow us to visualize it, get its data, and add or remove relationships to the final result. 




## Requirements

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

