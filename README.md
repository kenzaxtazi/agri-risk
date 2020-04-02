# Agri-Risk
This repo contains the code used to produce the Guided Team Challenge Report concerning predicting yield changes in maize globally.

It includes scripts that recreate steps detailed in the report.


## Notebooks and Files
In an effort to make this code as simple to understand as possible, the code is in multiple different scripts that each have a specific function as follows:

* Create Data Sets. This script is the first port of call. It creates the data set for 2010 used to train the machine learning models. It computes the agroclimatic indicators for both 2005 and 2010. It reads from netcdf4 files. For more information on the netcdf4 files used, please read the script and follow the link to the data source.

* Recursive Feature Elimination. This script details how we used recursive feature elimination to reduce the training data dimensions.

* Model Exploration. This Notebook contains the code to train three ensemble methods- Random Forest, Extra Trees and XGBoost on the crop data.

* Neural Network Training. This file contains the code used for training a Neural Network on one of these data sets.

* Prediction. This file contains the code for using a saved regressor to predict the yield given the agroclimatic indicators from computed from a GCM model.

* Prediction Analysis. This script contains a method to produce plots analyzing the residual of the prediction versus the truth for predicted yields.

* Data Exploration. Contains methods that are used to plot and map data including global crop yields as well as features like soil and elevation.

* shared_methods.py A python file containing methods used by multiple files to avoid repetition and errors.

* visualisation_tools. Methods for mapping and plotting prediction data.

All JSON data used is stored in the json_data folder.


## Requirements
All packages required are in 'requirements.txt'
The python version is 3.7.0
