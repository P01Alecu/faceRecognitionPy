# Description
The project involves the use of different machine learning models:
- face detection, performed with OpenCV's Cascade Classifier;
- emotion recognition, performed with a CNN.

This repository contains the code for:
 * data exploration and processing of the emotions dataset FER2013 (from Kaggle); 
 * training, fine-tuning;
 * face detection with OpenCV Face Cascade Classifier;
 * real-time video inference on webcam.

# Installation
After cloning or downloading the repository, I suggest you to create a virtual environment through the command `python3 -m venv venv` in the root directory. 
Or if you are using conda I suggest you to create a new virtual environment through the command `conda create --name newEnv python=3.8`.  
It's important as you need to use specific versions of the libraries used, for compatibility reasons.

## Dependencies
Once the venv has been created, install the dependencies with `pip3 install -r requirements.txt` in a venv enabled shell. <br>

In order to use the application, you should have the model named 'FER.h5' in the same directory with the .exe. <br>
You can download them from https://drive.google.com/drive/folders/1v4DG-_fu1R4weJ1A0MZQhQZcBT6TTxsQ?usp=sharing

## Quick info
train.ipynb  contains the code used for training the models
testData.py test the accuracy of the trained model using the webcam
getFilesForTest.py contains helper functions