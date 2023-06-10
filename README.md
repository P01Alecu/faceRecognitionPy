# Description
The project involves the use of different machine learning models:
- face detection, performed with OpenCV's Cascade Classifier;
- emotion recognition, performed with a CNN.

This repository contains the code for:
 * data exploration and processing of the emotions dataset FER2013 (from Kaggle); 
 * training, fine-tuning and onnx conversion for emotion model;
 * face detection with OpenCV Face Cascade Classifier;
 * real-time video inference on webcam.

# Installation
After cloning or downloading the repository, I suggest you to create a virtual environment through the command `python3 -m venv venv` in the root directory. 
Or if you are using conda I suggest you to create a new virtual environment through the command `conda create --name newEnv python=3.8`.  
It's important as you need to use specific versions of the libraries used, for compatibility reasons.

## Dependencies
Once the venv has been created, install the dependencies with `pip3 install -r requirements.txt` in a venv enabled shell.

## Quick info
testData.py test the accuracy of the trained model (dataset + haarcascade care se gaseste in opencv/data/haarcascades) 
testWithWeb.py test the model with the webcam (webcam + haarcascade care se gaseste in opencv/data/haarcascades)

# affectNet_v1.h5 accuracy - 65
# affectNet_v2 70.10

# FER v1 - 60.87 100epochs
# FER v2 - 54.34 56epochs trebuie re-rulat
# FER v3 - 54.79 300epochs