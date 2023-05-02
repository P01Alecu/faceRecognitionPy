# Description
The project involves the use of different machine learning models:
- face detection, performed with OpenCV's Cascade Classifier;
- face recognition, done with OpenCV's LBPHFaceRecognizer;
- emotion recognition, performed with a CNN built from scratch.

We decided to use only FER2013 Dataset and a CNN to understand the difficulties around this task and we tried to improve the accuracy with hyperparameter optimization. Furthermore, we added some CV techniques (confirmation window, depth segmentation) to improve real execution performance.

This repository contains the code for:
 * data exploration and processing of the emotions dataset FER2013 (from Kaggle); 
 * training, fine-tuning and onnx conversion for emotion model;
 * face detection with OpenCV Face Cascade Classifier;
 * data acquisition for face recognition;
 * face recognition with OpenCV LBPHFaceRecognizer; 
 * real-time video inference on webcam / kinect.

# Installation
After cloning or downloading the repository, I suggest you to create a virtual environment through the command `python3 -m venv venv` in the root directory. 
Or if you are using conda I suggest you to create a new virtual environment through the command `conda create --name newEnv python=3.8`.  
It's important as you need to use specific versions of the libraries used, for compatibility reasons.

## Dependencies
Once the venv has been created, install the dependencies with `pip3 install -r requirements.txt` in a venv enabled shell.

## Quick info
testData.py test the accuracy of the trained model (dataset + haarcascade care se gaseste in opencv/data/haarcascades) 
testWithWeb.py test the model with the webcam (webcam + haarcascade care se gaseste in opencv/data/haarcascades)



# TO-DO
Face recognition - use face recognition trained for each user in order to save their emotion history  
Transfer learning - train a model with better accuracy using transfer learning  
                - fine tune it  
GUI - add webcam with recognition  
    - button to add user (take photos and train a model)  
    - button to record emotions  
Plot for the trained models  
Write 