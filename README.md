# hostile-noise
API Final Project

# Data
Dataset needs to be obtained from https://github.com/karolpiczak/ESC-50

Dataset zip file should be unzipped in a folder called 'data'

The categorization of sounds into hostile and non-hostile categories can be found in hostility.json

# Training a model
Execute run.py

This will perform hyperparameter tuning followed by training and evaluation of models with the best configuration. 

A model will be created at Model/model.keras during this run.

Note: due to the absence of pooling layers, the model is very large and requires considerable computational resources.

# Classifying live sounds
Execute ClassifyLive.py

This will use the model at Model/model.keras to classify the sounds picked up by the microphone.