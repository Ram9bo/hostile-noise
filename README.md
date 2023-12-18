# hostile-noise
API Final Project

# Data
Dataset needs to be obtained from https://github.com/karolpiczak/ESC-50

# Training a model
Execute run.py

This will perform hyperparameter tuning followed by training and evaluation of models with the best configuration. 

Note: due to the absence of pooling layers, the model is very large and requires considerable computational resources.

# Classifying live sounds
Execute ClassifyLive.py

This will use the model at Model/model.keras to classify the sounds picked up by the microphone.