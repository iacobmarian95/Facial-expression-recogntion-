# Facial-expression-recogntion-
A solution for Facial Expression Recognition using RESNET50 Convolutional Neural Network.</br>

You can find more about RESNET50 here:</br>
https://medium.com/@14prakash/understanding-and-implementing-architectures-of-resnet-and-resnext-for-state-of-the-art-image-cf51669e1624

The dataset used is fer2013 from: <br>
https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

The images are comprimed into a .csv file. </br>
csvToImage.py extract data from this .csv file an organize it into a more convenient way. </br> </br>
The resnetModel.py build, train and evaluate the model. Also the weights are saved into a .json format.
