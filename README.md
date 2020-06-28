# LUNAR
LUNAR ：Novel Coronavirus Target Prediction and Drug Discovery Based on Representation Learning Graph Convolutional Network

environment:
  python:3.7.4
  tensorflow-gpu:1.14.0
  scikit-learn:0.22.1
  pandas:1.0.1
  numpy:1.16.2
  
start:
  1.python main.py                    #train model and get repositioning network
  2.python ncov_predict.py            #get predicted results
