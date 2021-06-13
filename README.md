# Create-Your-Own-Image-Classifier
Project 2 from Udacity AI Progaming with Python Nanodegree Program

## Description
This code does the following:
1. Train a deep learning model on a dataset of flower images.
2. Use the trained model to classify flower images.

There are two parts in this project:
1. A Jupyter Notebook file to make sure the code implementation works
2. A Python package that runs in command line to implement the functions 

## Requirements
To train network, use CUDA capable GPU.  
To predict class of flower image with trained model, normal CPU should be sufficient.
  
Libraries required are as follow:
- PyTorch
- Numpy

## Instruction
Run script in command with arguments as shown:
```
>>> python train.py flowers/ --epoch 6 --arch VGG --hidden_units 6272 --gpu
```
and
```
>>> python prediction.py my_images/image.jpg checkpoint_dir/ --top_k 5 --gpu
```
For information on other arguments, call help message with `-h`
