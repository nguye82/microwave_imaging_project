# Data 4010: Exploring Datasets for Machine-Learning-Enabled Microwave Imaging
This repository contains code for Data 4010 project. 

## Scripts for Generating Synthetic Data

To generate data needed for this project, contact Dr. Ian Jeffrey for access to the Richmond Solver and scripts to read the output image data then pick one of the following scripts to generate:
- To get the EMNIST database as input to the Richmond Solver, run download_emnist.py
- To generate shape object dataset, run either createCircleData.py, createSquareData.py or createTriangleData.py 

To plot sample image data, run plot_img_for_view.py. This script also allows saving the sample image to your local computer.

## Jupyter Notebooks for the Models

There are two notebooks containing the clustering algorithm (ShapeImageClustering.ipynb) and U-Net CNN (UNET CNN.ipynb).

Our theory was that Minibatch KMeans clustering on the field image to create clusters similar target (shape target) or frequency. 

Two models of U-Net CNN were built to work on image segmentation of EMNIST character data.

### Libraries you will need
#### All code is in Python 3.10.4

- keras
- matplotlib
- numpy
- pandas
- sklearn
- tensorflow
- urllib
- random
- os