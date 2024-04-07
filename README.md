# Data 4010: Exploring Datasets for Machine-Learning-Enabled Microwave Imaging
This repository contains code for Data 4010 project working on leveraging the power of machine learning and deep learning to explore microwave imaging datasets. Our theory is that Minibatch KMeans clustering can be used on the field image to create clusters similar target (shape target) or frequency. Moreover, two models of U-Net CNN were built to work on image segmentation of EMNIST character data. 

Please find the detailed project report in Data_4010_Report_Mai_Nguyen.pdf.

## Usage

### Scripts for Generating Synthetic Data

To generate data needed for this project, contact Dr. Ian Jeffrey for access to the Richmond Solver and scripts to read the output image data then pick one of the following scripts to generate:
- To get the EMNIST database as input to the Richmond Solver, run download_emnist.py
- To generate shape object dataset, run either createCircleData.py, createSquareData.py or createTriangleData.py 

To plot sample image data, run plot_img_for_view.py. This script also allows saving the sample image to your local computer.

### Jupyter Notebooks for the Models

There are two notebooks containing the clustering algorithm (ShapeImageClustering.ipynb) and U-Net CNN (UNET CNN.ipynb). Place the folder with all the script generated image data into the same directory with the notebooks and run each cell or all cells of the notebook.

### Requirement

- Python 3.10.4
- Jupyter Notebook
- keras
- matplotlib
- numpy
- pandas
- sklearn
- tensorflow
- urllib
- random
- os