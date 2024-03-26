import pickle
import numpy as np
import os
import math

class EMSample:
    def __init__(self, target, Esct, freq):
        self.target = target
        self.Esct = Esct
        self.freq = freq        
        
class GeneralData:
    def __init__(self, inp, label):
        self.inp = inp
        self.label = label
    

def getEMData(path):
    root = path
    files = os.listdir(root)
    allSamples = []

    for file in files:
        if os.path.isfile(root+file):
            with open(root+file, 'rb') as fd:
                try:
                    kai = pickle.load(fd)
                    sct = pickle.load(fd)
                    freq = pickle.load(fd)
                    assert kai.shape[2] == sct.shape[2]
                    for i in range(kai.shape[2]):
                        newSample = EMSample(kai[:, :, i], sct[:, :, i], freq)
                        allSamples.append(newSample)
                except:
                    print(file + " was not formatted as expected")
    print("Read " + str(len(allSamples)) + " samples")
    return allSamples

def getGenericData(path):
    root = path
    files = os.listdir(root)
    allSamples = []
    for file in files:
        if os.path.isfile(root + file):
            with open(root + file, 'rb') as fd:
                try:
                    kai = pickle.load(fd)
                    sct = pickle.load(fd)
                    assert kai.shape[2] == sct.shape[2]
                    for i in range(kai.shape[2]):
                        newSample = GeneralData(sct[:, :, i], kai[:, :, i])
                        allSamples.append(newSample)
                except:
                    print(file + " was not formatted as expected")
    print("Read " + str(len(allSamples)) + " samples")
    return allSamples

# To be used when Keeley gives you data.
# Disclaimer: Results may vary if Keeley varies her formatting
def getDataFromKeeley(path):
    root = path
    files = os.listdir(root)
    labels = None
    inps = None
    allSamples = []
    for file in files:
        if os.path.isfile(root + file):
            if "labels" in file:
                with open(root + file, 'rb') as fd:
                    labels = pickle.load(fd)
            else:
                with open(root + file, 'rb') as fd:
                    inps = pickle.load(fd)

    # okay now they are in two matricies so lets build my list of samples
    # This is unecessary but it meets what the U-Net expects.
    assert labels.shape == inps.shape
    for i in range(labels.shape[2]):
        newSample = GeneralData(inps[:, :, i], labels[:, :, i])
        allSamples.append(newSample)
    print("Read " + str(len(allSamples)) + " samples")
    return allSamples

def wrapData(inputData):
    # Column wrapping
    leftCol1 = np.expand_dims(inputData[:,:,-2,:], axis=2)
    rightCol1 = np.expand_dims(inputData[:,:,1,:], axis=2)
    leftCol0 = np.expand_dims(inputData[:,:,-1,:], axis=2)
    rightCol0 = np.expand_dims(inputData[:, :, 0, :], axis=2)
    inputData = np.concatenate([leftCol1, leftCol0, inputData, rightCol0, rightCol1], axis=2)

    # Row wrapping
    topRow1 = np.expand_dims(inputData[:,-2,:,:], axis=1)
    topRow0 = np.expand_dims(inputData[:,-1,:,:], axis=1)
    botRow1 = np.expand_dims(inputData[:,1,:,:], axis=1)
    botRow0 = np.expand_dims(inputData[:,0,:,:], axis=1)
    inputData = np.concatenate([topRow1, topRow0, inputData, botRow0, botRow1], axis=1)

    return inputData