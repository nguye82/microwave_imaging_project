from RichmondSolver import *
from ReadData import *
import matplotlib.pyplot as plt
import numpy as np
from scipy import io as sio
import random

mat = sio.loadmat('./emnist-letters.mat')
data = mat['dataset']

X_train = data['train'][0, 0]['images'][0, 0]
y_train = data['train'][0, 0]['labels'][0, 0]
X_test = data['test'][0, 0]['images'][0, 0]
y_test = data['test'][0, 0]['labels'][0, 0]

X_train = X_train.reshape((X_train.shape[0], 28, 28))
X_test = X_test.reshape((X_test.shape[0], 28, 28))

nn = 50  # no. of data that you want to generate
count = random.randint(0, X_train.shape[0]-nn)
# lets set up the forward solver
# freq = [1e9, 2e9, 3e9]

# Generate a random array with 20 values in the range [1e9, 3e9)
# freq = np.random.uniform(low=1e9, high=3e9, size=20)
freq = 2e9

numRx = 24
noOfTransmitters = 24

for f in range(0, len(freq)):
    A = np.ones(noOfTransmitters)
    fi = np.linspace(0, 2 * np.pi - 2 * np.pi / noOfTransmitters, noOfTransmitters)
    A = np.reshape(A, (noOfTransmitters, 1))
    fi = np.reshape(fi, (noOfTransmitters, 1))
    Transmitters = np.concatenate((A, fi), axis=1)

    targetNx = X_train[count].shape[0]
    targetNy = X_train[count].shape[1]
    domainNx = 35
    domainNy = 35
    backgroundPerm = 1
    epsBackground = np.ones((domainNy, domainNx)) * backgroundPerm


    CC = 299792458  # speed of light in vacuum, M/s
    mu0 = np.pi * 4e-7  # permeability
    epsilon0 = 1 / mu0 / CC / CC  # permitivity
    lambdaa = CC / np.sqrt(backgroundPerm.real) / freq[f]
    xMax = 0.2
    xMin = -0.2
    yMax = xMax
    yMin = xMin

    # cell widths and positions of the centroids
    cellXWidth = (xMax - xMin) / domainNx
    print(lambdaa/cellXWidth)
    cellYWidth = (yMax - yMin) / domainNy
    xCentroids = np.linspace(xMin + cellXWidth / 2, xMax - cellXWidth / 2, domainNx)
    yCentroids = np.linspace(yMin + cellYWidth / 2, yMax - cellYWidth / 2, domainNy)
    xCentroids = np.reshape(xCentroids, (domainNx, 1))
    yCentroids = np.reshape(yCentroids, (domainNy, 1))

    X, Y = np.meshgrid(xCentroids, yCentroids)

    # target grid stuff
    xTargetMax = 0.075
    xTargetMin = -0.075
    yTargetMax = xTargetMax
    yTargetMin = xTargetMin

    cellXWidthTarg = (xTargetMax - xTargetMin) / domainNx
    cellYWidthTarg = (yMax - yMin) / domainNy
    xCentroidsTarg = np.linspace(xTargetMin + cellXWidthTarg / 2, xTargetMax - cellXWidthTarg / 2, targetNx)
    yCentroidsTarg = np.linspace(yTargetMin + cellYWidthTarg / 2, yTargetMax - cellYWidthTarg / 2, targetNy)
    xCentroidsTarg = np.reshape(xCentroidsTarg, (targetNx, 1))
    yCentroidsTarg = np.reshape(yCentroidsTarg, (targetNy, 1))

    obsRadius = 0.15  # radius
    obsAngle = np.linspace(0, (2 * np.pi - (2 * np.pi / numRx)),
                            numRx)  # radian-based positions of the receiver points.
    obsAngle = np.reshape(obsAngle, (numRx, 1))

    xObs = obsRadius * np.cos(obsAngle)
    yObs = obsRadius * np.sin(obsAngle)

    ObsPoints = np.concatenate((xObs, yObs), axis=1)

    fwd = RichmondSolver(freq[f], epsBackground, X, Y, domainNx, domainNy, ObsPoints, Transmitters)

    for i in range(nn):
        count += 1
        print("Target Num " + str(count))
        print(i)
        # epsCylinder = randomPerm[count]
        epsTarget = np.where(X_train[count] != 0, 2.0, backgroundPerm)  # this will be the local
        # loop over the target grid and collect coords where the target is
        coordsWTarget = []
        for xInd in range(targetNx):
            for yInd in range(targetNy):
                if epsTarget[yInd, xInd] == 2:
                    coordsWTarget.append((xInd, yInd))

        assert len(coordsWTarget) > 0  # make sure we have a target
        peakPermSpot = random.choice(coordsWTarget)
        peakVal = random.gauss(2.7, 0.25)
        # now lets loop over the list of coords in the target and apply a perm value proportional to how far it is from the peak
        for coord in coordsWTarget:
            diff = np.linalg.norm([peakPermSpot[0] - coord[0], peakPermSpot[1] - coord[1]])

            epsTarget[coord[1], coord[0]] = (peakVal-backgroundPerm) * np.exp(-0.01*diff) + backgroundPerm

        # target which needs to be overlayed onto the domain
        epsTargetOnDomain = np.zeros(epsBackground.shape)+backgroundPerm
        xTCordsPlotting = []
        yTCordsPlotting = []
        xCordsPlotting = []
        yCordsPlotting = []

        for xInd in range(targetNx):
            for yInd in range(targetNy):
                xCoord = xCentroidsTarg[xInd]
                xTCordsPlotting.append(xCoord)
                yCoord = yCentroidsTarg[yInd]
                yTCordsPlotting.append(yCoord)
                diffX = np.abs(xCentroids - xCoord)
                diffY = np.abs(yCentroids - yCoord)
                xIndDomain = np.argmin(diffX)
                yIndDomain = np.argmin(diffY)
                xCordsPlotting.append(xCentroids[xIndDomain])
                yCordsPlotting.append(yCentroids[yIndDomain])
                epsTargetOnDomain[yIndDomain, xIndDomain] = epsTarget[yInd, xInd]
        

        print("value = ", str(np.max(epsTargetOnDomain)))
        '''
        fig, ax = plt.subplots(1, 2)
        ax[0].scatter(xTCordsPlotting, yTCordsPlotting)
        ax[1].scatter(xCordsPlotting, yCordsPlotting)
        plt.show()
        '''
        fwd.simulate(epsTargetOnDomain, show=False, path = "./20FreqInRange/")
