from RichmondSolver import RichmondSolver
import numpy as np
import matplotlib.pyplot as plt
import random

nn = 50
count = random.randint(0, nn)

numRx = 24
numTx = 24

# freq = 3e9
freq = [1e9, 2e9, 3e9]

for f in freq:
    A = np.ones(numTx)
    phi = np.linspace(0, 2 * np.pi - 2 * np.pi / numTx, numTx)
    A = np.reshape(A, (numTx, 1))
    phi = np.reshape(phi, (numTx, 1))
    transmitters = np.concatenate((A, phi), axis=1)

    domainNx = 50
    domainNy = 50

    backgroundPerm = 1
    targetPerm = 3.7
    epsBackground = backgroundPerm * np.ones((domainNx, domainNy))

    CC = 299792458  # speed of light in vacuum, M/s
    mu0 = np.pi * 4e-7  # permeability
    epsilon0 = 1 / mu0 / CC / CC  # permitivity
    lambdaa = CC / np.sqrt(backgroundPerm.real) / f
    xMax = 0.12
    xMin = -0.12
    yMax = xMax
    yMin = xMin

    cellXWidth = (xMax - xMin) / domainNx
    print(lambdaa / cellXWidth)
    cellYWidth = (yMax - yMin) / domainNy
    xCentroids = np.linspace(xMin + cellXWidth / 2, xMax - cellXWidth / 2, domainNx)
    yCentroids = np.linspace(yMin + cellYWidth / 2, yMax - cellYWidth / 2, domainNy)
    xCentroids = np.reshape(xCentroids, (domainNx, 1))
    yCentroids = np.reshape(yCentroids, (domainNy, 1))

    X, Y = np.meshgrid(xCentroids, yCentroids)
    obsRadius = 0.144  # radius
    obsAngle = np.linspace(0, (2 * np.pi - (2 * np.pi / numRx)),
                        numRx)  # radian-based positions of the receiver points.
    obsAngle = np.reshape(obsAngle, (numRx, 1))

    xObs = obsRadius * np.cos(obsAngle)
    yObs = obsRadius * np.sin(obsAngle)

    ObsPoints = np.concatenate((xObs, yObs), axis=1)

    solver = RichmondSolver(f, epsBackground, X, Y, domainNx, domainNy, ObsPoints, transmitters)

    for i in range(nn):
        count += 1
        print("Target Num " + str(count))
        print(i)

        # squareSide = np.random.uniform(0.04, 0.16)
        squareSide = 0.10
        squareCenter = np.random.uniform(low=[xMin + squareSide / 2, yMin + squareSide / 2],
                                        high=[xMax - squareSide / 2, yMax - squareSide / 2])
        target = np.zeros((domainNx, domainNy))
        for i in range(domainNx):
            for j in range(domainNy):
                if (
                    abs(X[i, j] - squareCenter[0]) < squareSide / 2
                    and abs(Y[i, j] - squareCenter[1]) < squareSide / 2
                ):
                    target[i, j] = 1

        target = target * targetPerm + epsBackground * (1 - target)

        solver.simulate(target, show=False, path="./SqrDifFreq/")
