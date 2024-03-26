from RichmondSolver import RichmondSolver
import numpy as np
import matplotlib.pyplot as plt
import random

nn = 300 
count = random.randint(0, nn)

numRx = 24
numTx = 24

freq = [1e9]
# freq = [1e9, 2e9, 3e9]

for f in freq:
    A = np.ones(numTx)
    phi = np.linspace(0, 2 * np.pi - 2 * np.pi / numTx, numTx)
    A = np.reshape(A, (numTx, 1))
    phi = np.reshape(phi, (numTx, 1))
    transmitters = np.concatenate((A, phi), axis=1)

    domainNx = 48
    domainNy = 48

    backgroundPerm = 1
    targetPerm = 2 #3.7333
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
    print(lambdaa/cellXWidth)
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
        
        # generate a circle target
        circleRadius = np.random.uniform(0.02, 0.08)  
        # circleRadius = 0.05 
        circleCenter = circleCenter = np.random.uniform(low=[xMin + circleRadius, yMin + circleRadius], high=[xMax - circleRadius, yMax - circleRadius])
        
        target = np.zeros((domainNx, domainNy))
        for i in range(domainNx):
            for j in range(domainNy):
                if np.sqrt((X[i, j] - circleCenter[0]) ** 2 + (Y[i, j] - circleCenter[1]) ** 2) < circleRadius:
                    target[i, j] = 1

        target = target * targetPerm + epsBackground * (1 - target)

        solver.simulate(target, show=False, path = "./CircleData/")