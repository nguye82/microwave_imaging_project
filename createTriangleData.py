from RichmondSolver import RichmondSolver
import numpy as np
import matplotlib.pyplot as plt
import random

nn = 150
count = random.randint(0, nn)

numRx = 24
numTx = 24

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

        # Generate a fixed-size triangle target
        triangleSize = 0.10  # Height of the triangle
        triangleCenter = np.random.uniform(low=[xMin + triangleSize / 2, yMin + triangleSize / 2],
                                           high=[xMax - triangleSize / 2, yMax - triangleSize / 2])

        vertices = np.array([
            [triangleCenter[0] - triangleSize / 2, triangleCenter[1] - triangleSize / (2 * np.sqrt(3))],
            [triangleCenter[0] + triangleSize / 2, triangleCenter[1] - triangleSize / (2 * np.sqrt(3))],
            [triangleCenter[0], triangleCenter[1] + triangleSize / np.sqrt(3)]
        ])
        
        target = np.zeros((domainNx, domainNy))
        for i in range(domainNx):
            for j in range(domainNy):
                # Check if the point is inside the triangle using a point-in-polygon algorithm
                x, y = X[i, j], Y[i, j]
                is_inside = False

                for k in range(3):
                    x1, y1 = vertices[k]
                    x2, y2 = vertices[(k + 1) % 3]

                    if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
                        is_inside = not is_inside

                if is_inside:
                    target[i, j] = targetPerm

        target = target * targetPerm + epsBackground * (1 - target)

        solver.simulate(target, show=False, path="./TriangleDifFreq/")