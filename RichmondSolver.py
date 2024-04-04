import pickle
import random

import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import scipy
import numpy.linalg as lin
from scipy import io as sio
import pandas
import uuid


class RichmondSolver:
    # may even want to refactor this so that ObsPoints and Transmitters gets passed in during the sim since we dont
    # need it to construct G but we may need it for constructing the outer G
    def __init__(self, freq, epsBackground, X, Y, nx, ny, ObsPoints, Transmitters):
        assert freq > 0
        assert nx > 0
        assert ny > 0
        # put in these asserts bc apparently its good practice but its probably
        # not neccessary
        # lets cache all these values we're given
        self.freq = freq
        #  self.epsTarget = epsTarget dont need this line now bc eps target will get passed in now
        self.epsBackground = epsBackground
        self.X = X
        self.Y = Y
        self.nx = nx
        self.ny = ny
        self.ObsPoints = ObsPoints
        self.Transmitters = Transmitters

        self.data = {}

        # start setting up a few pre reqs
        self.numTx = Transmitters.shape[0]
        self.N = nx * ny
        self.numRx = ObsPoints.shape[0]
        self.waveSpeed = 299792458  # assume its the speed of light in a vacuum
        self.mu0 = np.pi * 4e-7  # permeability
        self.epsilon0 = 1 / self.mu0 / self.waveSpeed / self.waveSpeed  # permitivity
        self.omega = 2 * np.pi * freq  # angular freq
        self.kb = self.omega * np.sqrt(self.mu0 * self.epsBackground * self.epsilon0)
        self.kb = self.kb.flatten('F')  # turn this thing into an array
        self.kb = np.reshape(self.kb, (self.N, 1))  # turn it into an Nx1 vector

        self.sampleLimit = 10
        self.sampleNum = 0
        self.dimension = max(max(max(max(self.nx, self.ny), self.numRx), self.numTx), 32)
        self.betterData = np.zeros((self.dimension, self.dimension, self.sampleLimit * 2))

        self.kaiData = np.zeros((self.nx, self.ny, self.sampleLimit))
        self.EsctData = np.zeros((self.numRx, self.numTx, self.sampleLimit), dtype=complex)

        self.dx = np.abs(self.X[0][1] - self.X[0][0])
        self.dy = np.abs(self.Y[0][0] - self.Y[1][0])
        assert self.dx == self.dy
        self.cellArea = self.dx * self.dy
        self.a = np.sqrt(self.cellArea / np.pi)

        # X and Y are 2 dims right now but we need them to be one dim
        self.X = self.X.flatten('F')
        self.X = np.reshape(self.X, (self.N, 1))

        self.Y = self.Y.flatten('F')
        self.Y = np.reshape(self.Y, (self.N, 1))

        # now X and Y positions are both column vectors of length N
        self.xMat = np.column_stack([self.X] * self.N)
        self.yMat = np.column_stack([self.Y] * self.N)
        # now its an NxN mat 
        self.rho = np.sqrt((self.xMat - self.xMat.T) ** 2 + (self.yMat - self.yMat.T) ** 2)

        '''
        Dont need this stuff anymore either
        self.epsTarget = self.epsTarget.flatten('F')
        self.epsTarget = np.reshape(self.epsTarget, (self.N, 1))
        self.epsTargetMat = np.column_stack([self.epsTarget] * self.N)
        '''

        # heres the big matrix generation that will be used in C*Etot = Einc but we dont do this here anymore
        '''
        self.C = (1j * np.pi * self.kb * self.a / 2) * (self.epsTargetMat.T - 1) * scipy.special.jv(1, self.kb * self.a)
        #                                                  ^ this is the kai term i think but idk why theres a -1
        self.C = self.C * scipy.special.hankel2(0, self.kb * self.rho)
        for i in range(self.N):
            self.C[i][i] = 1 + (self.epsTargetMat[i][i].T - 1) * (1j / 2) * (
                        np.pi * self.kb[i] * self.a * scipy.special.hankel2(1, self.kb[i] * self.a) - 2 * 1j)
        '''

        self.G = -(1j * np.pi * self.kb * self.a / 2) * scipy.special.jv(1, self.kb * self.a)
        self.G = self.G * scipy.special.hankel2(0, self.kb * self.rho)
        for i in range(self.N):
            self.G[i][i] = (-1j / 2) * (
                        np.pi * self.kb[i] * self.a * scipy.special.hankel2(1, self.kb[i] * self.a) - 2j)

        # set up the coordinates of the receivers for determining the scattered field at those points
        self.xObs = self.ObsPoints[:, 0]
        self.xObs = np.reshape(self.xObs, (self.numRx, 1))
        self.yObs = self.ObsPoints[:, 1]
        self.yObs = np.reshape(self.yObs, (self.numRx, 1))

        self.xObsMat = np.column_stack([self.xObs] * self.N)
        self.xInDMat = np.column_stack([self.X] * self.numRx)

        self.yObsMat = np.column_stack([self.yObs] * self.N)
        self.yInDMat = np.column_stack([self.Y] * self.numRx)

        # set up the epsilons for outside of the domain at the receivers
        #  self.epsTargetMaskMat = np.column_stack([self.epsTarget] * self.numRx) again, dont need this line anymore
        self.rhoRx = np.sqrt((self.xObsMat - self.xInDMat.T) ** 2 + (self.yObsMat - self.yInDMat.T) ** 2)
        # now we have the rho for outside of the imaging domain
        self.allEscat = np.zeros(
            (self.numRx, self.numTx), dtype=complex)  # allocate some space for the scattered fields only at the recievers
        '''
        # Build the C to be used for finding total fields at the 
        self.OuterC = -(1j * np.pi * self.kb.T * 0.5) * self.a * (self.epsTargetMaskMat.T - 1) * scipy.special.jv(1,
                                                                                                                  self.kb.T * self.a) * scipy.special.hankel2(
            0, self.kb.T * self.rhoRx)
            
        '''
        self.OuterG = (-1j * np.pi * self.kb.T * 0.5) * self.a * scipy.special.jv(1,
                                                                                 self.kb.T * self.a) * scipy.special.hankel2(
            0, self.kb.T * self.rhoRx)

    def simulate(self, epsTarget, show=False, path = "./RichmondDataToMatchABC/"):
        self.makeC(epsTarget)

        for i in range(self.numTx):
            print("Turning on transmitter " + str(i)) 

            # set up the initial conditions for the simulation based on the transceiver that is on
            #self.EincDomain = self.A * np.exp(-1j * self.kb * (self.X * np.cos(self.fi) + self.Y * np.sin(self.fi)))
            # do a point source now instead
            # since we often work with transceivers lets use the obs points as locations for the transmitter
            xp, yp = self.ObsPoints[i]
            self.EincDomain = self.pointSource(xp, yp, 1)
            self.EtotDomain = np.linalg.solve(self.C, self.EincDomain)

            self.EsctDomain = self.EtotDomain - self.EincDomain
            if show:
                self.show(epsTarget)

            self.EtotDomainMat = np.column_stack([self.EtotDomain] * self.numRx)

            self.EscatRx = self.OuterC * self.EtotDomainMat.T
            self.EscatRx = np.sum(self.EscatRx, axis=1)
            self.EscatRx = np.reshape(self.EscatRx, (self.numRx, 1))
            self.EscatRx = np.reshape(self.EscatRx, self.numRx)

            self.allEscat[:, i] = self.EscatRx

        if show:
            # show the obs data matrix
            fig, ax = plt.subplots(2, 2)
            # show the real and imag in the top row then the abs and phase in the bottom row
            ax[0, 0].matshow(np.transpose(np.real(self.allEscat)))
            ax[0, 0].set_title("Real")
            ax[0, 1].matshow(np.transpose(np.imag(self.allEscat)))
            ax[0, 1].set_title("Imag")
            ax[1, 0].matshow(np.transpose(np.abs(self.allEscat)))
            ax[1, 0].set_title("Abs")
            ax[1, 1].matshow(np.transpose(np.angle(self.allEscat)))
            ax[1, 1].set_title("Phase")
            plt.show()


        self.betterStore(self.kai, self.allEscat, path)

    def show(self, epsTarget):
        fig, ((ax1, ax2, ax3, ax4)) = plt.subplots(1, 4)
        ax1.matshow(epsTarget.reshape(self.nx, self.ny).real)
        ax1.set_title("Target")
        ax2.matshow(np.transpose(np.abs(self.EincDomain.reshape(self.nx, self.ny))))
        ax2.set_title("Incident Field")
        #ax2.plot(np.real(self.EincDomain))
        ax3.matshow(np.transpose(np.abs(self.EsctDomain.reshape(self.nx, self.ny))))
        ax3.set_title("Scattered Field")
        ax4.matshow(np.transpose(np.abs(self.EtotDomain.reshape(self.nx, self.ny))))
        ax4.set_title("Total Field")
        #x4.plot(np.abs(self.EtotDomain))
        plt.show()

    def makeC(self, epsTargetForKai):
        # holy crap these variable names are awful
        # sorry I made this when I knew so much less about EM.
        # I still kinda know nothing but now at least I know the difference between perm and chi
        if epsTargetForKai.shape[0] != self.N:
            epsTargetForKai = epsTargetForKai.flatten('F')
            epsTargetForKai = np.reshape(epsTargetForKai, (self.N, 1))

        self.kai = epsTargetForKai - np.reshape(self.epsBackground.flatten('F'), (self.N, 1))
        kaiMat = np.zeros((self.N, self.N))
        for i in range(self.N):
            kaiMat[i, i] = self.kai[i]
        self.C = (np.identity(self.N) - np.matmul(self.G, kaiMat))

        self.OuterC = (np.matmul(self.OuterG, kaiMat))
        # self.confirmC(epsTargetForKai)
        print("Done making C for this target")

    def confirmC(self, epsTarget):

        epsTargetMat = np.column_stack([epsTarget] * self.N)
        epsTargetMaskMat = np.column_stack([epsTarget] * self.numRx)
        testC = (1j * np.pi * self.kb * self.a / 2) * (epsTargetMat.T - 1) * scipy.special.jv(1, self.kb * self.a)
        #                                                  ^ this is the kai term i think but idk why theres a -1
        testC = testC * scipy.special.hankel2(0, self.kb * self.rho)
        for i in range(self.N):
            testC[i][i] = 1 + (epsTargetMat[i][i].T - 1) * (1j / 2) * (
                    np.pi * self.kb[i] * self.a * scipy.special.hankel2(1, self.kb[i] * self.a) - 2 * 1j)
        # Build the C to be used for finding total fields at the

        testOuterC = -(1j * np.pi * self.kb.T * 0.5) * self.a * (epsTargetMaskMat.T - 1) * scipy.special.jv(1,
                                                                                                            self.kb.T * self.a) * scipy.special.hankel2(
            0, self.kb.T * self.rhoRx)

        assert (np.round(testOuterC, decimals=12) == np.round(self.OuterC, decimals=12)).all()
        assert (np.round(testC, decimals=12) == np.round(self.C, decimals=12)).all()

    def betterStore(self, kai, allScat, path):
        kai = kai.reshape((self.nx, self.ny))

        self.kaiData[:, :, self.sampleNum] = kai
        self.EsctData[:, :, self.sampleNum] = allScat
        # self.freq = freq

        self.sampleNum += 1

        if self.sampleNum >= self.sampleLimit:
            # self.writeData("./RichmondDataToMatchABC/")
            self.writeData(path)
            self.sampleNum = 0

    def writeData(self, path):
        name = uuid.uuid4()
        fullpath = path + str(name)
        with open(fullpath, 'wb') as fd:
            pickle.dump(self.kaiData, fd)
            pickle.dump(self.EsctData, fd)
            pickle.dump(self.freq, fd)

        self.EsctData = np.zeros((self.numRx, self.numTx, self.sampleNum))
        self.kaiData = np.zeros((self.nx, self.ny, self.sampleNum))
        self.sampleNum = 0

    def pointSource(self, xp, yp, Jz):
        omega = 2 * np.pi * self.freq
        rx = self.X - xp
        ry = self.Y - yp
        r = np.sqrt(np.multiply(rx, rx) + np.multiply(ry, ry))

        # note eps and mu are not assumed relative but are absolute herein.
        k = omega * np.sqrt(self.epsilon0 * self.mu0)

        Ez = -Jz * complex(0, 1 / 4) * scipy.special.hankel2(0, k * r)
        # min should be (-0.07253342353970492-0.15416053245918554j)
        # max should be (0.8186654448570906-0.24999731683541274j)
        return Ez

if __name__ == "__main__":
    # mat = sio.loadmat('./matlab/emnist-letters.mat')
    mat = sio.loadmat('./emnist-letters.mat')
    data = mat['dataset']

    X_train = data['train'][0, 0]['images'][0, 0]
    y_train = data['train'][0, 0]['labels'][0, 0]
    X_test = data['test'][0, 0]['images'][0, 0]
    y_test = data['test'][0, 0]['labels'][0, 0]

    X_train = X_train.reshape((X_train.shape[0], 28, 28))
    X_test = X_test.reshape((X_test.shape[0], 28, 28))

    # randomPerm = np.linspace(1, 3, 10000)
    # np.random.seed()
    # np.random.shuffle(randomPerm)

    nn = 620  # no. of data that you want to generate
    count = random.randint(0, X_train.shape[0]-nn)
    # lets set up the forward solver
    freq = 1e9
    numRx = 24
    noOfTransmitters = 24

    A = np.ones(noOfTransmitters)
    fi = np.linspace(0, 2 * np.pi - 2 * np.pi / noOfTransmitters, noOfTransmitters)
    A = np.reshape(A, (noOfTransmitters, 1))
    fi = np.reshape(fi, (noOfTransmitters, 1))
    Transmitters = np.concatenate((A, fi), axis=1)

    targetNx = X_train[count].shape[0]
    targetNy = X_train[count].shape[1]
    domainNx = 28 #35
    domainNy = 28
    backgroundPerm = 1
    epsBackground = np.ones((domainNy, domainNx)) * backgroundPerm


    CC = 299792458  # speed of light in vacuum, M/s
    mu0 = np.pi * 4e-7  # permeability
    epsilon0 = 1 / mu0 / CC / CC  # permitivity
    lambdaa = CC / np.sqrt(backgroundPerm.real) / freq
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
    xTargetMax = 0.15 #0.075
    xTargetMin = -0.15
    yTargetMax = xTargetMax
    yTargetMin = xTargetMin

    cellXWidthTarg = (xTargetMax - xTargetMin) / domainNx
    cellYWidthTarg = (yMax - yMin) / domainNy
    xCentroidsTarg = np.linspace(xTargetMin + cellXWidthTarg / 2, xTargetMax - cellXWidthTarg / 2, targetNx)
    yCentroidsTarg = np.linspace(yTargetMin + cellYWidthTarg / 2, yTargetMax - cellYWidthTarg / 2, targetNy)
    xCentroidsTarg = np.reshape(xCentroidsTarg, (targetNx, 1))
    yCentroidsTarg = np.reshape(yCentroidsTarg, (targetNy, 1))

    obsRadius = 0.25  # radius 0.15
    obsAngle = np.linspace(0, (2 * np.pi - (2 * np.pi / numRx)),
                           numRx)  # radian-based positions of the receiver points.
    obsAngle = np.reshape(obsAngle, (numRx, 1))

    xObs = obsRadius * np.cos(obsAngle)
    yObs = obsRadius * np.sin(obsAngle)

    ObsPoints = np.concatenate((xObs, yObs), axis=1)

    fwd = RichmondSolver(freq, epsBackground, X, Y, domainNx, domainNy, ObsPoints, Transmitters)

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

        # this part creates images with different colors
        # now lets loop over the list of coords in the target and apply a perm value proportional to how far it is from the peak
        # for coord in coordsWTarget:
        #     diff = np.linalg.norm([peakPermSpot[0] - coord[0], peakPermSpot[1] - coord[1]])

        #     epsTarget[coord[1], coord[0]] = (peakVal-backgroundPerm) * np.exp(-0.01*diff) + backgroundPerm

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
                xIndDomain = xInd #np.argmin(diffX)
                yIndDomain = yInd #np.argmin(diffY)
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
        fwd.simulate(epsTargetOnDomain, show=False, path = "./EMNIST_img/")

    fwd.writeData("./EMNIST_img/")