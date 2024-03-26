from ReadData import *
import matplotlib.pyplot as plt
import numpy as np
import os

data = getEMData("./MNIST_UNET_24/")
# data = getEMData("./TriangleDifFreq/")



# print(data)
# print(len(data))
# data = data[0]
# folder_path = "./FreqInRangeImg/"

for i in range(0,2):
    target = data[i].target

    field = data[i].Esct

    # print(np.max(target))
    plt.matshow(field.real)
    plt.matshow(target)
    # file_name = 'target_pic_sqr.png'
    # file_path = os.path.join(folder_path, file_name)
    # plt.savefig(file_name)

    # plt.savefig("./FreqInRangeImg/"+file_name)
    plt.show()

# print(data[1].freq)  