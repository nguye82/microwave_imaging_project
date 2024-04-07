from ReadData import *
import matplotlib.pyplot as plt
import numpy as np
import os

data = getEMData("./EMNIST_img/")
# data = getEMData("./TriangleDifFreq/")

# folder_path = "./FreqInRangeImg/"

for i in range(0,10):
    target = data[i].target

    field = data[i].Esct

    # print(np.max(target))
    plt.matshow(field.real)
    plt.matshow(target)
    # file_name = 'target_pic_emnist_rand_perm.png'
    # file_path = os.path.join(folder_path, file_name)
    # plt.savefig(file_name)

    # plt.savefig("./FreqInRangeImg/"+file_name)
    plt.show()

# print(data[1].freq)  