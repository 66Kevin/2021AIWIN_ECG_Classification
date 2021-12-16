import glob
import os
import csv

import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import PIL.Image as Image

dataset_root_path = '/data/home/scv1442/run/AIWIN/resnet50/dataset/'

for mode in ['train', 'val']:
    files_list = glob.glob(f'/data/home/scv1442/run/AIWIN/data/{mode}/*.mat')

    for f in files_list:
        cgdata = sio.loadmat(f)['ecgdata']
        img = np.zeros((300 * 4, 500 * 3))
        img[:, :] = 255
        figure = plt.figure()
        plt.axis('off')
        for num in range(12):
            data = cgdata[num] * 100
            row = data.astype('int64')[:500, np.newaxis]
            col = np.array(list(range(500))).astype('int64')[:, np.newaxis]

            row_offset = num // 3
            col_offset = num % 3
            plt.plot(col + col_offset * 500, row + row_offset * 300,)
        if not os.path.exists(f'/data/home/scv1442/run/AIWIN/resnet50/dataset/{mode}'):
            os.makedirs(f'/data/home/scv1442/run/AIWIN/resnet50/dataset/{mode}')
        plt.savefig(os.path.join(dataset_root_path, mode, f"{f.split('/')[-1].split('.')[0]}.png"))
        plt.cla()
        plt.close("all")
