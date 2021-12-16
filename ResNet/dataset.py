import os
import csv

import cv2
import numpy as np
import paddle

class Dataset(paddle.io.Dataset):
    def __init__(self, dataset_root, transforms=None, mode="train"):
        reader = csv.reader(open(os.path.join(dataset_root, 'trainreference.csv'), 'r'))
        next(reader)
        np.random.seed(1234)
        self.file_list = []
        for l in reader:
            self.file_list += [[os.path.join(dataset_root, 'train', l[0] + '.png'), int(l[1])]]

        np.random.shuffle(self.file_list)
        if mode == "train":
            self.file_list = self.file_list[:1440]
        else:
            self.file_list = self.file_list[1440:]
        self.transforms = transforms



    def __getitem__(self, item):
        im_file, label = self.file_list[item]
        img, _ = self.transforms(im_file)
        return img.astype('float32'), label

    def __len__(self):
        return len(self.file_list)

if __name__ == '__main__':
    dataset = Dataset(dataset_root='dataset')

    for data in dataset:
        pass