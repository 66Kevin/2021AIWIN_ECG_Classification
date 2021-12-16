# -*- coding: utf-8 -*-
import os, torch
from config import config
import scipy.io as sio
import numpy as np
import glob
import pandas as pd

# random seed
np.random.seed(0)


def split_train_data(train_path, val_ratio=0.2):
    '''
    注意：这里的train和val是从train数据集中分开的，用于比赛测试模型准确率的数据集也叫val注意不要混淆！！！
    还没写用于比赛测试模型准确率的数据集val的加载函数
    return: train set and val set
    '''
    data = glob.glob(train_path)
    data.sort()

    val_set_size = int(len(data)*val_ratio) # 320
    train_set_size = int(len(data)-val_set_size) # 1280
    val_mat = data[train_set_size:] # [1280-1600]
    train_mat = data[:train_set_size] # [0-1280]

    # 加载train的数据
    train_mat = [sio.loadmat(x)['ecgdata'].reshape(12, 5000) for x in train_mat]
    val_mat = [sio.loadmat(x)['ecgdata'].reshape(12, 5000) for x in val_mat]

    return train_mat, val_mat


def get_label(label_path):

    train_label = pd.read_csv(label_path)
    train_label['tag'] = train_label['tag'].astype(np.float32)

    return train_label['tag']


def get_test_data(test_path):

    test_path = glob.glob(test_path)
    test_path = sorted(test_path)

    test_name = [i.split('/')[-1].split('.')[0] for i in test_path]
    test_name = sorted(test_name)

    test_mat = [sio.loadmat(x)['ecgdata'].reshape(12, 5000) for x in test_path]

    return test_mat, test_name


def train():
    train, val = split_train_data(config.train_dir)
    label = get_label(config.train_label)
    test, test_name = get_test_data(config.test_dir)

    train_data = {'train': train, 'val': val, 'label': label}
    test_data = {'test_data': test, 'test_name': test_name}

    print(train_data)
    print(test_data)
    print(config.train_data)

    torch.save(train_data, config.train_data)
    torch.save(test_data, config.test_data)


if __name__ == '__main__':
    print(config.train_dir)
    print(config.train_label)
    train()

