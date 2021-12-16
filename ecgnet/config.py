# -*- coding: utf-8 -*-
import os


class Config:
    # path to your root dir
    root = '/data/home/scv1442/run/AIWIN'
    # path to training data dir
    train_dir = os.path.join(root, 'data/train/*.mat')
    # path to test data dir
    test_dir = os.path.join(root, 'data/val/*.mat')
    # path to train_label file
    train_label = os.path.join(root, 'data/trainreference_new.csv')

    train_data = os.path.join(root, 'train_2200.pth')
    train_full_data = os.path.join(root, 'train_2200_full.pth')
    test_data = os.path.join(root, 'test.pth')
    
    # for train
    # select train model SE_ECGNet ECGNet BiRCNN ResNet101
    model_name = 'ECGNet'
    # learning rate decay method
    stage_epoch = [32, 64, 128, 256]
    # batch_size
    batch_size = 64
    # number of labels
    num_classes = 2
    # max_epoch
    max_epoch = 256
    # resampling points(default 2048)
    target_point_num = 2048
    # model save dir
    ckpt = '/data/home/scv1442/run/AIWIN/ckpt'
    model_path = '/data/home/scv1442/run/AIWIN/ckpt/SE_ECGNet_202111260029/best_w.pth'
    # learning rate
    lr = 1e-3
    # saved current weight path
    current_w = 'current_w.pth'
    # saved best weight path
    best_w = 'best_w.pth'
    # learning rate decay lr/=lr_decay
    lr_decay = 10


    # Configuration of augmentation pipeline
    AUGMENTATION_PIPELINE_CONFIG_2C = {
        "p_scale": 0.4,
        "p_drop": 0.4,
        "p_cutout": 0.4,
        "p_shift": 0.4,
        "p_resample": 0.4,
        "p_random_resample": 0.4,
        "p_sine": 0.4,
        "p_band_pass_filter": 0.4,
        "scale_range": (0.85, 1.15),
        "drop_rate": 0.03,
        "interval_length": 0.05,
        "max_shift": 4000,
        "resample_factors": (0.8, 1.2),
        "max_offset": 0.075,
        "resampling_points": 12,
        "max_sine_magnitude": 0.3,
        "sine_frequency_range": (.2, 1.),
        "kernel": (1, 6, 15, 20, 15, 6, 1),
        "ecg_sequence_length": 18000,
        "fs": 500,
        "frequencies": (0.2, 45.)
    }

config = Config()
