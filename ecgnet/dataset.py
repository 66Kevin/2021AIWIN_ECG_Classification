# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from config import config
from scipy import signal
import scipy as sc
from utils import pan_Tompkin
import torch.nn as nn
from typing import Tuple, Dict, Any

import torch.nn.functional as F
import math
import random



def resample(sig, target_point_num=None):
    sig = signal.resample(sig, target_point_num) if target_point_num else sig
    return sig


def scaling(X, sigma=0.1):
    # print("scaling")
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1, X.shape[1]))
    myNoise = np.matmul(np.ones((X.shape[0], 1)), scalingFactor)
    return X * myNoise


def verflip(sig):
    # print("verflip")
    return sig[::-1, :]


def shift(sig, interval=0.3):
    for col in range(0, sig.shape[1], 10):
        if np.random.randn() > 0.5:
            offset = np.random.choice(np.arange(-interval, interval, 0.1))
            sig[:, col] += offset
    return sig

def transform(sig, train=False):
    # data augmentation
    print('data aug')
    if train:
        if np.random.randn() > 0.5:
            sig = scaling(sig)
        if np.random.randn() > 0.5:
            sig = verflip(sig)
        # if np.random.randn() > 0.5:
        #     sig = shift(sig)
    sig = torch.tensor(sig.copy(), dtype=torch.float)
    return sig

class ECGDataset(Dataset):

    def __init__(self, path, train=True):
        super(ECGDataset, self).__init__()
        dd = torch.load(path)
        self.train = train
        self.data = dd['train'] if train else dd['val']
        # self.labels = dd['label'][:int(1600 * 0.8)] if train else dd['label'][int(1600 * 0.8):]
        # self.labels = dd['label'][:1880] if train else dd['label'][1880:]
        self.labels = dd['label'][:]
        self.labels = pd.DataFrame(self.labels).reset_index(drop=True)
        self.labels = self.labels.values.tolist()
        self.data_aug = AugmentationPipeline(config.AUGMENTATION_PIPELINE_CONFIG_2C)


    def __getitem__(self, index):
        # data
        torch_train_simple_data = self.data[index]
        np_train_simple_data = np.array(torch_train_simple_data)
        _, _, heart_rate, _, _, _, _, _ = pan_Tompkin(np_train_simple_data[1][:], 500)
        tran_data = transform(np_train_simple_data, train=True)
        # tran_data = self.data_aug(torch.tensor(torch_train_simple_data))
        data = {"train" : tran_data, # torch.tensor(torch_train_simple_data,dtype=torch.float)
                "heart_rate" : torch.tensor(heart_rate, dtype=torch.float)}
        # label
        labels = np.zeros(config.num_classes)
        if self.labels[index][0] == 1.0:
            labels[1] = 1.0
        labels = torch.tensor(labels, dtype=torch.float)
        return data, labels

    def __len__(self):
        return len(self.data)


class ECGTestDataset(Dataset):

    def __init__(self, path):
        super(ECGTestDataset, self).__init__()
        test_dict = torch.load(config.test_data)
        self.data = test_dict['test_data']
        self.test_name = test_dict['test_name']

    def __getitem__(self, index):
        torch_test_simple_data = self.data[index]
        np_test_simple_data = np.array(torch_test_simple_data)
        _, _, heart_rate, _, _, _, _, _ = pan_Tompkin(np_test_simple_data[1][:], 500)
        test_data = {'test': torch_test_simple_data, 'heart_rate': torch.tensor(heart_rate, dtype=torch.float)}
        test_name = np.array(self.test_name)
        return test_data, test_name[index]

    def __len__(self):
        return len(self.data)


class AugmentationPipeline(nn.Module):
    """
    This class implements an augmentation pipeline for ecg leads.
    Inspired by: https://arxiv.org/pdf/2009.04398.pdf
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Constructor method
        :param config: (Dict[str, Any]) Config dict
        """
        # Call super constructor
        super(AugmentationPipeline, self).__init__()
        # Save parameters
        self.ecg_sequence_length: int = config["ecg_sequence_length"]
        self.p_scale: float = config["p_scale"]
        self.p_drop: float = config["p_drop"]
        self.p_cutout: float = config["p_cutout"]
        self.p_shift: float = config["p_shift"]
        self.p_resample: float = config["p_resample"]
        self.p_random_resample: float = config["p_random_resample"]
        self.p_sine: float = config["p_sine"]
        self.p_band_pass_filter: float = config["p_band_pass_filter"]
        self.fs: int = config["fs"]
        self.scale_range: Tuple[float, float] = config["scale_range"]
        self.drop_rate = config["drop_rate"]
        self.interval_length: float = config["interval_length"]
        self.max_shift: int = config["max_shift"]
        self.resample_factors: Tuple[float, float] = config["resample_factors"]
        self.max_offset: float = config["max_offset"]
        self.resampling_points: int = config["resampling_points"]
        self.max_sine_magnitude: float = config["max_sine_magnitude"]
        self.sine_frequency_range: Tuple[float, float] = config["sine_frequency_range"]
        self.kernel: Tuple[float, ...] = config["kernel"]
        self.fs: int = config["fs"]
        self.frequencies: Tuple[float, float] = config["frequencies"]

    def scale(self, ecg_lead: torch.Tensor, scale_range: Tuple[float, float] = (0.9, 1.1)) -> torch.Tensor:
        """
        Scale augmentation:  Randomly scaling data
        :param ecg_lead: (torch.Tensor) ECG leads
        :param scale_range: (Tuple[float, float]) Min and max scaling
        :return: (torch.Tensor) ECG lead augmented
        """
        # Get random scalar
        random_scalar = torch.from_numpy(np.random.uniform(low=scale_range[0], high=scale_range[1], size=1)).float()
        # Apply scaling
        ecg_lead = random_scalar * ecg_lead
        ecg_lead = ecg_lead[:5000]
        return ecg_lead

    def drop(self, ecg_lead: torch.Tensor, drop_rate: float = 0.025) -> torch.Tensor:
        """
        Drop augmentation: Randomly missing signal values
        :param ecg_lead: (torch.Tensor) ECG leads
        :param drop_rate: (float) Relative number of samples to be dropped
        :return: (torch.Tensor) ECG lead augmented
        """
        print("drop")
        # Estimate number of sample to be dropped
        num_dropped_samples = int(ecg_lead.shape[-1] * drop_rate)
        # Randomly drop samples
        ecg_lead[..., torch.randperm(ecg_lead.shape[-1])[:max(1, num_dropped_samples)]] = 0.
        ecg_lead = ecg_lead[:5000]
        return ecg_lead

    def cutout(self, ecg_lead: torch.Tensor, interval_length: float = 0.1) -> torch.Tensor:
        """
        Cutout augmentation: Set a random interval signal to 0
        :param ecg_lead: (torch.Tensor) ECG leads
        :param interval_length: (float) Interval lenght to be cut out
        :return: (torch.Tensor) ECG lead augmented
        """
        print("cutout")
        # Estimate interval size
        interval_size = int(ecg_lead.shape[-1] * interval_length)
        # Get random starting index
        index_start = torch.randint(low=0, high=ecg_lead.shape[-1] - interval_size, size=(1,))
        # Apply cut out
        ecg_lead[index_start:index_start + interval_size] = 0.
        ecg_lead = ecg_lead[:5000]
        return ecg_lead

    def shift(self, ecg_lead: torch.Tensor, ecg_sequence_length: int = 18000, max_shift: int = 4000) -> torch.Tensor:
        """
        Shift augmentation: Shifts the signal at random
        :param ecg_lead: (torch.Tensor) ECG leads
        :param ecg_sequence_length: (int) Fixed max length of sequence
        :param max_shift: (int) Max applied shift
        :return: (torch.Tensor) ECG lead augmented
        """
        print("shift")
        # Generate shift
        shift = torch.randint(low=0, high=max_shift, size=(1,))
        # Apply shift
        ecg_lead = torch.cat([torch.zeros_like(ecg_lead)[..., :shift], ecg_lead], dim=-1)[:ecg_sequence_length]
        ecg_lead = ecg_lead[:5000]
        return ecg_lead

    def resample(self, ecg_lead: torch.Tensor, ecg_sequence_length: int = 18000,
                 resample_factors: Tuple[float, float] = (0.8, 1.2)) -> torch.Tensor:
        """
        Resample augmentation: Resamples the ecg lead
        :param ecg_lead: (torch.Tensor) ECG leads
        :param ecg_sequence_length: (int) Fixed max length of sequence
        :param resample_factor: (Tuple[float, float]) Min and max value for resampling
        :return: (torch.Tensor) ECG lead augmented
        """
        # Generate resampling factor
        resample_factor = torch.from_numpy(
            np.random.uniform(low=resample_factors[0], high=resample_factors[1], size=1)).float()
        # Resample ecg lead
        ecg_lead = F.interpolate(ecg_lead[None, None], size=int(resample_factor * ecg_lead.shape[-1]), mode="linear",
                                 align_corners=False)[0, 0]
        # Apply max length if needed
        ecg_lead = ecg_lead[:ecg_sequence_length]
        return ecg_lead

    def random_resample(self, ecg_lead: torch.Tensor, ecg_sequence_length: int = 18000,
                        max_offset: float = 0.03, resampling_points: int = 4) -> torch.Tensor:
        """
        Random resample augmentation: Randomly resamples the signal
        :param ecg_lead: (torch.Tensor) ECG leads
        :param ecg_sequence_length: (int) Fixed max length of sequence
        :param max_offset: (float) Max resampling offsets between 0 and 1
        :param resampling_points: (int) Initial resampling points
        :return: (torch.Tensor) ECG lead augmented
        """
        # Make coordinates for resampling
        coordinates = 2. * (torch.arange(ecg_lead.shape[-1]).float() / (ecg_lead.shape[-1] - 1)) - 1
        # Make offsets
        offsets = F.interpolate(((2 * torch.rand(resampling_points) - 1) * max_offset)[None, None],
                                size=ecg_lead.shape[-1], mode="linear", align_corners=False)[0, 0]
        # Make grid
        grid = torch.stack([coordinates + offsets, coordinates], dim=-1)[None, None].clamp(min=-1, max=1)
        # Apply resampling
        ecg_lead = F.grid_sample(ecg_lead[None, None, None], grid=grid, mode='bilinear', align_corners=False)[0, 0, 0]
        # Apply max lenght if needed
        ecg_lead = ecg_lead[:ecg_sequence_length]
        return ecg_lead

    def sine(self, ecg_lead: torch.Tensor, max_sine_magnitude: float = 0.2,
             sine_frequency_range: Tuple[float, float] = (0.2, 1.), fs: int = 300) -> torch.Tensor:
        """
        Sine augmentation: Add a sine wave to the entire sample
        :param ecg_lead: (torch.Tensor) ECG leads
        :param max_sine_magnitude: (float) Max magnitude of sine to be added
        :param sine_frequency_range: (Tuple[float, float]) Sine frequency rand
        :param fs: (int) Sampling frequency
        :return: (torch.Tensor) ECG lead augmented
        """
        print("sine")
        # Get sine magnitude
        sine_magnitude = torch.from_numpy(np.random.uniform(low=0, high=max_sine_magnitude, size=1)).float()
        # Get sine frequency
        sine_frequency = torch.from_numpy(
            np.random.uniform(low=sine_frequency_range[0], high=sine_frequency_range[1], size=1)).float()
        # Make t vector
        t = torch.arange(ecg_lead.shape[-1]) / float(fs)
        # Make sine vector
        sine = torch.sin(2 * math.pi * sine_frequency * t + torch.rand(1)) * sine_magnitude
        # Apply sine
        ecg_lead = sine + ecg_lead
        ecg_lead = ecg_lead[:5000]
        return ecg_lead

    def band_pass_filter(self, ecg_lead: torch.Tensor, frequencies: Tuple[float, float] = (0.2, 45.),
                         fs: int = 300) -> torch.Tensor:
        """
        Low pass filter: Applies a band pass filter
        :param ecg_lead: (torch.Tensor) ECG leads
        :param frequencies: (Tuple[float, float]) Frequencies of the band pass filter
        :param fs: (int) Sample frequency
        :return: (torch.Tensor) ECG lead augmented
        """
        print("band pass")
        # Init filter
        sos = signal.butter(10, frequencies, 'bandpass', fs=fs, output='sos')
        ecg_lead = torch.from_numpy(signal.sosfilt(sos, ecg_lead.numpy()))
        ecg_lead = ecg_lead[:5000]
        return ecg_lead

    def forward(self, ecg_lead: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applies augmentation to input tensor
        :param ecg_lead: (torch.Tensor) ECG leads
        :return: (torch.Tensor) ECG lead augmented
        """
        # Apply cut out augmentation
        if random.random() <= self.p_cutout:
            ecg_lead = self.cutout(ecg_lead, interval_length=self.interval_length)
        # Apply drop augmentation
        if random.random() <= self.p_drop:
            ecg_lead = self.drop(ecg_lead, drop_rate=self.drop_rate)
        # Apply random resample augmentation
        # if random.random() <= self.p_random_resample:
        #     ecg_lead = self.random_resample(ecg_lead, ecg_sequence_length=self.ecg_sequence_length,
        #                                     max_offset=self.max_offset, resampling_points=self.resampling_points)
        # Apply resample augmentation
        # if random.random() <= self.p_resample:
        #     ecg_lead = self.resample(ecg_lead, ecg_sequence_length=self.ecg_sequence_length,
        #                              resample_factors=self.resample_factors)
        # Apply scale augmentation
        if random.random() <= self.p_scale:
            ecg_lead = self.scale(ecg_lead, scale_range=self.scale_range)
        # Apply shift augmentation
        if random.random() <= self.p_shift:
            ecg_lead = self.shift(ecg_lead, ecg_sequence_length=self.ecg_sequence_length, max_shift=self.max_shift)
        # Apply sine augmentation
        if random.random() <= self.p_sine:
            ecg_lead = self.sine(ecg_lead, max_sine_magnitude=self.max_sine_magnitude,
                                 sine_frequency_range=self.sine_frequency_range, fs=self.fs)
        # Apply low pass filter
        if random.random() <= self.p_band_pass_filter:
            ecg_lead = self.band_pass_filter(ecg_lead, frequencies=self.frequencies, fs=self.fs)
        return ecg_lead


# if __name__ == '__main__':
#     x, target = ECGDataset(config.train_data, train=True)[3]
#     print(x['train'].shape)
#     print(x['heart_rate'])
#     print(target)
#
#     data, name = ECGTestDataset(config.test_data)[1]
#     print(data['test'].shape)
#     print(data['heart_rate'])
#     print(name)