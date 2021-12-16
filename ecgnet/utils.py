# -*- coding: utf-8 -*-

import torch
import numpy as np
import time, os
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from torch import nn
from scipy import signal
import torch.nn.functional as F


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def cal_percision_score(y_true, y_pre, threshold=0.5):
    y_true = y_true.view(-1).cpu().detach().numpy().astype(int)
    y_pre = y_pre.view(-1).cpu().detach().numpy() > threshold
    return precision_score(y_true, y_pre)


def cal_recall_score(y_true, y_pre, threshold=0.5):
    y_true = y_true.view(-1).cpu().detach().numpy().astype(int)
    y_pre = y_pre.view(-1).cpu().detach().numpy() > threshold
    return recall_score(y_true, y_pre)


def cal_accuracy_score(y_true, y_pre, threshold=0.5):
    # y_true = y_true.cpu().view(-1)
    # y_pre =y_pre.cpu().view(-1,5)
    # print(y_pre.shape)
    # print(y_true.shape)
    y_true = y_true.view(-1).cpu().detach().numpy().astype(int)
    y_pre = y_pre.view(-1).cpu().detach().numpy() > threshold
    return accuracy_score(y_pre, y_true)


# calcute score
def calc_f1(y_true, y_pre, threshold=0.5):
    y_true = y_true.view(-1).cpu().detach().numpy().astype(int)
    y_pre = y_pre.view(-1).cpu().detach().numpy() > threshold
    # print(y_pre)
    # print(y_true)
    return f1_score(y_true, y_pre)


def print_time_cost(since):
    time_elapsed = time.time() - since
    return '{:.0f}m{:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60)


# adjust learning rate
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


# Multi label loss
class WeightedMultilabel(nn.Module):
    def __init__(self, weights: torch.Tensor):
        super(WeightedMultilabel, self).__init__()
        self.cerition = nn.BCEWithLogitsLoss(reduction='none')
        self.weights = weights

    def forward(self, outputs, targets):
        loss = self.cerition(outputs, targets)
        return (loss * self.weights).mean()

# Binary label loss
class BinaryLabel(nn.Module):
    def __init__(self):
        super(BinaryLabel, self).__init__()
        self.cerition = nn.BCEWithLogitsLoss(reduction='none')
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, outputs, targets):
        loss = self.cerition(outputs, targets)
        return loss.mean()

# focal loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.sigmoid = nn.Sigmoid()

    def forward(self, y_pred, y_true):
        y_true = y_true.float()
        # print(y_true)
        y_pred = self.sigmoid(y_pred)  # 之前有sigmoid的话记得注释掉这一句
        fl = - self.alpha * y_true * torch.log(y_pred + 1e-8) * ((1.0 - y_pred) ** self.gamma) - (1.0 - self.alpha) * (
                    1.0 - y_true) * torch.log((1.0 - y_pred) + 1e-8) * (y_pred ** self.gamma)
        fl_sum = torch.sum(fl)
        return fl_sum


def pan_Tompkin(EcgVector, FS):
    qrs_c = []  # amplitude of R
    qrs_i = []  # index
    SIG_LEV = 0
    nois_c = []
    nois_i = []
    delay = 0
    skip = 0  # becomes one when a T wave is detected
    not_nois = 0  # it is not noise when not_nois = 1
    selected_RR = []  # Selected RR intervals
    m_selected_RR = 0
    mean_RR = 0
    qrs_i_raw = []
    qrs_amp_raw = []
    ser_back = 0
    test_m = 0
    SIGL_buf = []
    NOISL_buf = []
    THRS_buf = []
    SIGL_buf1 = []
    NOISL_buf1 = []
    THRS_buf1 = []

    f1 = 5  # cuttoff low frequency to get rid of baseline wander
    f2 = 15  # cuttoff frequency to discard high frequency noise
    param1 = f1 * 2 / FS
    param2 = f2 * 2 / FS
    Wn = [param1, param2]  # cutt off based on fsv
    N = 3  # order of 3 less processing
    [a, b] = signal.butter(N, Wn, 'bandpass', analog=False)  # b,a bandpass filtering
    # print(a, " ", b)
    # [a,b] = butter(N,Wn)

    ecg_h = signal.filtfilt(a, b, EcgVector, axis=-1, padtype='odd', padlen=None, method='pad', irlen=None)
    # ecg_h1 = signal.filtfilt(a, b, EcgVector, axis=-1, padtype='odd', padlen=None, method='pad', irlen=None)
    ecg_h = ecg_h / max(abs(ecg_h))  # waiting for process

    h_d = [-1 / 8, -2 / 8, 0, 2 / 8, 1 / 8]  # 1/8*fs
    # ecg_d = conv (ecg_h ,h_d);
    ecg_d = signal.convolve(ecg_h, h_d)
    ecg_d = ecg_d / max(ecg_d)
    delay = delay + 2  # delay of derivative filter 2 samples
    # ecg_s = ecg_d.^2

    ecg_s = np.power(ecg_d, 2)
    # Moving average Y(nt) = (1/N)[x(nT-(N - 1)T)+ x(nT - (N - 2)T)+...+x(nT)]
    # arr = np.ones((1, round(0.150*FS))) / float(0.150*FS)
    ls = [1 / float(0.150 * FS)] * round(0.150 * FS)
    ecg_m = signal.convolve(ecg_s, ls)
    delay = delay + 15;

    # waiting for process
    peak_id, peak_property = signal.find_peaks(ecg_m, height=0, threshold=None, distance=100,
                                               prominence=0, width=None, wlen=None,
                                               rel_height=None, plateau_size=None)

    ecg_mm = ecg_m[0:999]
    THR_SIG = max(ecg_mm) * 1 / 4  # 0.25 of the max amplitude
    THR_NOISE = np.mean(ecg_mm) * 1 / 2  # of the mean signal is considered to be noise
    SIG_LEV = THR_SIG;
    NOISE_LEV = THR_NOISE;

    ecg_hh = ecg_h[0:999]
    THR_SIG1 = max(ecg_hh) * 1 / 4  # 0.25 of the max amplitude
    THR_NOISE1 = np.mean(ecg_hh) * 1 / 2
    SIG_LEV1 = THR_SIG1
    NOISE_LEV1 = THR_NOISE1

    index = len(peak_property['peak_heights'])
    for i in range(index):
        y_i = 0.0
        x_i = 0
        if peak_id[i] - round(0.150 * FS) >= 0 and peak_id[i] <= len(ecg_h) - 1:
            arry = ecg_h[peak_id[i] - round(0.150 * FS):peak_id[i]]
            # y_i = max(ecg_h[peak_id[i] - round(0.150*FS):peak_id[i]])
            y_i = max(arry)
            arrylist = arry.tolist()
            x_i = arrylist.index(y_i)
        else:
            if i == 0:
                arry = ecg_h[0:peak_id[i] - 1]
                y_i = max(arry)
                arrylist = arry.tolist()
                x_i = arrylist.index(y_i)
                ser_back = 1
            elif peak_id[i] - 1 >= len(ecg_h):
                arry = ecg_h[peak_id[i] - round(0.150 * FS) - 1:]
                y_i = max(arry)
                arrylist = arry.tolist()
                x_i = arrylist.index(y_i)
        if len(qrs_c) >= 9:
            diffRR = np.diff(qrs_i[len(qrs_c) - 9:len(qrs_c)])
            mean_RR = np.mean(diffRR)
            comp = qrs_i[len(qrs_i) - 1] - qrs_i[len(qrs_i) - 2]
            if comp <= 0.92 * mean_RR or comp >= 1.16 * mean_RR:
                THR_SIG = 0.5 * (THR_SIG)
                THR_SIG1 = 0.5 * (THR_SIG1)
            else:
                m_selected_RR = mean_RR
            if m_selected_RR:
                test_m = m_selected_RR
            elif mean_RR and m_selected_RR == 0:
                test_m = mean_RR
            else:
                test_m = 0
        if test_m:
            if (peak_id[i] - qrs_i[len(qrs_i) - 1]) >= round(1.66 * test_m):
                arr1 = ecg_m[qrs_i[len(qrs_i) - 1] + round(0.200 * FS) - 1:peak_id[i] - round(0.200 * FS) - 1]
                pks_temp = max(arr1)
                arr1list = arr1.tolist()
                locs_temp = arr1list.index(pks_temp)
                locs_temp = qrs_i[len(qrs_i) - 1] + round(0.200 * FS) + locs_temp - 1
                if pks_temp > THR_NOISE:
                    qrs_c.append(pks_temp)
                    qrs_i.append(locs_temp)
                    if locs_temp <= len(ecg_h):
                        arr2 = ecg_h[locs_temp - round(0.150 * FS) - 1:locs_temp - 1]
                        y_i_t = max(arr2)
                        arrslist = arr2.tolist()
                        x_i_t = arrslist.index(y_i_t)
                    else:
                        arr2 = ecg_h[locs_temp - round(0.150 * FS) - 1:len(ecg_h) - 1]
                        y_i_t = max(arr2)
                        arrslist = arr2.tolist()
                        x_i_t = arrslist.index(y_i_t)
                    if y_i_t > THR_NOISE1:
                        qrs_i_raw.append(locs_temp - round(0.150 * FS) + (x_i_t - 1))
                        qrs_amp_raw.append(y_i_t)
                        SIG_LEV1 = 0.25 * y_i_t + 0.75 * SIG_LEV1
                    not_nois = 1
                    SIG_LEV = 0.25 * pks_temp + 0.75 * SIG_LEV
            else:
                not_nois = 0

        if peak_property['peak_heights'][i] >= THR_SIG:
            if len(qrs_c) >= 3:
                if (peak_id[i] - qrs_i[len(qrs_i) - 1]) <= round(0.3600 * FS):
                    Slope1 = np.mean(np.diff(ecg_m[peak_id[i] - round(0.075 * FS):peak_id[i]]))
                    Slope2 = np.mean(np.diff(ecg_m[qrs_i[len(qrs_i) - 1] - round(0.075 * FS):qrs_i[len(qrs_i) - 1]]))
                    if abs(Slope1) <= abs(0.5 * (Slope2)):
                        nois_c.append(peak_property['peak_heights'][i])
                        nois_i.append(peak_id[i])
                        skip = 1
                        NOISE_LEV1 = 0.125 * y_i + 0.875 * NOISE_LEV1
                        NOISE_LEV = 0.125 * peak_property['peak_heights'][i] + 0.875 * NOISE_LEV
                    else:
                        skip = 0
            if skip == 0:
                qrs_c.append(peak_property['peak_heights'][i])
                qrs_i.append(peak_id[i])
                if y_i >= THR_SIG1:
                    if ser_back:
                        qrs_i_raw.append(x_i)
                    else:
                        qrs_i_raw.append(peak_id[i] - round(0.150 * FS) + (x_i - 1))
                    qrs_amp_raw.append(y_i)
                    SIG_LEV1 = 0.125 * y_i + 0.875 * SIG_LEV1
                SIG_LEV = 0.125 * peak_property['peak_heights'][i] + 0.875 * SIG_LEV
        elif THR_NOISE <= peak_property['peak_heights'][i] and peak_property['peak_heights'][i] < THR_SIG:
            NOISE_LEV1 = 0.125 * y_i + 0.875 * NOISE_LEV1
            NOISE_LEV = 0.125 * peak_property['peak_heights'][i] + 0.875 * NOISE_LEV
        elif peak_property['peak_heights'][i] < THR_NOISE:
            nois_c.append(peak_property['peak_heights'][i])
            nois_i.append(peak_id[i])
            NOISE_LEV1 = 0.125 * y_i + 0.875 * NOISE_LEV1
            NOISE_LEV = 0.125 * peak_property['peak_heights'][i] + 0.875 * NOISE_LEV;
        if NOISE_LEV != 0 or SIG_LEV != 0:
            THR_SIG = NOISE_LEV + 0.25 * (abs(SIG_LEV - NOISE_LEV))
            THR_NOISE = 0.5 * (THR_SIG)
        if NOISE_LEV1 != 0 or SIG_LEV1 != 0:
            THR_SIG1 = NOISE_LEV1 + 0.25 * (abs(SIG_LEV1 - NOISE_LEV1))
            THR_NOISE1 = 0.5 * (THR_SIG1)

        SIGL_buf.append(SIG_LEV)
        NOISL_buf.append(NOISE_LEV)
        THRS_buf.append(THR_SIG)

        SIGL_buf1.append(SIG_LEV1)
        NOISL_buf1.append(NOISE_LEV1)
        THRS_buf1.append(THR_SIG1)

        skip = 0;
        not_nois = 0;
        ser_back = 0;

        # print(qrs_i_raw)
    # print(qrs_amp_raw)

    # heart rate
    heart_rate = len(qrs_i_raw) * 6
    heart_R_mean = np.mean(qrs_amp_raw)
    heart_R_max = np.max(qrs_amp_raw)
    heart_R_min = np.min(qrs_amp_raw)
    heart_R_maxdiff = heart_R_max - heart_R_min
    # square root
    sum = 0.0
    for i in qrs_amp_raw:
        sum += (i - heart_R_mean) * (i - heart_R_mean)
    square_root = np.sqrt(sum)

    return qrs_i_raw, qrs_amp_raw, heart_rate, heart_R_mean, heart_R_max, heart_R_min, heart_R_maxdiff, square_root

