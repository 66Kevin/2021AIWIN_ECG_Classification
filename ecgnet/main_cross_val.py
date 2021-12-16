import scipy.io as sio
import pandas as pd
import numpy as np
import time
# from gensim.models import word2vec
from sklearn import preprocessing
from sklearn import metrics
import os
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.autograd import Variable
import time
import warnings
from models import ECGNet

warnings.filterwarnings("ignore")
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(2021)

path = '/data/home/scv1442/run/AIWIN/data/'
model_save_root = '/data/home/scv1442/run/AIWIN/ckpt'
df = pd.read_csv(path + 'trainreference.csv')


class myDataset(Dataset):
    def __init__(self, df, idx=None, if_train=True):
        self.if_train = if_train
        if self.if_train:
            self.paths = df.loc[idx, 'name'].reset_index(drop=True)
            self.labels = df.loc[idx, 'tag'].reset_index(drop=True)
        else:
            self.paths = df['name'].reset_index(drop=True)
            self.labels = df['tag'].reset_index(drop=True)

    def __getitem__(self, index):
        if self.if_train:
            sample = sio.loadmat(path + 'train/' + self.paths[index])['ecgdata']
        else:
            sample = sio.loadmat(path + 'val/' + self.paths[index])['ecgdata']
        return sample, self.labels[index]

    def __len__(self):
        return len(self.paths)


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
max_epoch = 30
model_save_dir = model_save_root


def train_model(model, criterion, optimizer, lr_scheduler=None):
    total_iters = len(trainloader)
    print('--------------total_iters:{}'.format(total_iters))
    since = time.time()
    best_loss = 1e7
    best_epoch = 0
    best_f1 = 0
    #
    iters = len(trainloader)
    for epoch in range(1, max_epoch + 1):
        model.train()
        begin_time = time.time()
        # print('learning rate:{}'.format(optimizer.param_groups[-1]['lr']))
        print('Fold{} Epoch {}/{}'.format(fold + 1, epoch, max_epoch))
        running_corrects_linear = 0
        count = 0
        train_loss = []
        for i, (inputs, labels) in (enumerate(trainloader)):
            # print(inputs)
            model.train()
            count += 1
            inputs = inputs.to(device)
            labels = labels.float().to(device)

            labels1 = np.zeros((train_batch_size,2))
            for i in range(len(labels)):
                if labels[i] == 1.0:
                    labels1[i][1] = 1.0
            labels1 = torch.tensor(labels1).to(device)
            out_linear = model(inputs).to(device)
            loss = criterion(out_linear, labels1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 更新cosine学习率
            if lr_scheduler != None:
                lr_scheduler.step(epoch + count / iters)
            if print_interval > 0 and (i % print_interval == 0 or out_linear.size()[0] < train_batch_size):
                spend_time = time.time() - begin_time
                print(
                    ' Fold:{} Epoch:{}({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
                        fold + 1, epoch, count, total_iters,
                        loss.item(), optimizer.param_groups[-1]['lr'],
                        spend_time / count * total_iters // 60 - spend_time // 60))
                #
                train_loss.append(loss.item())
                # lr_scheduler.step()
            val_f1, val_loss = val_model(model, criterion)
            print('valf1: {:.4f}  valLogLoss: {:.4f}'.format(val_f1, val_loss))
            model_out_path = model_save_dir + "/" + 'fold_' + str(fold + 1) + '_' + str(epoch) + '.pth'
            best_model_out_path = model_save_dir + "/" + 'fold_' + str(fold + 1) + '_best' + '.pth'
            # save the best model
            if val_f1 >= best_f1:
                best_loss = val_loss
                best_f1 = val_f1
                best_epoch = epoch
                torch.save(model.state_dict(), best_model_out_path)
                print("save best epoch: {} best f1: {:.5f} best logloss: {:.5f}".format(best_epoch, val_f1, val_loss))

        print('Fold{} Best f1: {:.3f} Best logloss: {:.3f} Best epoch:{}'.format(fold + 1, best_f1, best_loss,
                                                                                 best_epoch))
        time_elapsed = time.time() - since
        return best_loss, best_f1


def val_model(model, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    cont = 0
    outPre = []
    outLabel = []
    pres_list = []
    labels_list = []
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            pres_list += outputs.sigmoid().view(-1).detach().cpu().numpy().tolist()
            labels_list += labels.view(-1).detach().cpu().numpy().tolist()

    preds = np.array(pres_list)
    labels = np.array(labels_list)
    val_f1 = metrics.f1_score(labels, list(map(lambda x: 1 if x > 0.5 else 0, preds)))
    log_loss = metrics.log_loss(labels, preds)  #
    return val_f1, log_loss

if __name__ == '__main__':
    setup_seed(2021)
    skf = StratifiedKFold(n_splits=5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_batch_size = 64
    print(device)
    criterion = nn.BCEWithLogitsLoss()
    print_interval = -1
    kfold_best_loss = []
    kfold_best_f1 = []
    # print(len(df))
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['tag'].values)):
        trainloader = torch.utils.data.DataLoader(
            myDataset(df, train_idx),
            batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=1)
        val_loader = torch.utils.data.DataLoader(
            myDataset(df, val_idx),
            batch_size=128, shuffle=False, pin_memory=True, num_workers=1)
        model = ECGNet()
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)

        best_loss, best_acc = train_model(model, criterion, optimizer, lr_scheduler=scheduler)
        kfold_best_loss.append(best_loss)
        kfold_best_f1.append(best_acc)

    print(kfold_best_f1)
    print('loss...', np.mean(kfold_best_loss), 'f1...', np.mean(kfold_best_f1))
