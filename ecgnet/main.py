# -*- coding: utf-8 -*-
import torch, time, os, shutil
import models, utils
import numpy as np
import pandas as pd
# from tensorboard_logger import Logger
from torch import nn, optim
from ranger import Ranger
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import ECGDataset
from dataset import ECGTestDataset
from config import config
import os
import warnings
import glob
import logging
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("---------" + str(device) + "------------")

torch.manual_seed(41)
torch.cuda.manual_seed(41)

torch.autograd.set_detect_anomaly(True)


# save current weight and update new best weight
def save_ckpt(state, is_best, model_save_dir):
    current_w = os.path.join(model_save_dir, config.current_w)
    best_w = os.path.join(model_save_dir, config.best_w)
    torch.save(state, current_w)
    if is_best: shutil.copyfile(current_w, best_w)


def train_epoch(model, optimizer, criterion, train_dataloader, show_interval=10):
    print('start train........')
    model.train()
    f1_meter, loss_meter, it_count = 0, 0, 0
    for inputs, target in tqdm(train_dataloader):
        train_data = inputs['train']
        heart_rate = inputs['heart_rate'] // 100
        inputs = train_data.to(device)
        heart_rate = heart_rate.to(device)
        # inputs = inputs.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        # forward
        output = model(inputs, heart_rate)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_meter += loss.item()
        it_count += 1
        # print(torch.sigmoid(output))
        f1 = utils.calc_f1(target, torch.sigmoid(output))
        f1_meter += f1
        if it_count != 0 and it_count % show_interval == 0:
            print("%d,loss:%.3e f1:%.3f" % (it_count, loss.item(), f1))
    return loss_meter / it_count, f1_meter / it_count


def val_epoch(model, criterion, val_dataloader, threshold=0.5):
    print('start val........')
    model.eval()
    f1_meter, acc_meter, recall_meter, precision_meter, loss_meter, it_count = 0, 0, 0, 0, 0, 0
    with torch.no_grad():
        for inputs, target in tqdm(val_dataloader):
            train_data = inputs['train']
            heart_rate = inputs['heart_rate'] // 100
            inputs = train_data.to(device)
            heart_rate = heart_rate.to(device)
            # inputs = inputs.to(device)
            target = target.to(device)
            output = model(inputs, heart_rate)
            loss = criterion(output, target)
            loss_meter += loss.item()
            it_count += 1
            output = torch.sigmoid(output)
            f1 = utils.calc_f1(target, output, threshold)
            acc = utils.cal_accuracy_score(target, output)
            recall = utils.cal_recall_score(target, output)
            precision = utils.cal_percision_score(target, output)
            f1_meter += f1
            acc_meter += acc
            recall_meter += recall
            precision_meter += precision
    return loss_meter / it_count, f1_meter / it_count


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def train(args):
    # get model
    model = getattr(models, config.model_name)()
    print("model name: "+ str(config.model_name))
    if args.ckpt and not args.resume:
        state = torch.load(args.ckpt, map_location='cpu')
        model.load_state_dict(state['state_dict'])
        print('train with pretrained weight val_f1', state['f1'])
    model = model.to(device)
    # data
    print('finished model loading')
    train_dataset = ECGDataset(config.train_data, train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_dataset = ECGDataset(config.train_data, train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=0)
    print("train_datasize", len(train_dataset), "val_datasize", len(val_dataset))
    # optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    # optimizer = Ranger(model.parameters(), lr=config.lr)
    # criterion = utils.FocalLoss()
    criterion = utils.BinaryLabel()
    # model save dir
    model_save_dir = '%s/%s_%s' % (config.ckpt, config.model_name, time.strftime("%Y%m%d%H%M"))
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    if args.ex: model_save_dir += args.ex
    best_f1 = -1
    lr = config.lr
    start_epoch = 1
    stage = 1
    # train from last save point
    if args.resume:
        if os.path.exists(args.ckpt):  # weight path
            model_save_dir = args.ckpt
            current_w = torch.load(os.path.join(args.ckpt, config.current_w))
            best_w = torch.load(os.path.join(model_save_dir, config.best_w))
            best_f1 = best_w['loss']
            start_epoch = current_w['epoch'] + 1
            lr = current_w['lr']
            stage = current_w['stage']
            model.load_state_dict(current_w['state_dict'])
            if start_epoch - 1 in config.stage_epoch:
                stage += 1
                lr /= config.lr_decay
                utils.adjust_learning_rate(optimizer, lr)
                model.load_state_dict(best_w['state_dict'])
            print("=> loaded checkpoint (epoch {})".format(start_epoch - 1))

    # =========>start training<=========
    start_epoch = 1
    logger = get_logger('/data/home/scv1442/run/AIWIN/se-ecgnet/train1.log')
    logger.info('start training!')
    train_losses=[]
    val_losses=[]

    for epoch in range(start_epoch, config.max_epoch + 1):
        since = time.time()
        train_loss, train_f1 = train_epoch(model, optimizer, criterion, train_dataloader, show_interval=100)
        val_loss, val_f1 = val_epoch(model, criterion, val_dataloader)
        print('#epoch:%02d stage:%d train_loss:%.3e train_f1:%.3f  val_loss:%0.3e val_f1:%.3f time:%s\n'
              % (epoch, stage, train_loss, train_f1, val_loss, val_f1, utils.print_time_cost(since)))
        logger.info('Epoch:[{}/{}]\t train_loss={:.5f}\t val_loss={:.5f}\t train_f1={:.3f}\t val_f1={:.3f}'.format(
                        epoch, config.max_epoch, train_loss, val_loss, train_f1, val_f1))
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        state = {"state_dict": model.state_dict(), "epoch": epoch, "loss": val_loss, 'f1': val_f1, 'lr': lr,
                 'stage': stage}

        # Save model and early stopping
        if(best_f1 < val_f1):
            patience=0
            save_ckpt(state, True, model_save_dir)
        else:
            patience+=1
            print("Early Stop Counter: {} of 40".format(patience))
            if(patience>40):
                print("Early stopping ......")
                break
        best_f1 = max(best_f1, val_f1)
        print("best F1 score: "+ str(best_f1))

        # Learning rate adjustment policy
        if epoch in config.stage_epoch:
            stage += 1
            logger.info('Enter Stage: {}...'.format(stage))
            lr /= config.lr_decay
            best_w = os.path.join(model_save_dir, config.best_w)
            model.load_state_dict(torch.load(best_w)['state_dict'])
            print("*" * 10, "step into stage%02d lr %.3ef" % (stage, lr))
            utils.adjust_learning_rate(optimizer, lr)

    # Finish Training
    print(train_losses)
    print(val_losses)
    logger.info('finish training!')



def train_full(args):
    # get model
    model = getattr(models, config.model_name)()
    if args.ckpt and not args.resume:
        state = torch.load(args.ckpt, map_location='cpu')
        model.load_state_dict(state['state_dict'])
        print('train with pretrained weight val_f1', state['f1'])
    model = model.to(device)
    # data
    print('finished model loading')
    train_dataset = ECGDataset(config.train_full_data, train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    print("train_datasize", len(train_dataset))
    # optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = utils.BinaryLabel()
    # model save dir
    model_save_dir = '%s/%s_%s' % (config.ckpt, config.model_name, time.strftime("%Y%m%d%H%M"))
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    if args.ex: model_save_dir += args.ex
    best_f1 = -1
    lr = config.lr
    stage = 1
    # =========>start training<=========
    start_epoch = 1
    logger = get_logger('train_full_data.log')
    logger.info('start training!')
    for epoch in range(start_epoch, config.max_epoch + 1):
        since = time.time()
        train_loss, train_f1 = train_epoch(model, optimizer, criterion, train_dataloader, show_interval=100)

        print('#epoch:%02d stage:%d train_loss:%.3e train_f1:%.3f time:%s\n'
              % (epoch, stage, train_loss, train_f1, utils.print_time_cost(since)))
        logger.info('Epoch:[{}/{}]\t train_loss={:.5f}\t train_f1={:.3f}'.format(
                        epoch, config.max_epoch, train_loss, train_f1))

        state = {"state_dict": model.state_dict(), "epoch": epoch, 'lr': lr, 'stage': stage}
        # Save model and early stopping
        if(best_f1 < train_f1):
            patience=0
            save_ckpt(state, True, model_save_dir)
        else:
            patience+=1
            print("Early Stop Counter: {} of 15".format(patience))
            if(patience>15):
                print("Early stopping ......")
                break
        best_f1 = max(best_f1, train_f1)
        print("best F1 score: "+ str(best_f1))

        if epoch in config.stage_epoch:
            stage += 1
            lr /= config.lr_decay
            best_w = os.path.join(model_save_dir, config.best_w)
            model.load_state_dict(torch.load(best_w)['state_dict'])
            print("*" * 10, "step into stage%02d lr %.3ef" % (stage, lr))
            utils.adjust_learning_rate(optimizer, lr)
    logger.info('finish training!')


# test
def val(args):

    test_path = glob.glob(config.test_dir)
    test_path = [i.split('/')[-1].split('.')[0] for i in test_path]
    test_path = sorted(test_path)

    model = models.ECGNet()
    if args.ckpt:
        print(args.ckpt)
        model_dict = torch.load(args.ckpt, map_location='cpu')
        model.load_state_dict(model_dict['state_dict'])
        print("model parameters has loaded!!")
    model = model.to(device)
    test_dataset = ECGTestDataset(config.test_data)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=1)
    model.eval()
    with torch.no_grad():
        test_pred_dict = dict()
        for x, y in tqdm(test_dataloader):
            test_data = x['test']
            heart_rate = x['heart_rate'] // 100
            x = test_data.to(device)
            heart_rate = heart_rate.to(device)
            x = x.to(device)
            pred = model(x,heart_rate)
            for i in range(len(y)):
                pre_res = (torch.sigmoid(pred[i]).cpu().numpy() > 0.5).astype(int)
                test_pred_dict[y[i]] = pre_res[1]
        print(test_pred_dict)

    test_answer = pd.DataFrame({
            'name': test_path,
            'tag': test_pred_dict.values()
        })

    test_answer.to_csv('answer.csv', index=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("command", metavar="<command>", help="train or infer")
    parser.add_argument("--ckpt", type=str, help="the path of model weight file")
    parser.add_argument("--ex", type=str, help="experience name")
    parser.add_argument("--resume", action='store_true', default=False)
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    if args.command == "test":
        val(args)
    if args.command == "train_full":
        train_full(args)
    
    
