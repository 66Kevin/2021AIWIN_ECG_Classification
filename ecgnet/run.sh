#!/bin/bash

#加载环境，此处加载 anaconda 环境以及通过 anaconda 创建的名为 pytorch 的环境
module load anaconda/2020.11

source activate pytorch

#python 程序运行，需在.py 文件指定调用 GPU，并设置合适的线程数，batch_size 大小等
python main.py train