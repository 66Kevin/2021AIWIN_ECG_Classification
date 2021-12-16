# 2021 AIWIN 12-Leads ECG Classification Competition


![](https://img.shields.io/badge/license-MIT-blue)

This repository includes our team solution for 2021 AIWIN 12-Leads ECG Classification Competition Task 1.

## Requirements

All required packages are shown in `requirements.txt` file.

```python
Bottleneck @ file:///tmp/build/80754af9/bottleneck_1607575130224/work
certifi==2021.10.8
joblib @ file:///tmp/build/80754af9/joblib_1635411271373/work
mkl-fft==1.3.1
mkl-random @ file:///tmp/build/80754af9/mkl_random_1626186066731/work
mkl-service==2.4.0
numexpr @ file:///tmp/build/80754af9/numexpr_1618856529730/work
numpy @ file:///tmp/build/80754af9/numpy_and_numpy_base_1634095651905/work
olefile @ file:///Users/ktietz/demo/mc3/conda-bld/olefile_1629805411829/work
pandas==1.3.4
Pillow==8.4.0
python-dateutil @ file:///tmp/build/80754af9/python-dateutil_1626374649649/work
pytorchtools==0.0.2
pytz==2021.3
# Editable install with no version control (ranger==0.1.dev0)
-e /data/run01/scv1442/AIWIN/se-ecgnet/Ranger-Deep-Learning-Optimizer-master
scikit-learn @ file:///tmp/build/80754af9/scikit-learn_1635187048948/work
scipy @ file:///tmp/build/80754af9/scipy_1630606796912/work
six @ file:///tmp/build/80754af9/six_1623709665295/work
threadpoolctl @ file:///Users/ktietz/demo/mc3/conda-bld/threadpoolctl_1629802263681/work
torch==1.10.0
torchaudio==0.10.0
torchvision==0.11.1
tqdm @ file:///tmp/build/80754af9/tqdm_1635330843403/work
typing-extensions @ file:///tmp/build/80754af9/typing_extensions_1631814937681/work
```

## Installation

All dependencies can be installed by running the following commands:

```shell
git clone https://github.com/66Kevin/2021AIWIN_ECG_Classification.git
cd 2021AIWIN_ECG_Classification
pip install --no-deps -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

The implementation was tested on Linux, Python 3.8.5, CUDA11.1.

Training and inference on the CPU is supported but not recommended. The functionality of this repository can not be guaranteed for other system configurations.

## Usage

1. The first thing is modify the `config.py` file with your computer configeration and prefer training configrations.

2. Before train the model, you need to pre-process the dataset (More details about dataset is shown below Data section)

   ```
   cd ECGNet
   python data_process.py
   ```

3. Train the model

   Perform `run.sh` on the GPU cluster or perform `python main.py train `

4. Inference

   `python main.py test`

## Data
