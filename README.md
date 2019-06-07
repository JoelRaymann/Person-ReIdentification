# Person-ReIdentification
A A.I Based model for person re-identification using Xception Network

# Introduction
This is a repo that contains implementation of different A.I Model powered by TF 2.0
The first iteration uses the Xception Network model over the MARS dataset for Re-Identification
problem.

# Pre-requisities
We need a set of few requirements before we can use these code. First, we need to download
the MARS Dataset from the Liangzheng website. You can get it from [here](http://www.liangzheng.com.cn/Project/project_mars.html).
Next, we need the following software and hardware requirements
## Software Requirements:
### For Windows - 7, 8, 8.1, 10 - Recommended OS
 - Python 3.5.x or greater or Python 2.7.x or greater but python 2.7.x series not recommended
 - Nvidia Drivers Updated (Only needed if you are training the model from scratch)
 - CUDA - 10.0 (Only needed if you are training the model from scratch)
 - CuDNN - 7.x (Only needed if you are training the model from scratch)

### For Linux - Ubuntu - 16, 18; Fedora; OpenSUSE; RHEL
 - Python 3.5.x or greater or Python 2.7.x or greater but python 2.7.x series not recommended
 - Nvidia Drivers Updated (Only needed if you are training the model from scratch)
 - CUDA - 10.0 (Only needed if you are training the model from scratch)
 - CuDNN - 7.x (Only needed if you are training the model from scratch)

### For MacOS - Sierra, High Sierra, Mojave - Training is not supported ;_;
 - Python 3.5.x or greater or Python 2.7.x or greater but python 2.7.x series not recommended

## Hardware Requirements:
 - Atleast any core with atleast 4 logical cores
 - Atleast 8 GB RAM 
 - Nvidia GPU Card needed for training

## Python Packages Requirements
Next, we need to get all the python packages. For that, we have prepared two requirements.txt file.
### For CPU Only
In order to run the program in CPU alone system. Use the following command to 
install all the packages
```bash
pip install -r requirements-only-cpu.txt
```
### For Nvidia GPU
In order to run the program in GPU included system.Use the following command to 
install all the packages
```bash
pip install -r requirements-gpu.txt
```

# How to execute
## Step 1: Get the Dataset
 - As mentioned, Get the dataset from the official website from [here](http://www.liangzheng.com.cn/Project/project_mars.html)
 - pull this git repo and extract the MARS dataset in a new folder called
 `dataset`. The final extracted path should be like `./dataset/bbox_train/`  for `bbox_train.zip` and `./dataset/bbox_test/` for `bbox_test.zip`

## Step 2: Clean the Dataset
- First, make a new folder in the `./dataset/` folder such as `totat_data` in such a way we get a final path as `./dataset/total_data/`
- Second, copy everything from `./dataset/bbox_train/` and `./dataset/bbox_test/` into `./dataset/total_data/` to make a big combined
data
- Now, we will delete sample `0000` and sample `00-1` from `./dataset/total_data/` folder

## Step 3: Make Hard - Pairs for training
- Now, we will make .csv files for our training. To do this, just run the following python code in the terminal like:
```bash
python make_data.py -I <input-dataset-path> -O <csv-save-path> -C <max count of dataset to generate> -S <checkpoint saves if needed>
```
A sample example would be like: 
```bash
python make_data.py -I "./dataset/total_data/" -O "./dataset/total_data.csv" -C 2000000 -S 100000
```
- Now, make your train, val and test .csv splits. it is advisable to keep your .csv files name as `train_data.csv`, `val_data.csv` and `test_data.csv`

## Step 4: Training Model
- To train the model, we need to configure the config.yaml file. 
Set your own configuration in it. Set your batch_size and all there.
- Once set. You have 2 options.
### Option 1: Fresh Train.
- This will train from scratch. To do this. Open your terminal and give
a python call to train_model.py like
```bash
python train_model.py -C config.yaml
```
- This will automatically train and save the weights in "./models/" folder
We can then use the weights for the testing
### Option 2: Re-train Model / Resume Training
- This will take the models weights and retrain it from a particular epoch.
To do this, give a python call in your terminal like
```bash
python train_model.py -L <load model file path> -R <resume epoch> -W 1 -C <config yaml file path>
```
- This will give the new weights in the ```./models/saved_model_<resume_epoch>/``` folder

## Step 4: Testing Model
Coming Soon.

# Results:
Coming Soon.

# LICENSE
BSD 3-Clause License

Copyright (c) 2019, Joel Raymann
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# REFERENCE AND CITATIONS
 - zheng2016mars, title={MARS: A Video Benchmark for Large-Scale Person Re-identification}, author = {Zheng, Liang and Bie, Zhi and Sun, Yifan and Wang Jingdong and Su, Chi and Wang, Shengjin and Tian, Qi}, year={2016},

# THANK YOU
Wait for more updates. Further Updates in progress
 - Results
 - Portability to Conversion to .tflite for all platform portabality
 - Andriod App that runs this model
