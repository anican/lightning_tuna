# lightning_tuna
Second iteration of my pytorch template for deep learning experiments. 
The main difference here is that I use `pytorch_lightning` to improve the 
clarity and structure of my code

## Installation
First, `git clone` and `cd` into the repository. Then, create a new `conda`
environment using the given yaml file:
```bash
conda env create -f environment.yml
```
### Installing miniconda
Don't have conda? Run
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```
and then run 
```bash
bash Miniconda3-latest-Linux-x86_64.sh
```
## Models
By default, the following models are currently supported:
1. LeNet
2. VGGNet 
3. ResNet
I plan on adding the following models very soon:
1. AlexNet
2. GoogLeNet
3. DenseNet
4. ResNeXt
## Run Experiments!
The default method of running the experiments is simple:
```bash
./main.py
```
### Clearing the Logs and Checkpoints
It can be irritating to have to manually remove checkpoints and logs if you want
to start a training process from scratch. Have no fear! Run
```bash
./reset.sh
```
Warning: this will permanently delete all previous checkpoints and log files.
## Visualization with Tensorboard
```bash
tensorboard --logdir logging/<desired_version>/
```
## TODO:
1. add the tqdm 'progress_bar' option to the training steps (and val and test)
2. add accuracy score metrics (sklearn)
3. save log files and checkpoints with a name that tells you the model being
   used.

