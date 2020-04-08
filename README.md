# lightning_tuna
Second iteration of my pytorch template for deep learning experiments. 
The main difference here is that I use `pytorch_lightning` to improve the 
clarity and structure of my code

## Installation
First, `git clone` and `cd` into the repository. Then, create a new `conda`
environment using the given yaml file:
```
conda env create -f environment.yml
```
### Installing miniconda
Don't have conda? Run
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```
and then run 
```
bash Miniconda3-latest-Linux-x86_64.sh
```
## Run Experiments!
The default method of running the experiments is simple:
```
./main.py
```
## Visualization with Tensorboard
```
tensorboard --logdir logging/<desired_version>/
```
## TODO:
1. add the tqdm 'progress_bar' option to the training steps (and val and test)
2. add accuracy score metrics (sklearn)

