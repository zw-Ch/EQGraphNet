# EQGraphNet
[![DOI](https://zenodo.org/badge/578110292.svg)](https://zenodo.org/badge/latestdoi/578110292) <br>
EQGraphNet is an graph deep learning model used for earthquake magnitude estimation. <br>
The paper is available in 

## Installation
EQGraphNet is based on [Pytorch](https://pytorch.org/docs/stable/index.html) and [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html)<br>
Firstly please create a virtual environment for yourself<br>
`conda create -n your-env-name python=3.9`<br><br>
Then, there are some Python packages need to be installed<br>
`conda install pytorch torchvision torchaudio cudatoolkit=11.3`<br>
`conda install pyg -c pyg`<br>
`conda install matplotlib`<br>
`conda install h5py==2.10.0`<br>

## Dataset Preparation
The Dataset used in our paper can be downloaded from [https://github.com/smousavi05/STEAD](https://github.com/smousavi05/STEAD). Before running, you should donwload and  store the data file in the folder [dataset](https://github.com/czw1296924847/EQGraphNet/tree/main/dataset) like<br>

![image](https://github.com/czw1296924847/EQGraphNet/blob/main/dataset_structure.png)

## Program Description
### Training and Testing Models
After the preparation of Dataset, you can run the programs in the foloder [run](https://github.com/czw1296924847/EQGraphNet/blob/main/run) to test the performance : <br>
`python run_EQGraphNet.py`
