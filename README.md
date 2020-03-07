# LAr Neural Network Filtering
A Python 3 class library for training neural networks for energy reconstruction in the ATLAS LAr-Calorimeter at CERN. Different metrics are monitored during the training. For more details see [my master thesis](https://iktp.tu-dresden.de/IKTP/pub/19/Grubitz_Masterarbeit.pdf) 

## Training Data Sets
Training Data Sets are generated using [Atlas Read-out Upgrade Electronics Simulation (AREUS)](gitlab.cern.ch/AREUS/AREUS/)

## Tensorflow
The code has been upgraded to work with Tensorflow 2.0. However execution with Tensorflow 1.14 should also work.

## Dependencies
tensorflow, numpy, h5py, pandas

## Usage 
`cp examples/Run_NN.py .`  
`python3 Run_NN.py`

