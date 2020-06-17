import argparse
import torch
import MyFunction
from torch import nn
from torch import optim
from torchvision import datasets,transforms,models
from collections import OrderedDict
import numpy as np
import json


parser = argparse.ArgumentParser(
	description = 'This is to train the network using given data'
	)

parser.add_argument(action = 'store',dest = 'data_dir')

parser.add_argument('--arch', action = 'store',dest = 'model_selected')
parser.add_argument('--learning_rate', action = 'store',dest = 'lr', type = float)
parser.add_argument('--hidden_units', action = 'store',dest = 'hidden_units', type = int)
parser.add_argument('--epochs', action = 'store',dest = 'epochs', type = int)
parser.add_argument('--save_dir', action = 'store',dest = 'save_directory')
# add commend line for users to specify using GPU
parser.add_argument('--gpu', action = 'store_true',dest = 'device', default = False)

result = parser.parse_args()

trainloaders, validloaders, testloaders = MyFunction.load_data(result.data_dir)

# use the GPU only when it is available and the user has specified to use it
device = torch.device("cuda" if torch.cuda.is_available() and result.device else "cpu")

model, optimizer = MyFunction.train_model(result.model_selected, result.hidden_units, result.lr,result.epochs,trainloaders, validloaders, device)

# MyFunction.test_model(model, testloaders, device)

MyFunction.save_model(result.save_directory, model, optimizer)

