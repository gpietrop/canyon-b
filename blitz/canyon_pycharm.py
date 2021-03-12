import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from IPython import display
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
from datetime import datetime as dt
import time

def get_n_params(model):
    np=0
    for p in list(model.parameters()):
        np += p.nelement()
    return np

A = 4 / 3
a = 1.7159
def mysigmoid(x):
    return A * torch.sigmoid(x * a / 2)  # A*(np.exp(a*x) -1)/(np.exp(a*x)+1)

class MySigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return mysigmoid(x)


activation_function = MySigmoid()

def fsigmoid(pre, lat, lon):
    res=1/(1+np.exp((pre-prespivot(lat,lon))/50))
    return res

presgrid=pd.read_csv("../dataset/CY_doy_pres_limit.csv", "\t").to_numpy()
data=pd.read_csv("../dataset/data_CT.csv")
mont_dict = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}

dataset=data[data.categ=='training']
validation=data[data.categ=='validation']

out_d=dataset['tcarbn'].to_numpy()
in_d=dataset[['date','date', 'date','latitude', 'longitude', 'pres', 'temp', 'doxy', 'psal']].to_numpy()

out_v=validation['tcarbn'].to_numpy()
in_v=validation[['date','date', 'date', 'latitude', 'longitude', 'pres', 'temp', 'doxy', 'psal']].to_numpy()




