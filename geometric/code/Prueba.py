# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 12:03:42 2021

@author: u164110
"""

import os, random, torch , argparse, numpy as np, pandas as pd
from pathlib import Path
from os.path import normpath,join
from itertools import product
import wandb # Hyperdash is not supported anymore. Replaced by Weights & Bias
from torch.nn import DataParallel
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T
from torchvision.utils import make_grid
import subprocess

if __name__ == "__main__":
    
    print('Start main')
    
    number = 42
    
    subprocess.call(['python', 'Secondary.py','-n',str(number)])
    
    print('Success main')