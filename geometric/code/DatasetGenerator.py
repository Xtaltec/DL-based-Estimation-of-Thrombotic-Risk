# -*- coding: utf-8 -*-
"""
This code loads a set of .vtk,.vtp,.vtu files and convert them to a pytorch 
geometric dataset in graph format

@author: Xabier Morales Ferez - xabier.morales@upf.edu
"""

from glob import glob
from ntpath import basename
from pathlib import Path
import os, re, shutil, getpass, numpy as np, pyvista as pv

import torch
import torch_geometric.data as data
from torch_geometric.data import InMemoryDataset
from torch_geometric import transforms as T
from sklearn.preprocessing import PowerTransformer

#%%  Initialization

if os.name == 'nt' and getpass.getuser()=='Xabier':
    base_path = 'C:\\Users\\Xabier\\PhD\\Frontiers\\geo\\data\\' # The general path to the LAA .vtk
elif os.name == 'nt' and getpass.getuser()=='u164110':    
    base_path = 'D:\\PhD\\Frontiers\\GitHub\\geometric\\data\\' # The general path to the LAA .vtk
elif os.name == 'posix':
    base_path = '/media/u164110/Data/PhD/Frontiers/geo/data/' # The general path to the LAA .vtk

name = 'ECAP' #Define the name of the dataset. Options: ECAP,Burdeos,ECAP_rot,ECAPrem
path_in = base_path+'LAA_Smoothed' # Base directory    
path_torch = base_path+'TorchData/' # Output directory for temporal torch dataset

#%% Function definitions   

# PyTorch geometric transforms for pseudo-coordinates,...
def torch_transform(graph):
    
    # Make the required transformations through the pytorch module
    # Save the relative Cartesian coordinates of linked nodes in its edge attributes
    # Convert faces to edges through Face to Edge
    # Generate Mesh normals
    # Add a constant value to each node feature through Constant
    
    pre_transform = T.Compose([T.Constant(value=1),T.GenerateMeshNormals(),T.FaceToEdge(), T.Cartesian()])
    transformed = pre_transform(graph) # Apply transform
    
    return transformed

# PyTorch InMemoryDataset class

class ECAPdataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(ECAPdataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return [path_torch+'processed.data']

    def download(self):
        pass
    
    def process(self):
        data, slices = self.collate(dataset)
        torch.save((data, slices), self.processed_paths[0]) 

#%% Create dataset list

dataset =[]

files = glob(path_torch+'*') # List with all the files in the input folder
Path(path_torch).mkdir(parents=True, exist_ok=True) # Create folder for processed data

for f in files:
    try:
        shutil.rmtree(f)
    except:
        os.remove(f)
    
geo = sorted(glob(path_in+"\\*.vt*") if os.name == 'nt' else sorted(glob(path_in+"/*.vt*"))) # This was due to order issues in linux

for g in geo:

    mesh = pv.read(g)
    
    # Save case ID
    case = re.sub("[^0-9]", "", basename(g)).zfill(3)
    
    # Points position of the mesh (x,y,z)
    pos = torch.tensor(mesh.points).float()
    #features.append(shape)
    
    coord = pos
    
    # Target feature of each node (ECAP)
    ECAP = torch.tensor(mesh.point_arrays['ECAP_Both']).unsqueeze(1).float()   
    
    # Obtain the faces from the .vtk files
    face = torch.tensor(np.moveaxis(mesh.faces.reshape(-1, 4)[:,1:],0,1))
    
    # Compute the point-wise curvature
    curve = torch.tensor(mesh.curvature()).float().unsqueeze(1)
    
    # Create graph data class
    graph = data.Data(case = case, y=ECAP, face=face, pos=pos, coord=coord, curve=curve)
    
    # Make transformations
    transformed = torch_transform(graph)
    
    #Append the graph to the dataset
    dataset.append(transformed)

#%% Create dataset instance 

data_final = ECAPdataset(path_in)

# Scale the curvature data
scaler = PowerTransformer()
data_final.data.curve = torch.tensor(scaler.fit_transform(data_final.data.curve)).float()

# Save the dataset
torch.save(data_final,base_path+name+'.dataset')