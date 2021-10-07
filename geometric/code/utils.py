# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 14:59:45 2021

@author: u164110
"""

import os
import re
import shutil
import numpy as np
from glob import glob
import pyvista as pv
from ntpath import basename

import torch
import torch_geometric as tgeo
from torch_geometric.data import InMemoryDataset
from torch_geometric import transforms as T
from torch_geometric.transforms import RandomRotate,Compose
from torch_geometric.utils import to_undirected


#%% Generate dataset

# Defining the InMemoryDataset class
class ECAPdataset(InMemoryDataset):
    def __init__(self, root, path_in, aug, name, transform=None, pre_transform=None):
        
        self.path_in = path_in
        self.path_out = root
        self.aug = aug
        self.name = name
        super(ECAPdataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load( self.path_out+self.name+'.dataset')
        
    @property
    def raw_file_names(self):
        return [self.name+'.dataset']
    @property
    def processed_file_names(self):
        return []
    def download(self):
        pass
    
    # Rotation for data augmentation
    def torch_rotate(self,graph):
        
        rtx = RandomRotate(10,axis=0)
        rty = RandomRotate(10,axis=1)
        rtz = RandomRotate(10,axis=2)
        rotation = Compose([rtx,rty,rtz])
            
        return rotation(graph)    
    
    # Pytorch geometric transforms for pseudo-coordinates,...
    def torch_transform(self,graph):
        
        # Make the required transformations through the pytorch module
        # Save the relative Cartesian coordinates of linked nodes in its edge attributes
        # Convert faces to edges through Face to Edge
        # Generate Mesh normals
        # Add a constant value to each node feature through Constant
        
        pre_transform = T.Compose([T.Constant(value=1),T.GenerateMeshNormals(),T.FaceToEdge(), T.Cartesian()])
        transformed = pre_transform(graph) # Apply transform
        
        # Make undirected
        transformed.edge_index = to_undirected(transformed.edge_index)
        
        return transformed 
    
    def process(self):
        
        dataset =[]
        curvature = []
        
        #files = glob(self.path_out+'*')
        
        #for f in files:
        #   shutil.rmtree(f)
        
        geo = glob(self.path_in+"\\*.vt*")
        
        for g in geo:
 
    
            mesh = pv.read(g)
            
            # Save case ID
            case = re.sub("[^0-9]", "", basename(g)).zfill(3)
            
            # Points position of the mesh (x,y,z)
            pos = torch.tensor(mesh.points).float()
            #features.append(shape)
            
            # Target feature of each node (ECAP)
            ECAP = torch.tensor(mesh.point_arrays['ECAP_Both']).unsqueeze(1).float()
            
            # Obtain the faces from the .vtk files
            face = torch.tensor(np.moveaxis(mesh.faces.reshape(-1, 4)[:,1:],0,1))
            
            # Compute the point-wise curvature
            curve = torch.tensor(mesh.curvature()).float()
            
            # Create graph data class
            graph = tgeo.data.Data(y=ECAP,face=face,pos=pos,case = case, curve=curve)
            
            # Make transformations
            transformed = self.torch_transform(graph)
            
            #Append the graph to the dataset
            dataset.append(transformed)
    
            # Data augmentation by rotation of the meshes (didn't work)
            if self.aug == True:
                
                pre_rot = tgeo.data.Data(y=ECAP,face=face,pos=pos,case = case, curve=curve)
                rot = self.torch_rotate(pre_rot)
                transformed_rot = self.torch_transform(rot)
                transformed_rot.case = transformed_rot.case+'_rot' 
                
                dataset.append(transformed_rot)

        
        data, slices = self.collate(dataset)
        torch.save((data, slices), self.path_out+self.name+'.dataset')


def flatten(a, start=0, count=2):
    """ Reshapes numpy array a by combining count dimensions, 
        starting at dimension index start """
    s = a.shape
    return np.reshape(a, s[:start] + (-1,) + s[start+count:])
