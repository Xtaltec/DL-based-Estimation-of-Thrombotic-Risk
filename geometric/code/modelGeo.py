# -*- coding: utf-8 -*-
"""
This code contains the GeometricPointNet model

@author: Xabier Morales Ferez - xabier.morales@upf.edu
"""

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, ELU, BatchNorm1d as BN, Dropout
from torch_geometric.nn import SplineConv, SAGEConv, BatchNorm, global_max_pool
from torch_geometric.data import InMemoryDataset

#%% Create MLP with given channels sizes
def MLP(channels, drop = True, p = 0.5):
    
    if drop==True:
        mlp = Seq(*[ Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]),Dropout(p=p))for i in range(1, len(channels))])    
    else:
        mlp = Seq(*[ Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))for i in range(1, len(channels))])
    
    return mlp

#%% Dataset class

class ECAPdataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(ECAPdataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return ['ECAP.dataset']

    def download(self):
        pass
    
    # def process(self):
    #     data, slices = self.collate(dataset)
    #     torch.save((data, slices), self.processed_paths[0])
        
class BurdeosDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(BurdeosDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return ['Burdeos.dataset']

    def download(self):
        pass
    
    # def process(self):
    #     data, slices = self.collate(dataset)
    #     torch.save((data, slices), self.processed_paths[0]) 
#%% Geometric Point Net

class GeometricPointNet(torch.nn.Module):
    def __init__(self, spline = True, depth=10,hidden=16,drop=0.1,kernel=5,act='elu'):
        super(GeometricPointNet, self).__init__()
        
        self.drop = drop # Dropout
        self.kernel = kernel # Kernel size
        self.depth = depth # Hidden layer numbers
        self.hidden = hidden # Hidden layer depth
        self.in_channels = 7 if spline == False else 4 # Input channels
        self.spline = spline # SplineConv == True, SageConv == False
        self.act = ELU() if act == 'elu' else ReLU() # Activation function
        
        dim = 3 # SplineConv dimension
        
        #Initialize layer list
        self.local = torch.nn.ModuleList()
        self.bn_local = torch.nn.ModuleList()
        
        ## Local feature extraction    
        
        if self.spline == True:
            self.local.append(SplineConv(self.in_channels, hidden, dim=dim, kernel_size=kernel))
            self.bn_local.append(BatchNorm(hidden))            
                           
            for i in range(1,depth):
                
                self.local.append(SplineConv(hidden, hidden, dim=dim, kernel_size=kernel))
                self.bn_local.append(BatchNorm(hidden))
        else: 
            self.local.append(SAGEConv(self.in_channels, hidden))
            self.bn_local.append(BatchNorm(hidden))            
                           
            for i in range(1,depth):
                
                self.local.append(SAGEConv(hidden, hidden))
                self.bn_local.append(BatchNorm(hidden))
                
        ## Global feature extraction      
        self.glob = MLP([self.in_channels+hidden*depth,256,512,1024],False,self.drop)           
            
        ## Prediction layer
        self.pred = MLP([self.in_channels+hidden*depth+1024,512,256,128],True,self.drop)
        self.out = Lin(128,1)

                                  
    def forward(self, data):
        x_spline, x_sage =  torch.cat((data.norm,data.curve),1),torch.cat((data.pos,data.norm,data.curve),1)
        edge_index,edge_attr,batch = data.edge_index,data.edge_attr,data.batch
        
        ## Extract local features with splineconv or sageconv
        local = []
        x = x_sage if self.spline == False else x_spline
        
        for i in range(self.depth):
            
            if self.spline == True:
                x = F.dropout(self.act(self.bn_local[i](self.local[i](x,edge_index,edge_attr))),self.drop)
            else:
                x = F.dropout(self.act(self.bn_local[i](self.local[i](x,edge_index))),self.drop)
                
            local +=[x]

        local_features = torch.cat((torch.cat((local),1),x_spline if self.spline == True else x_sage),dim=1)
        
        ## Global feature extraction
        
        x = self.glob(local_features)
        repeats = torch.unique(batch,return_counts= True)[1]
        
        #%% Needs fixing for meshes with no correspondance    
        global_features = global_max_pool(x, batch).repeat_interleave(repeats,dim=0)
        
        ## Final prediction of the ECAP mapping
        
        x = self.pred(torch.cat((local_features,global_features),dim=1))
        x = self.out(x)
        
        return x

