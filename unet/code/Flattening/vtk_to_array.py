#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This code converts the the .vtk files representing the LAA data and saves the
connectivity and coordinate data to .csv and .mat datasets

Follow by running the flattening_main.m file to obtain the cartesian representation of the data

@author: Xabier Morales Ferez - xabier.morales@upf.edu

"""

from paraview.simple import *
from glob import glob
from ntpath import basename
import os, re, numpy as np, scipy.io as sio, matlab.engine, pyvista as pv

inp_path = 'D:\PhD\DL\Data\CNN' # Base directory

#%% Intitialize input and output folders

path_vtk = inp_path + '\LAA_Smooth'
path_ply = inp_path + '\LAA_PLY'
path_inter = inp_path + '\LAA_remeshed'
path_out = inp_path + '\Excel'
                      
#%% Save ECAP, connectivity and coordinate (x,y,z) position data for each node

rem = glob(path_inter+"\\*.vt*")

for r in rem:
        
    case = re.sub("[^0-9]", "", basename(r))
    
    #%% Save the connectivity and coordinate info in .mat files
    
    mesh=pv.read(r)
    nCells=mesh.GetNumberOfCells()
    cell2p=mesh.point_data_to_cell_data()
    points = [mesh.GetPoint(i) for i in range(int(mesh.GetNumberOfPoints()))]
    connectivity = mesh.faces
    connectivity=np.reshape(connectivity,[nCells,4])
    connectivity=connectivity[:,1:4]
    
    X=matlab.double(list(points))
    F=matlab.double(connectivity.tolist())
    
    sio.savemat(os.path.join(path_out,'X_python'+str(case)+'.mat'),{'X_python':points})
    sio.savemat(os.path.join(path_out,'F_python'+str(case)+'.mat'),{'F_python':connectivity})
    
    #%% Save the ECAP data

    gT1rvtk = LegacyVTKReader(FileNames=r) if r[-3:]=='vtk' else  XMLPolyDataReader(FileName=r)

    SetActiveSource(gT1rvtk)
    spreadSheetView1 = GetActiveViewOrCreate('SpreadSheetView')
    gT1rvtkDisplay = Show(gT1rvtk, spreadSheetView1)
    spreadSheetView1.HiddenColumnLabels = ['Points','Points_Magnitude','Point ID']   
    ExportView(path_out+'\\ECAP_'+str(case)+'.csv', view=spreadSheetView1)

