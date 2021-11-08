# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 17:07:27 2021

@author: u164110
"""

import os,argparse,numpy as np, pyvista as pv
from os.path import join

def parseArguments():  
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p",  type=str, default= 'C:\\Users\\Xabier\\PhD\\Frontiers\\GitHub\\geometric\\results\\Frontiers_Prueba_Run',
                        help="Path with the result data")
    
    args = parser.parse_args()

    return args

args = parseArguments()

temp = np.load(join(args.path,'Temp.npz'))
labels,predictions,indices,coord,connectivity = temp['labels'],temp['predictions'],temp['indices'],temp['coord'],temp['connectivity']

# Start xvfb
pv.start_xvfb() if os.name == 'posix' else None

mesh_GT, mesh_PR = pv.PolyData(coord,connectivity), pv.PolyData(coord,connectivity) 

mesh_GT.point_data['ECAP'], mesh_PR.point_data['ECAP'] = labels, predictions

# PyVista headless display
pv.set_plot_theme("document") # White background

p_GT = pv.Plotter(shape=(1,1), border=False, title = str(indices),window_size=[200,250],off_screen=True)
p_PR = pv.Plotter(shape=(1,1), border=False,window_size=[200,250],off_screen=True)

p_GT.add_text("Ground truth",font_size=15)
p_GT.add_mesh(mesh_GT, scalars='ECAP',clim=[0,6], cmap="jet",
                  scalar_bar_args=dict(position_x = 0.23,label_font_size=10,title_font_size=15))
   
p_PR.add_text("Prediction",font_size=15)
p_PR.add_mesh(mesh_PR, scalars='ECAP',clim=[0,6], cmap="jet", show_scalar_bar=False)

p_PR.show()
p_GT.show()

p_GT.screenshot(join(args.path,'Temp_GT.png'))
p_PR.screenshot(join(args.path,'Temp_PR.png'))
    
                                                                    