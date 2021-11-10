# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 17:07:27 2021

@author: u164110
"""
import os,torch,argparse,numpy as np,pyvista as pv
from os.path import join
from torchvision.utils import make_grid,save_image

def parseArguments():  
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p",  type=str, default= 'C:\\Users\\Xabier\\PhD\\Frontiers\\GitHub\\geometric\\results\\Frontiers_Prueba_Run',
                        help="Path with the result data")
    parser.add_argument("--epoch", "-e",  type=str, default='0',
                        help="Epoch the results are coming from")
    parser.add_argument("--n_images", "-ni",  type=int, default=5,
                        help="Number of images to plot")
    
    args = parser.parse_args()

    return args

args = parseArguments()

# Load the arrays with the prediction data for this timestep
temp = np.load(join(args.path,'Temp.npz'),allow_pickle=True)
labels,predictions,indices,coord,connectivity = temp['labels'],temp['predictions'],temp['indices'],temp['coord'],temp['connectivity']

# Start xvfb if Sys == Linux
pv.start_xvfb() if os.name == 'posix' else None

# Create PolyData objects and plot them through a headless display with PyVista
image_GT,image_PR = [],[]
for i in range(args.n_images):
    
    mesh_GT, mesh_PR = pv.PolyData(coord[i],connectivity[i]), pv.PolyData(coord[i],connectivity[i]) 

    mesh_GT.point_data['ECAP'], mesh_PR.point_data['ECAP'] = labels[i], predictions[i]

    pv.set_plot_theme("document") # White background

    p_GT = pv.Plotter(shape=(1,1), border=False, title = str(indices),window_size=[200,250],off_screen=True)
    p_PR = pv.Plotter(shape=(1,1), border=False,window_size=[200,250],off_screen=True)

    p_GT.add_text("Ground truth",font_size=15)
    p_GT.add_mesh(mesh_GT, scalars='ECAP',clim=[0,6], cmap="jet",
                  scalar_bar_args=dict(position_x = 0.23,label_font_size=10,title_font_size=15))
       
    p_PR.add_text("Prediction",font_size=15)
    p_PR.add_mesh(mesh_PR, scalars='ECAP',clim=[0,6], cmap="jet", show_scalar_bar=False)
    
    image_GT += [p_GT.screenshot(None, return_img=True)]
    image_PR += [p_PR.screenshot(None, return_img=True)]

# Arrange ground truth and prediction plots in an array and save it as .png
image = np.moveaxis(np.concatenate([np.stack(image_GT,axis=0),np.stack(image_PR,axis=0)]),3,1).astype('int')/255
image_grid = make_grid(torch.tensor(image), nrow=args.n_images)
save_image(image_grid,join(args.path,'Result_'+args.epoch.zfill(3)+'.png'))
                                                                    