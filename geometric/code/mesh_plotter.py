# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 17:07:27 2021

@author: u164110
"""
import argparse,torch,numpy as np, pyvista as pv
from torchvision.utils import make_grid,save_image
from os.path import join

def parseArguments():  
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--path", "-p",  type=str, 
                        help="Path with the result data")
    parser.add_argument("--epoch", "-e",  type=str, default='0',
                        help="Training epoch")
    parser.add_argument("--n_images", "-ni",  type=int, default=5,
                        help="Number of cases to plot")
    
    args = parser.parse_args()

    return args

args = parseArguments()

temp = np.load(join(args.path,'Result.npz'))
labels,predictions,indices = temp['labels'],temp['predictions'],temp['indices']

#dataset = torch.load(join(args.path,'Result.dataset'))

# Start xvfb
pv.start_xvfb()

image_GT,image_PR = [],[]
for k in range(args.n_images):
    
    or_case = int(np.where(np.array(dataset.data.case)==indices[k])[0][0])
    mesh_GT = pv.PolyData(dataset[or_case].coord.numpy(),dataset[or_case].connectivity)
    mesh_PR = pv.PolyData(dataset[or_case].coord.numpy(),dataset[or_case].connectivity)
    
    mesh_GT.point_data['ECAP'] = labels[k]
    mesh_PR.point_data['ECAP'] = predictions[k]
    
    # PyVista headless display
    pv.set_plot_theme("document") # White background

    p_GT = pv.Plotter(shape=(1,1), border=False, title = str(indices[k]),window_size=[200,300],off_screen=True)
    p_PR = pv.Plotter(shape=(1,1), border=False,window_size=[200,300],off_screen=True)

    p_GT.add_text("Ground truth",font_size=15)
    p_GT.add_mesh(mesh_GT, scalars='ECAP',clim=[0,6], 
                      scalar_bar_args=dict(position_x = 0.8,label_font_size=10,title_font_size=15))
   
    p_PR.add_text("Prediction",font_size=15)
    p_PR.add_mesh(mesh_PR, scalars='ECAP',clim=[0,6],show_scalar_bar=False)
    
    image_GT += [p_GT.screenshot(None, return_img=True)]
    image_PR += [p_PR.screenshot(None, return_img=True)]
    
    # a=pv.read('/homedtic/xmorales/ECAP/GitHub/geometric/code/Prueba.vtk')
    # p=pv.Plotter(shape=(1,1),off_screen=True)
    # p1=pv.Plotter(shape=(1,1),off_screen=True)
    # p.add_mesh(a)
    # p1.add_mesh(a)

    # image_GT += [p.screenshot(None, return_img=True)]
    # image_PR += [p1.screenshot(None, return_img=True)]

image = np.moveaxis(np.concatenate([np.stack(image_GT,axis=0),np.stack(image_PR,axis=0)]),3,1).astype('int')/255
image_grid = make_grid(torch.tensor(image), nrow=args.n_images)
save_image(image_grid,join(args.path,'Result_'+args.epoch.zfill(3)+'.png'))
                                                                    