# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 12:06:04 2021

@author: u164110
"""

import pyvista as pv
import argparse

def parseArguments():    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--number", "-n",  type=int, default=0)
    
    args = parser.parse_args()

    return args

args = parseArguments()


pv.start_xvfb()

print('The number is: '+str(args.number)+'\n')
print('Start_Xvfb')

p=pv.Plotter(shape=(1,1),off_screen=True)
a=pv.read('/homedtic/xmorales/ECAP/GitHub/geometric/code/Prueba.vtk')
p.add_mesh(a)
p.screenshot('Prueba.jpg', return_img=True)

print('Success with jpg')
