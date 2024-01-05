# The code is developed to convert large LAMMPS data files into numpy arrays, which are then utilized for training neural networks.

# import packages
import time
import random
import numpy as np
import pandas as pd
import scipy
import os
import pathlib
import math
import natsort
from natsort import index_natsorted
from ovito.io import *
from ovito.modifiers import *
from ovito.data import *
from ovito.pipeline import *

# Constants
a = 3.27 #lattice constant
X_v= float (math.sqrt(6)/6*a) #X, Y, Z distance depends on orientation
Y_v=float(math.sqrt(2)/2*a)
Z_v=float(math.sqrt(3)/6*a)
X_box=float(15*X_v)  #Numpy array scale 
Y_box=float(8*Y_v)
Z_box=float(6*Z_v)

import convert_image
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
current_path = pathlib.Path(__file__).parent.absolute()

def main():
    
    image_conversion = convert_image.convert_image()
    
    file_names = [f for f in os.listdir(current_path) if f.endswith(".lmp")]
    file_names.sort(key=lambda x:int(x.split('.')[0]))
    count = 0
    data_list=[]
    for file_name in file_names:
        
        data_list=get_data_points(file_name)
        simulation_box_lengths = np.array ([[X_box], [Y_box], [Z_box]])
        simulation_box_lengths = simulation_box_lengths.reshape((1, -1))  #-1 means this dimension is referred to the previous data
        
        # mesh
        voxel_size = np.array([[X_v],[Y_v],[Z_v]])
        voxel_size=voxel_size.reshape((1,-1))
        mesh_info = image_conversion.create_mesh(simulation_box_lengths.T, voxel_size.T)
        
        # convert the local chemistry to image through on lattice representation
        image = image_conversion.convert_data_to_image(data_list, mesh_info, simulation_box_lengths)
        # image_list = [image_conversion.convert_data_to_image(rotated_data, mesh_info) for rotated_data in rotated_data_list]
        
        print(image.shape)
        npy_name = '0-'+str(count)+'.npy'
        np.save(npy_name, image)
        count +=1

# get data points within certain cutoff from initial LAMMPS data file
def get_data_points (file):
    pipeline = import_file((os.path.join(__location__,file)), atom_style = 'atomic')
    pipeline.modifiers.append(ExpressionSelectionModifier(expression = '(Position.X-40.0491)^2+(Position.Y-16.9564)^2>91.72'))
    pipeline.modifiers.append(DeleteSelectedModifier())
    pipeline.modifiers.append(AffineTransformationModifier(transformation = [[1.0, 0.0, 0.0, -30.6], [0.0, 1.0, 0.0, -9.1641], [0.0, 0.0, 1.0, 0.0]], 
        operate_on = {'dislocations', 'vector_properties', 'surfaces', 'voxels', 'particles'}))
    data=pipeline.compute()
    pos = data.particles.position
    atom_type=data.particles.particle_types
    type=[]
    for x in atom_type:
        type.append(int(x))

    x_data=[]
    y_data=[]
    z_data=[]
    for x in pos[:,0]:
        x_data.append(x)
    for y in pos[:,1]:
        y_data.append(y)
    for z in pos[:,2]:
        z_data.append(z)
  
    c=np.dstack((type,x_data,y_data,z_data))
    c=c.reshape(96,4)
    # print(c)
    return (c)

    return 0

if __name__ == "__main__":
    
    main()
