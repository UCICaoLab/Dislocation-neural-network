# convert discrete atomic information to image

import time
import numpy as np
import pandas as pd
from scipy import spatial

class convert_image():

    def create_mesh(self, cutoff, voxel_size):
       
        # print(cutoff, voxel_size)
        number_of_voxels = np.round(cutoff.flatten() / voxel_size.flatten())
        mesh_shape = number_of_voxels
        mesh_size = number_of_voxels * voxel_size.flatten()
      
        x = np.arange(0, number_of_voxels[0]) * voxel_size[0] 
        y = np.arange(0, number_of_voxels[1]) * voxel_size[1] 
        z = np.arange(0, number_of_voxels[2]) * voxel_size[2] 

        mesh_x, mesh_y, mesh_z = np.meshgrid(x, y, z) 
        mesh_x, mesh_y, mesh_z = mesh_x.reshape(-1, 1), mesh_y.reshape(-1, 1), mesh_z.reshape(-1, 1)    
        mesh = np.c_[mesh_x, mesh_y, mesh_z]

        return mesh, np.array(mesh_shape), np.array(mesh_size)

    def convert_data_to_image(self, data, mesh_info, new_mesh_size):
        
        mesh, mesh_shape, mesh_size = mesh_info 
        
        kdtree = spatial.cKDTree(data=mesh, boxsize=mesh_size)
        
        distance, index = kdtree.query(data[:, 1:4] - 0.5 * new_mesh_size + 0.5 * mesh_size, k=1)
        intensity = np.zeros(len(mesh))
        intensity[index] = data[:, 0] # using atom type to represent voxel intensity
        
        return intensity
