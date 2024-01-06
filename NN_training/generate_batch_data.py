# this script is used for generating mini-batch of data for deep learning

import numpy as np
import tensorflow as tf
from tensorflow import keras

class DataGenerator(keras.utils.Sequence):
    
    def __init__(self, directory_path, data_IDs, labels, batch_size=32, dim=(32, 32, 32), number_of_channels=3, shuffle=True):
        
        self.directory_path = directory_path
        self.data_IDs = data_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.dim = dim
        self.number_of_channels = number_of_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):

        # return number of batches
        return int(np.floor(len(self.data_IDs) / self.batch_size))

    def __getitem__(self, index):  # getitem, when used in a class, allows its instances to use the [] (indexer) operators
        
        # indexes of different batches of data
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
# index = bactch index self.indexes= number in batch index
        # ids of input data
        current_data_IDs = [self.data_IDs[k] for k in indexes]

        # Generate data
        input_data, output_data = self.__data_generation(current_data_IDs)

        return input_data, output_data

    def on_epoch_end(self):
        
        # generate indexes of data
        self.indexes = np.arange(len(self.data_IDs))

        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, current_data_IDs):

        # generate batch data 
        input_data = np.zeros((self.batch_size, self.dim, self.number_of_channels))
        output_data = np.zeros((self.batch_size), dtype=float)
        # new_input_data = np.zeros((self.batch_size, 1440, self.number_of_channels))

        for i, ID in enumerate(current_data_IDs):
            
            displace=np.load('0_displacement.npy').reshape(720,1)
            input_data[i] = np.load(self.directory_path + ID + ".npy").reshape(720,1)
            new_input_data[i]=np.concatenate((input_data[i],displace), axis=None).reshape(1440,1)
            #print ("new_input shape:", new_input_data[i].shape)
            output_data[i] = self.labels[ID]

        return new_input_data, output_data
