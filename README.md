# Dislocation neural network

This project provides input files and scripts for predicting Peierls barriers using a neural network model. It includes various components structured in different folders, each serving a specific purpose in the workflow.  

1. Folder: NN_training

Generate_batch_data.py: Defines a ‘DataGenerator’ Python class for real-time data feeding to the Keras model. It reads the NumPy array of each example from its corresponding file (e.g., 0-0.npy). For a detailed explanation, please refer to https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly 

NN_training.py: This script trains the neural network model, which consists of 4 layers with 256 neurons each as an example. 

Training_validation.npy: Contains predicted results versus the actual barriers computed using the Nudged Elastic Band (NEB) method in LAMMPS, for both training and validation datasets.

Prediction.npy: Features predicted results compared to the actual barriers for prediction dataset. 

2. Folder Lattice_representation:

lattice_representation.py: This script, when executed, utilizes ‘0.lmp’ to generate the file 0-0.npy, which is the inputs for neural network training.

3. Folder Random_configurations: 

Assign_screw_type.py: This Python script uses ‘initial_screw.lmp’ as input to randomly assign 10,000 different configurations with a specified composition for NEB calculations.

