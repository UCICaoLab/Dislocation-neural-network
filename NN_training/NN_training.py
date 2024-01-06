# this script is used for neural network training

# predefined packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau
from keras.utils import Sequence

import os
import math
import numpy as np
import pandas as pd

# user defined packages
from generate_batch_data import DataGenerator

class LossHistory(keras.callbacks.Callback): 
   
    def on_train_begin(self, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):  #{} dictionary
        
        loss = logs.get("loss")
        val_loss = logs.get("val_loss")
        mean_absolute_error = logs.get("mean_absolute_error")
        val_mean_absolute_error = logs.get("val_mean_absolute_error")

        with open("log.txt","a+") as f:
            f.write(str(loss) + " " + str(val_loss) + " " + str(mean_absolute_error) + " " + str(val_mean_absolute_error))
            f.write("\n")



def neural_network(width, channels):

    # set up activation function and initialization way
    activation = "relu"
    initializer = tf.keras.initializers.he_normal()  #initialize initial weights paremeters

    # input 9x9x9x1
    inputs = keras.Input((width, channels)) #build model by knowing the input

    x = inputs
    #print (x.shape)
    
    x = layers.Flatten()(x)
    
    for _ in range(4):

        x = layers.Dense(units=256, kernel_initializer=initializer)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x) 
        
    
    outputs = layers.Dense(units=1, kernel_initializer=initializer)(x)

    model = Model(inputs, outputs, name = "nn")

    return model

################################# Prepare datasets
# check gpu 
physical_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# general parameters
task = "train"

# upload file path
data_directory =""
label_directory =""

number_of_data_groups = 78
number_of_data_per_group = 12000
train_percent, val_percent = 0.8, 0.2
index = int(number_of_data_per_group * train_percent)
train_index=index


# settings for reading input data
params = {"dim": (720),
          "number_of_channels": 1,
          "batch_size": 30,
          "shuffle": True}

if task == "predict":
    params["shuffle"] = False

# labels of datasets  
labels = [np.load(label_directory + str(i) + ".npy") for i in range(number_of_data_groups)]
labels = np.array(labels)

print("dataset labels shape:", labels.shape)

# assign training/validation ids & labels

training_ids = [str(i) + "-" + str(j) for i in range(number_of_data_groups) for j in range(train_index)]
training_labels = {str(i) + "-" + str(j) : labels[i, j] for i in range(number_of_data_groups) for j in range(train_index)} 

validation_ids = [str(i) + "-" + str(j) for i in range(number_of_data_groups) for j in range(index, number_of_data_per_group)]
validation_labels = {str(i) + "-" + str(j) : labels[i, j] for i in range(number_of_data_groups) for j in range(index, number_of_data_per_group)}

print("training set size", len(training_ids))
print("validation set size", len(validation_ids))

# generate data
training_generator = DataGenerator(data_directory, training_ids, training_labels, **params)  #** represents unpack key-value match
validation_generator = DataGenerator(data_directory, validation_ids, validation_labels, **params)

width, channels= 720, 1

# build model

# variables
epochs = 100
best_model_path = "model_weights"

# call model
model = neural_network(width, channels)
print(model.summary(line_length=120))

if task == "train":

    # schedule learning rate
    # steps_per_epoch = np.floor((number_of_train_data / params["batch_size"]))
    initial_learning_rate = 0.001
    lr_schedule = ReduceLROnPlateau(monitor="val_loss",
                                    factor=0.1,
                                    patience=10,
                                    mode="auto",
                                    cooldown=0,
                                    min_lr=1e-5,)
                                                          
    # lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=steps_per_epoch, decay_rate=0.96, staircase=True)

    # compile model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=initial_learning_rate),
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[tf.keras.metrics.MeanAbsoluteError()],)

    loss_info = LossHistory()

    # additional settings
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_mean_absolute_error", patience=10)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(best_model_path, 
                                                                   monitor='val_loss', 
                                                                   save_best_only=True, 
                                                                   save_weights_only=True, 
                                                                   mode='auto')

    # fit model
    history = model.fit_generator(training_generator, 
                                  epochs = epochs, 
                                  validation_data = validation_generator, 
                                  use_multiprocessing=True,
                                  workers=10,
                                  callbacks=[loss_info, lr_schedule, model_checkpoint_callback])

    weights = model.get_weights()

    np.save("weights.npy", weights)
    # model.save ('NN.model')

if task == "predict":
    
     
    # model.load_weights("4/model_weights")
    model = tf.keras.models.load_model ('NN.model2')
    weights = model.get_weights()

    np.save("weights.npy", weights)

    
    n1 = int(number_of_data_groups * train_index/ params["batch_size"])
    train_predicted = model.predict_generator(generator = training_generator, steps = n1, workers=4)
    print(train_predicted)
    
    n2 = int(number_of_data_groups * number_of_data_per_group * val_percent / params["batch_size"])
    val_predicted = model.predict_generator(generator = validation_generator, steps = n2, workers=20)
    
    print("steps for training:", n1)
    print("steps for validation:", n2)

    data = [labels, train_predicted, val_predicted]
    np.save("model_prediction.npy", data)
    
