# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 12:47:29 2020

@author: ashwi
"""

#import cv2

import tensorflow as tf
from glob import glob
#import matplotlib
#matplotlib.use("agg")
#import matplotlib.pyplot as plt

files_rgb = sorted(glob('/home/asjo/research_project/dataset/rgb/*.jpg')) # Path to the dataset in lab system
files_gt = sorted(glob('/home/asjo/research_project/dataset/gt/*.jpg'))

files_rgb_val = sorted(glob('/home/asjo/research_project/dataset/rgb_val/*.jpg')) # Path to the Validation dataset in lab system
files_gt_val = sorted(glob('/home/asjo/research_project/dataset/gt_val/*.jpg'))

# Map function for rgb & gt
def process_images(x, y):
    x = tf.io.read_file(x) # Reads and outputs the entire contents of the i/p filename
    x = tf.image.decode_jpeg(x, channels=3)
    x = tf.image.resize(x,(512,512))  
    x = tf.image.convert_image_dtype(x, tf.float32)
    
    x /= 255.0
    
    y = tf.io.read_file(y)
    y = tf.image.decode_jpeg(y, channels=1)
    y = tf.image.resize(y,(512,512))    
    y = tf.image.convert_image_dtype(y, tf.float32)
    
    y /= 255.0
        
    return x,y

ds = tf.data.Dataset.from_tensor_slices((files_rgb, files_gt)) # Here the object ds is iterable
ds = ds.map(process_images)
ds = ds.batch(4)

iterator = iter(ds) 
files_rgb, files_gt = iterator.get_next()

#ds = ds.repeat(400)
# Map function for rgb_val & gt_val
def process_images_val(a, b):
    a = tf.io.read_file(a) # Reads and outputs the entire contents of the i/p filename
    a = tf.image.decode_jpeg(a, channels=3)
    a = tf.image.resize(a,(512,512))  
    a = tf.image.convert_image_dtype(a, tf.float32)
    
    a /= 255.0
    
    b = tf.io.read_file(b)
    b = tf.image.decode_jpeg(b, channels=1)
    b = tf.image.resize(b,(512,512))    
    b = tf.image.convert_image_dtype(b, tf.float32)
    
    b /= 255.0
        
    return a,b

ds_val = tf.data.Dataset.from_tensor_slices((files_rgb_val, files_gt_val)) # Here the object ds is iterable
ds_val = ds_val.map(process_images_val)
ds_val = ds_val.batch(4)

iterator_val = iter(ds_val) 
files_rgb_val, files_gt_val = iterator_val.get_next()

"""
for d in ds.take(1):
    x,y = d
    x_n = x.numpy()
    y_n = y.numpy()
    plt.subplot(1,2,1)
    plt.imshow(x_n[0] * 255)
    plt.subplot(1,2,2)
    plt.imshow((y_n[0]*255).squeeze(axis=2))   # Use squeeze(axis=2) in order to plot images with 1 channel
    print(x.shape, y.shape)
    break
"""
# Network Design Architecture
ResNet50_encoder = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(512,512,3)) # This does not include the last layer

# Block  1
Layer = tf.keras.layers.UpSampling2D()(ResNet50_encoder.output) # .output should be added because it takes i/p from an entire model which is ResNet50 here
for i in range (0,2):
    Layer = tf.keras.layers.Conv2D(1024,(3,3),activation = 'linear',  padding = 'same')(Layer)
    Layer = tf.keras.layers.BatchNormalization()(Layer)
    Layer = tf.keras.activations.relu(Layer)    
# Block 2
Layer = tf.keras.layers.UpSampling2D()(Layer)  # .output is not added because the previous one is a single layer
for i in range (0,2):
    Layer = tf.keras.layers.Conv2D(512,(3,3),activation = 'linear',  padding = 'same')(Layer)
    Layer = tf.keras.layers.BatchNormalization()(Layer)
    Layer = tf.keras.activations.relu(Layer)    
# Block 3
Layer = tf.keras.layers.UpSampling2D()(Layer)  # .output is not added because the previous one is a single layer
for i in range (0,2):
    Layer = tf.keras.layers.Conv2D(256,(3,3),activation = 'linear', padding = 'same')(Layer)
    Layer = tf.keras.layers.BatchNormalization()(Layer)
    Layer = tf.keras.activations.relu(Layer)
# Block 4
Layer = tf.keras.layers.UpSampling2D()(Layer)  # .output is not added because the previous one is a single layer
for i in range (0,2):
    Layer = tf.keras.layers.Conv2D(128,(3,3),activation = 'linear', padding = 'same')(Layer)
    Layer = tf.keras.layers.BatchNormalization()(Layer)
    Layer = tf.keras.activations.relu(Layer)
    
Layer = tf.keras.layers.UpSampling2D()(Layer)  # .output is not added because the previous one is a single layer
#here 64, 64, 1
Layer = tf.keras.layers.Conv2D(64,(3,3),activation = 'linear',  padding = 'same')(Layer)
Layer = tf.keras.activations.relu(Layer)
Layer = tf.keras.layers.Conv2D(64,(3,3),activation = 'linear', padding = 'same')(Layer)
Layer = tf.keras.activations.relu(Layer)
Layer = tf.keras.layers.Conv2D(1,(3,3), activation = 'sigmoid', padding = 'same')(Layer)

encoder_decoder = tf.keras.models.Model(ResNet50_encoder.input, Layer)
encoder_decoder.summary()

# try sparse, mse, 
# 
encoder_decoder.compile(loss = tf.keras.losses.MeanSquaredError(),#'binary_crossentropy',#tf.keras.losses.SparseCategoricalCrossentropy(),#'categorical_crossentropy',
              optimizer =tf.keras.optimizers.SGD(learning_rate = 0.0001),#tf.keras.optimizers.Adam(learning_rate = 0.0008),#'sgd',
              metrics = ['accuracy'])


encoder_decoder.fit(ds, validation_data = ds_val, epochs=20, shuffle=True)

#encoder_decoder = model 

encoder_decoder.save('home/asjo/saved_models')




