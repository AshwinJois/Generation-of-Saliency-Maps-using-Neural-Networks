# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 12:47:29 2020

@author: ashwi
"""
import pickle
import tensorflow as tf
from glob import glob


files_rgb = sorted(glob('/home/asjo/rgb/*.jpg')) # Path to the dataset in lab system
files_gt = sorted(glob('/home/asjo/gt/*.jpg'))

files_rgb_val = sorted(glob('/home/asjo/rgb_val/*.jpg')) # Path to the Validation dataset in lab system
files_gt_val = sorted(glob('/home/asjo/gt_val/*.jpg'))

def process_images(x, y):
    x = tf.io.read_file(x) # Reads and outputs the entire contents of the i/p filename
    x = tf.image.decode_jpeg(x, channels=3)
    
    #x = tf.image.convert_image_dtype(x, tf.float32) ###  normalized to values between [0..1]
    x = tf.cast(x, tf.float32)
    x = tf.keras.applications.resnet.preprocess_input(x)#preprocess_input(x)
    #x /= 255.0
    x = tf.image.resize(x,(512,512))
    
    
    y = tf.io.read_file(y) 
    y = tf.image.decode_jpeg(y, channels=1) ########################################
    
    y = tf.cast(y, tf.float32)
    #y = tf.image.convert_image_dtype(y, tf.float32)
    
    y /= 255.0
    y = tf.image.resize(y,(512,512))
    
    return x,y

ds = tf.data.Dataset.from_tensor_slices((files_rgb, files_gt)) # Here the object ds is iterable
ds = ds.map(process_images)
ds = ds.batch(4)

ds_val = tf.data.Dataset.from_tensor_slices((files_rgb_val, files_gt_val)) # Here the object ds is iterable
ds_val = ds_val.map(process_images)
ds_val = ds_val.batch(4)

# Network Design Architecture
ResNet50_encoder = tf.keras.applications.ResNet50(include_top=False, input_shape=(512,512,3)) # This does not include the last layer
#Layer =tf.keras.applications.resnet50.preprocess_input(include_top=False, input_shape=(512,512,3))
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
Layer = tf.keras.layers.Conv2D(1,(3,3), activation = 'sigmoid', padding = 'same')(Layer)  #########################################

encoder_decoder = tf.keras.models.Model(ResNet50_encoder.input, Layer)
encoder_decoder.summary()


#SSIM function
def SSIM(y_true, y_pred):

    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))


encoder_decoder.compile(loss = 'binary_crossentropy',
              optimizer =  tf.keras.optimizers.SGD(learning_rate = 0.0001),
              metrics = [SSIM])
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='home/asjo/saved_models',
        save_best_only=True,  # Only save a model if `val_loss` has improved.
        monitor="loss",#"val_loss",
        verbose=1,
    )
]

hist = encoder_decoder.fit(ds, validation_data = ds_val, epochs=50, callbacks=callbacks, shuffle=True)
hist.history

f = open('home/asjo/saved_models/history.pckl', 'wb')
pickle.dump(hist.history, f)
f.close()




