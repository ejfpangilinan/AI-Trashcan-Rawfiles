import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt

import os
import glob



# TRAINING_DIR = 'DATASET\TRAIN'
# VALIDATION_DIR = 'DATASET\TEST'



# validation_datagen = ImageDataGenerator(rescale = 1./255)

# validation_generator = validation_datagen.flow_from_directory(
#     VALIDATION_DIR,
#     target_size = (200,200),
#     class_mode = 'binary',
#     batch_size=10
# )

# training_datagen = ImageDataGenerator(
#         rescale = 1./255,
#         fill_mode='nearest')

# train_generator = training_datagen.flow_from_directory(
#     TRAINING_DIR,
#     target_size = (200,200),
#     class_mode = 'binary',
#     batch_size=10
# )



# model = tf.keras.models.Sequential([
#     # since Conv2D is the first layer of the neural network, we should also specify the size of the input
#     tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(200, 200, 3)),
#     # apply pooling
#     tf.keras.layers.MaxPooling2D(2,2),
#     # and repeat the process
#     tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2,2), 
#     tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 
#     tf.keras.layers.MaxPooling2D(2,2),
#     # flatten the result to feed it to the dense layer
#     tf.keras.layers.Flatten(), 
#     # and define 512 neurons for processing the output coming by the previous layers
#     tf.keras.layers.Dense(512, activation='relu'), 
#     # a single output neuron. The result will be 0 if the image is a cat, 1 if it is a dog
#     tf.keras.layers.Dense(1, activation='sigmoid')  
# ])

# steps_per_epoch = len(train_generator)//10
# validation_steps = len(validation_generator)//10

# model.summary()

# model.compile(optimizer="adam",
#               loss='binary_crossentropy',
#               metrics = ['accuracy'])

# history = model.fit(train_generator, epochs=20, steps_per_epoch=steps_per_epoch, validation_data = validation_generator, verbose=1,validation_steps= validation_steps)

# model.save("rps.h5")


# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs = range(len(acc))

# plt.plot(epochs, acc, 'r', label='Training accuracy')
# plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
# plt.title('Training and validation accuracy')
# plt.legend(loc=0)
# plt.figure()


# plt.show()



from keras.models import load_model
import numpy as np
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img


model = load_model('./rps.h5')
# we define a new custom Keras model that receives an input image

successive_outputs = [layer.output for layer in model.layers]
visualization_model = tf.keras.models.Model(inputs=model.input, outputs=successive_outputs)
img = load_img('DATASET\TEST\O\O_12574.jpg', target_size=(200, 200))  # this is a raw image in PIL format
x   = img_to_array(img)                           # np array with shape (150, 150, 3)
x   = x.reshape((1,) + x.shape)                   # np array with shape  (1, 150, 150, 3)

# rescale pixel values 1/255
x /= 255.0

# by making calling the predict method we obtain
# the intermediate representations of this image from the previous model
successive_feature_maps = visualization_model.predict(x)

# let's map the layers of this model with their name
layer_names = [layer.name for layer in model.layers]

# plot everything
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
  
  if len(feature_map.shape) == 4: # if it is a conv or pooling layer
    n_features = feature_map.shape[-1]  # n features
    size       = feature_map.shape[ 1]  # shape
    
    # create a grid to display the data
    display_grid = np.zeros((size, size * n_features))
    np.seterr(invalid='ignore')
    
    # some post-processing
    for i in range(n_features):
      x  = feature_map[0, :, :, i]
      x -= x.mean()
      x /= x.std()
      x *=  64
      x += 128
      x  = np.clip(x, 0, 255).astype('uint8')
      display_grid[:, i * size : (i + 1) * size] = x

    # show the chart
    scale = 20. / n_features
    plt.figure( figsize=(scale * n_features, scale) )
    plt.title ( layer_name )
    plt.grid  ( False )
    plt.imshow( display_grid, aspect='auto', cmap='viridis' ) 
plt.show()