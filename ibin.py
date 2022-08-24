import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt

import os
import glob



TRAINING_DIR = 'DATASET\DATASET\TRAIN'
VALIDATION_DIR = 'DATASET\DATASET\TEST'



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
#     # Note the input shape is the desired size of the image 200x200 with 3 bytes color
#     # This is the first convolution
#     tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(200, 200, 3)),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     # The second convolution
#     tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2,2),
#     # The third convolution
#     tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2,2),
#     #Fourth Convolution
#     tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2,2),

#     # # Flatten the results to feed into a DNN
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dropout(0.5),
#     # 512 neuron hidden layer
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

# steps_per_epoch = len(train_generator)//10
# validation_steps = len(validation_generator)//10

# model.summary()

# model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])

# history = model.fit(train_generator, epochs=100, steps_per_epoch=steps_per_epoch, validation_data = validation_generator, verbose=1,validation_steps= validation_steps)

# model.save("rps.h5")


# import matplotlib.pyplot as plt
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


import numpy as np
from keras.preprocessing import image
from keras.models import load_model

from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tkinter import filedialog as fd
import random


import matplotlib.pyplot as plt
import matplotlib.image as mpimg


model = load_model('./rps.h5')
mylist = [f for f in glob.glob(VALIDATION_DIR+"/O/*.jpg")]
mylist1 = [f for f in glob.glob(VALIDATION_DIR+"/R/*.jpg")]

# print(len(mylist))
mlstcom = mylist+mylist1


print(len(mylist))
random.shuffle(mylist)

test_num = 100
correct = 0
incorrect = 0

#test for bio accuracy
for i in range(test_num):
    fn = random.randrange(0,len(mylist))
    path = mylist[fn] 

    img = image.load_img(path, target_size=(200, 200))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)

    #1 non-bio, 0 bio
    if classes[0][0]<0.5:
        if mylist[fn][23]=='O':
            plt.title("Biodegradable - Correct")
            correct+=1
        else:
            plt.title("Biodegradable - Wrong")
            incorrect+=1
        # plt.title("Biodegradable")
        print(mylist[fn],":",classes,": Biodegradable")
    else:
        if mylist[fn][23]=='R':
            plt.title("Non-Biodegradable - Correct")
            correct+=1
        else:
            plt.title("Non-Biodegradable - Wrong")
            incorrect+=1
        # plt.title("Non-Biodegradable")
        print(mylist[fn],":",classes,": Non-Biodegradable")

    # print(classes)

    img = mpimg.imread(path)
    imgplot = plt.imshow(img)


    plt.show(block=False)
    plt.pause(2)
    plt.close('all')

print("Accuracy = ", (correct/test_num )*100)
print("Wrong= ", (incorrect/test_num)*100)

correct = 0
incorrect = 0

random.shuffle(mylist1)

for i in range(test_num):
    fn = random.randrange(0,len(mylist1))
    path = mylist1[fn] 

    img = image.load_img(path, target_size=(200, 200))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)

    print(mylist1[fn][23])

    #1 non-bio, 0 bio
    if classes[0][0]<0.5:
        if mylist1[fn][23]=='O':
            plt.title("Biodegradable - Correct")
            correct+=1
        else:
            plt.title("Biodegradable - Wrong")
            incorrect+=1
        # plt.title("Biodegradable")
        print(mylist1[fn],":",classes,": Biodegradable")
    else:
        if mylist1[fn][23]=='R':
            plt.title("Non-Biodegradable - Correct")
            correct+=1
        else:
            plt.title("Non-Biodegradable - Wrong")
            incorrect+=1
        # plt.title("Non-Biodegradable")
        print(mylist1[fn],":",classes,": Non-Biodegradable")

    # print(classes)

    img = mpimg.imread(path)
    imgplot = plt.imshow(img)


    plt.show(block=False)
    plt.pause(2)
    plt.close('all')

print("Accuracy = ", (correct/test_num )*100)
print("Wrong= ", (incorrect/test_num)*100)



# for i in range(test_num):
#     fn = random.randrange(0,len(mylist1))
#     path = mylist[fn]



#     img = image.load_img(path, target_size=(200, 200))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)

#     images = np.vstack([x])
#     classes = model.predict(images, batch_size=10)

#     #1 non-bio, 0 bio
#     if classes[0][0]<0.5:
#         if mylist[fn][11]=='B':
#             plt.title("Biodegradable - Correct")
#             correct+=1
#         else:
#             plt.title("Biodegradable - Wrong")
#         # plt.title("Biodegradable")
#         print(mylist[fn],":",classes,": Biodegradable")
#     else:
#         if mylist[fn][11]=='N':
#             plt.title("Non-Biodegradable - Correct")
#             correct+=1
#         else:
#             plt.title("Non-Biodegradable - Wrong")
#         # plt.title("Non-Biodegradable")
#         print(mylist[fn],":",classes,": Non-Biodegradable")

#     # print(classes)
    
#     img = mpimg.imread(path)
#     imgplot = plt.imshow(img)

    
#     plt.show(block=False)
#     plt.pause(2)
#     plt.close('all')

# print("Accuracy = ", (correct/test_num )*100)



