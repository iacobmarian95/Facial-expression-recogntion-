import numpy as np
import h5py

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, InputLayer
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

from keras.models import model_from_json

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.applications import resnet50

from batch_fscore import fbeta_score

resnet_model = resnet50.ResNet50(include_top=False, weights='imagenet', pooling='avg');

for layer in resnet_model.layers[:-1]:
    layer.trainable = False

model = Sequential()
model.add(resnet_model)
model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', fbeta_score])

seed = 7
np.random.seed(seed)
training_size = 28273
validation_size = 3533 # size of either test set

batch_size = 48

# Preprocess inputted data
train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range = 30,
        shear_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest',
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale = 1./255)

# Fit the model
train_generator = train_datagen.flow_from_directory(
        './Training',  # this is the target directory
        target_size = (224, 224),  # all images will be resized to 48x48
        batch_size = batch_size,
        #color_mode = 'rbg'
        )

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        './PublicTest',
        target_size = (224, 224),
        batch_size = batch_size,
        #color_mode = 'rbg'
        )

# ~~~~~~~~~~~~~~~~ Check accuracy & F-score ~~~~~~~~~~~~~~~
score = model.evaluate_generator(validation_generator, validation_size // batch_size)
print("TEST")
print(score)
print(("Loss: {0:.3} \nAccuracy: {1:.3%} \nF-Score: {2:.3%}").format(score[0], score[1], score[2]))

# ~~~~~~~~~~~~~~~~~~~~~~ Train model ~~~~~~~~~~~~~~~~~~~~~~
# callback functions
save_best = ModelCheckpoint('weights2.h5', monitor='val_acc', verbose=2, save_best_only=True, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001, verbose=1)

model.fit_generator(
        train_generator,
        steps_per_epoch = training_size // batch_size,
        epochs=10,
        callbacks = [save_best, reduce_lr],
        validation_data=validation_generator,
        validation_steps= validation_size // batch_size
        )
#model.save_weights('VGG16_regular_second_try.h5')  # always save your weights after training or during training
model_json = model.to_json()
with open("facial_expression_model_structure.json", "w") as json_file:
    json_file.write(model_json)
# ~~~~~~~~~~~~~~~~ Check accuracy & F-score ~~~~~~~~~~~~~~~
score = model.evaluate_generator(validation_generator, validation_size // batch_size)
print("TEST")
print(score)
print(("Loss: {0:.3} \nAccuracy: {1:.3%} \nF-Score: {2:.3%}").format(score[0], score[1], score[2]))