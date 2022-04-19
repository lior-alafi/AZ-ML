import keras_preprocessing.image
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, ReLU, Conv1D, Dropout
from keras.optimizers import adam_v2

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('../data/dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory('../data/dataset/test_set', target_size=(64,64),batch_size=32,class_mode='binary')

model = keras.Sequential()
model.add(Conv2D(filters=16,kernel_size=7,strides=(1,1),padding='same',activation='relu',input_shape=(64,64,3)))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2),padding='same'))
model.add(Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'))
model.add(Dropout(rate=0.2))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2),padding='same'))
model.add(Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2),padding='same'))
model.add(Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'))
model.add(Dropout(rate=0.2))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2),padding='same'))
model.add(Flatten())
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer=adam_v2.Adam(learning_rate=0.001),loss='binary_crossentropy',metrics=['accuracy'])

model.fit_generator(training_set,epochs=150,validation_data=test_set)
model.save('../saves/cnn2')

y_pred = model.predict(test_set)
print(y_pred)

