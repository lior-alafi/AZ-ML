import keras_preprocessing.image
import numpy as np
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

model = keras.models.load_model('../saves/cnn2')

# model.fit_generator(training_set,epochs=150)
# model.save('../saves/cnn')
# #
y_pred = model.predict(test_set)
print(y_pred)
a = model.evaluate(test_set)
print(a)

def load_image(path,size):
    original = keras.preprocessing.image.load_img(path,target_size=(64,64))
    # img = tf.image.resize(original, size=size)
    img = keras.preprocessing.image.img_to_array(original)

    img = np.expand_dims(img, 0)
    img = img * 1. / 255
    return original,img
def label_pred(y_pred):
    return 'dog' if y_pred >= 0.5 else 'cat'
orig_img,img = load_image('dog1.PNG',(64,64))
y_pred = model.predict(img)
print(y_pred)

plt.title(label_pred(y_pred))
plt.imshow(orig_img)
plt.show()
