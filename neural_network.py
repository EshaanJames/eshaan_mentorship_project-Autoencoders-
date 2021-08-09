import streamlit as st
from keras.layers import Conv2D, UpSampling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb
from skimage.transform import resize
from skimage.io import imsave, imshow
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True




model = Sequential()
@st.cache()
def network():
    global model
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    path = "Landscape"
    train = train_datagen.flow_from_directory(path,
                                              target_size=(256, 256),
                                              batch_size=340,
                                              class_mode=None)
    X = []
    Y = []
    for img in train[0]:
        try:
            lab = rgb2lab(img)
            X.append(lab[:, :, 0])
            Y.append(lab[:, :, 1:] / 128)
        except:
            print('Error')

    X = np.array(X)
    Y = np.array(Y)
    X = X.reshape(X.shape + (1,))

    print(f"""
    Batches in X = {X.shape[0]}
    Size of images {X.shape[1]} x {X.shape[2]} pixels
    Channel  = {X.shape[3]} i.e.L channel
    Batches in Y = {Y.shape[0]}
    Size of images {Y.shape[1]} x {Y.shape[2]} pixels
    Channel  = {Y.shape[3]} i.e. a and b channel""")

    # Encoder:
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2, input_shape=(256, 256, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))

    # Decoder:
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    model.summary()

    model.fit(X, Y, validation_split=0.1, epochs=1, batch_size=16)

network()