import json

import cv2
import os
import os.path

from keras.layers import Convolution2D, ELU
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Lambda, Flatten
from keras.optimizers import SGD, Adam, RMSprop

# constants
IMG_DIR = 'C:\\Users\\AW51R2\\code\\carnd\\simulator-windows-64\\IMG'
LOG = 'C:\\Users\\AW51R2\\code\\carnd\\simulator-windows-64\\driving_log.csv'
BATCH_SIZE = 32
EPOCH = 5

def load_training_data():
    steerings = []
    images = []
    with(open(LOG, 'r')) as f:
        lines = f.readlines()
        for line in lines:
            t = line.split(',')
            [center_img, left_img, right_img, steering, throttle, is_break, speed] = t
            steerings.append(steering)

            img = cv2.imread(os.path.join(IMG_DIR, center_img)) # this is BGR
            images.append(img)

    return images, steerings

def normalize_img(x):
    x = np.array(x)
    x = x.astype('float32')
    return x

def get_model():
    ch, row, col = 3, 160, 320

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
                     input_shape=(row, col, ch),
                     output_shape=(row, col, ch)))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

    return model


def main():

    images, steerings = load_training_data()

    assert len(images) == len(steerings)

    print("shape of image", images[0].shape)
    print("count of data points", len(steerings))

    # split
    x_train, x_val, y_train, y_val = train_test_split(images, steerings, test_size=0.25, random_state=42)
    x_train = normalize_img(x_train)
    x_val   = normalize_img(x_val)
    y_train = np.array(y_train)
    y_val   = np.array(y_val)

    print("train data size", x_train.shape, y_train.shape)

    #keras
    model = get_model()
    history = model.fit(x_train, y_train,
                        batch_size=BATCH_SIZE, nb_epoch=EPOCH,
                        verbose=1, validation_data=(x_val, y_val))

    # save model
    if not os.path.exists("./outputs"):
        os.makedirs("./outputs")
    model.save_weights("./outputs/model.h5", True)
    with open('./outputs/model.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)



if __name__ == '__main__':
    main()