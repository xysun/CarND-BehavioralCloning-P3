import json

import cv2
import os
import os.path

from keras.layers import Convolution2D, ELU
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Lambda, Flatten

# todo: generator approach; resize img; train drifting leftside by agile approach

# constants
IMG_DIR = 'C:\\Users\\AW51R2\\code\\carnd\\simulator-windows-64\\IMG'
LOG = 'C:\\Users\\AW51R2\\code\\carnd\\simulator-windows-64\\driving_log.csv'
BATCH_SIZE = 32
EPOCH = 10
STEERING_DELTA = 0.25

ROW = 64
COL = 64

def generator(x_train, y_train):

    train_size = len(x_train)

    while 1:
        for i in range(train_size % BATCH_SIZE): #32 * 200 = 6400
            if i % 100 == 0:
                print("i = ", i)
            yield x_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE], y_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

def normalize_img(img):
    # x is a list of images, we normalize each of them
    img = np.array(img)
    img = cv2.resize(img, dsize = (COL,ROW), interpolation = cv2.INTER_AREA)
    img = img.astype('float32')
    img = img / 255. - 0.5

    return img

def load_training_data():
    steerings = []
    images = []
    with(open(LOG, 'r')) as f:
        lines = f.readlines()
        for line in lines:
            t = line.split(',')
            [center_img, left_img, right_img, steering, throttle, is_break, speed] = t
            steering = float(steering)
            steerings.append(steering)
            img = cv2.imread(center_img) # this is BGR
            images.append(normalize_img(img))

            # flip
            flip_img = cv2.flip(img, 1)
            steerings.append(-1*steering)
            images.append(normalize_img(flip_img))

            # brightness adjust

            image_bright = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            random_bright = .25 + np.random.uniform()
            # print(random_bright)
            image_bright[:, :, 2] = image_bright[:, :, 2] * random_bright
            image_bright = cv2.cvtColor(image_bright, cv2.COLOR_HSV2BGR)
            steerings.append(steering)
            images.append(normalize_img(image_bright))


            # left, steering + delta
            if steering != 0:
                radius = 1. / steering
                left_radius = radius + STEERING_DELTA
                right_radius = radius - STEERING_DELTA
                steerings.append(1. / left_radius)
                images.append(normalize_img(cv2.imread(left_img.strip())))

                # right, steering - delta
                steerings.append(1. / right_radius)
                images.append(normalize_img(cv2.imread(right_img.strip())))

    return np.array(images), steerings



def get_model():
    ch, row, col = 3, ROW, COL

    model = Sequential()
    model.add(Lambda(lambda x: x,
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

    model.compile(optimizer="adam", loss="mse")

    return model


def main():

    if not os.path.exists("./outputs"):
        os.makedirs("./outputs")

    try:
        os.remove("./outputs/model.h5")
        os.remove("./outputs/model.json")
    except OSError:
        pass

    images, steerings = load_training_data()

    assert len(images) == len(steerings)

    print("shape of image", images[0].shape) #160,320,3
    print("count of data points", len(steerings))

    # split
    x_train, x_val, y_train, y_val = train_test_split(images, steerings, test_size=0.25, random_state=42)
    y_train = np.array(y_train)
    y_val   = np.array(y_val)

    print("train data size", x_train.shape, y_train.shape)

    #keras
    model = get_model()

    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE, nb_epoch=EPOCH,
              verbose=1, validation_data=(x_val, y_val))

    #model.fit_generator(generator(x_train, y_train), samples_per_epoch=len(x_train), nb_epoch=EPOCH, verbose=1, show_accuracy=True, callbacks=[],
    #                    validation_data=(x_val, y_val), class_weight=None, nb_worker=1)

    # save model
    model.save_weights("./outputs/model.h5", True)
    with open('./outputs/model.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)
    print("model saved!")


if __name__ == '__main__':
    main()