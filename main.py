import cv2
import os
import os.path

# constants
IMG_DIR = 'C:\\Users\\AW51R2\\code\\carnd\\simulator-windows-64\\IMG'
LOG = 'C:\\Users\\AW51R2\\code\\carnd\\simulator-windows-64\\driving_log.csv'


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


def main():

    images, steerings = load_training_data()

    assert len(images) == len(steerings)

    print("shape of image", images[0].shape)
    print("count of data points", len(steerings))


if __name__ == '__main__':
    main()