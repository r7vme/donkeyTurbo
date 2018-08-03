#!/usr/bin/env python3
# Author: Roma Sokolkov (r7vme) 2018

'''
Train and run obstacle model.
'''

from PIL import Image
import numpy as np
import time
import sys
import json
import os
import subprocess


def dt_obstacle():
    from keras.models import Model
    from keras.layers import Input, Convolution2D, Dropout, Flatten, Dense

    # Use here 78 pixels height instead default 120.
    img_in = Input(shape=(78, 160, 3), name='img_in')                      # First layer, input layer, Shape comes from camera.py resolution, RGB
    x = img_in
    x = Convolution2D(24, (5,5), strides=(2,2), activation='relu')(x)       # 24 features, 5 pixel x 5 pixel kernel (convolution, feauture) window, 2wx2h stride, relu activation
    x = Convolution2D(32, (5,5), strides=(2,2), activation='relu')(x)       # 32 features, 5px5p kernel window, 2wx2h stride, relu activatiion
    x = Convolution2D(64, (5,5), strides=(2,2), activation='relu')(x)       # 64 features, 5px5p kernal window, 2wx2h stride, relu
    x = Convolution2D(64, (3,3), strides=(2,2), activation='relu')(x)       # 64 features, 3px3p kernal window, 2wx2h stride, relu
    x = Convolution2D(64, (3,3), strides=(1,1), activation='relu')(x)       # 64 features, 3px3p kernal window, 1wx1h stride, relu

    # Possibly add MaxPooling (will make it less sensitive to position in image).  Camera angle fixed, so may not to be needed

    x = Flatten(name='flattened')(x)                                        # Flatten to 1D (Fully connected)
    x = Dense(100, activation='relu')(x)                                    # Classify the data into 100 features, make all negatives 0
    x = Dropout(.1)(x)                                                      # Randomly drop out (turn off) 10% of the neurons (Prevent overfitting)
    x = Dense(50, activation='relu')(x)                                     # Classify the data into 50 features, make all negatives 0
    x = Dropout(.1)(x)                                                      # Randomly drop out 10% of the neurons (Prevent overfitting)
    #categorical output of the angle
    obstacle_out = Dense(3, activation='softmax', name='obstacle_out')(x)        # Connect every input with every output and output 15 hidden units. Use Softmax to give percentage. 15 categories and find best one based off percentage 0.0-1.0

    model = Model(inputs=[img_in], outputs=[obstacle_out])
    model.compile(
            optimizer='adam',
            loss={'obstacle_out': 'categorical_crossentropy'}
            )

    return model

def train(data_path):
    files = os.listdir(data_path)
    img_files = []
    for f in files:
        if f.endswith("jpg"):
            img_files.append(f)

    n = len(img_files)

    data = np.zeros((n, 78, 160, 3), dtype="uint16")
    labels = np.zeros((n, 3), dtype="uint16")

    for i, _ in enumerate(img_files):
        f = img_files[i]
        fpath = os.path.join(data_path, f)
        fpath_meta = os.path.splitext(fpath)[0] + ".json"

        # Load image.
        image = Image.open(fpath)
        image_array = np.asarray(image)
        image.close()
        data[i] = image_array

        # Load meta.
        with open(fpath_meta) as j:
            meta = json.load(j)

        obstacle = int(meta["obstacle"])
        l = [0,0,0]
        l[obstacle] = 1
        labels[i] = np.array(l)

    # Finally load model and start training.
    model = dt_obstacle()
    model.fit(data, labels, nb_epoch=10, validation_split=0.2, verbose=1)

    # Save model.
    model.save("obstacle_model" + str(int(time.time())))

def run(model_path, data_path):
    from keras.models import load_model
    model = load_model(model_path)

    # Load image.
    files = os.listdir(data_path)
    img_files = []
    for f in files:
        if f.endswith("jpg"):
            img_files.append(f)

    img_files.sort()

    for i, _ in enumerate(img_files):
        f = img_files[i]
        fpath = os.path.join(data_path, f)
        viewer = subprocess.Popen(["feh", fpath], stdin=subprocess.DEVNULL)
        image_array = np.zeros((1, 78, 160, 3), dtype="uint16")
        image = Image.open(fpath)
        image_array[0] = np.asarray(image)
        image.close()

        out = model.predict(image_array)[0].tolist()
        print("Index: {} Value: {}".format(out.index(max(out)), max(out)))

        time.sleep(0.5)
        viewer.terminate()

if __name__ == "__main__":
    print("Starting...")
    if len(sys.argv) < 2:
        sys.exit(1)

    if sys.argv[1] == 'train':
        train(sys.argv[2])
    if sys.argv[1] == 'run':
        run(sys.argv[2], sys.argv[3])
