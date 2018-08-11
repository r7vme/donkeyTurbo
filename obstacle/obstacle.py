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
    from donkeyturbo.model import dt_obstacle
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
