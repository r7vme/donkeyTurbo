#!/usr/bin/env python3
# Author: Roma Sokolkov (r7vme) 2018

'''
Label images.

Script does following:
    - show jpg image
    - ask user for input
    - write user input to json file
    - repeat steps above every image in directory

Usage:
    ./obstacle_label.py <path to images>
'''

import os
import json
import sys
import subprocess
import time
import pyautogui

VIEW_TIMEOUT = 5

# Read single byte from input. Prevents additional "Enter" keypress.
def getch():
    import termios
    import sys, tty
    def _getch():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch
    return _getch()

if __name__ == "__main__":
    # Exit if no target directory.
    if len(sys.argv) == 1 :
        print("usage: {0} <img directory to label>".format(sys.argv[0]))
        sys.exit(1)

    try:
        from PIL import Image
    except ModuleNotFoundError:
        print("error: Pillow not found. To install use \"pip install Pillow\".")
        sys.exit(1)

    img_path = os.path.normpath(sys.argv[1])

    if not os.path.exists(img_path):
        print("error: {0} does not exist".format(img_path))
        sys.exit(1)

    files = os.listdir(img_path)
    files.sort()
    processed = 0
    for f in files:
        fpath = os.path.join(img_path, f)
        fpath_meta = os.path.splitext(fpath)[0] + ".json"

        # Process only jpg files.
        if not fpath.endswith("jpg"):
            continue

        # Skip if meta already exists.
        if os.path.isfile(fpath_meta):
            print("info: skipping {0}".format(fpath))
            continue

        with Image.open(fpath) as im:
            viewer = subprocess.Popen(["feh", fpath], stdin=subprocess.DEVNULL)
            time.sleep(0.1)
            # Specific for Awesome WM, swich to next window.
            pyautogui.hotkey('winleft', 'k')
            print("info: input 0 - no obstacle, 1 - left, 2 - right")
            value = getch()
            viewer.terminate()
            if int(value) not in [0, 1, 2]:
                print("error: input is incorrect {0}".format(value))
                sys.exit(1)
            meta = {"image": f, "obstacle": value}

        # Write meta.
        with open(fpath_meta, "w") as m:
            print("info: writing {0}".format(meta))
            json.dump(meta, m)

        # Show progress to the user.
        processed =+ 1
        print('info: processed {0} files\r'.format(processed))
