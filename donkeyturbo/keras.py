'''
Module contains pilot class and Keras model.
'''

import donkeycar as dk
from donkeycar.parts.keras import KerasPilot


class DTKerasPilot(KerasPilot):
    def __init__(self, *args, **kwargs):
        super(DTKerasPilot, self).__init__(*args, **kwargs)

        self.model = dt_categorical()

    def run(self, img_arr):
        # Do NN prediction.
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        angle_binned, _ = self.model.predict(img_arr)
        angle_unbinned = dk.utils.linear_unbin(angle_binned)

        # Do post processing.
        angle = self.compute_angle(angle_unbinned)
        throttle = self.compute_throttle(angle_unbinned)

        return angle, throttle

    def compute_angle(self, angle):
        return angle

    def compute_throttle(self, angle):
        throttle = 1 - abs(angle)
        # TODO(r7vme): Make this configurable.
        if throttle <= 0.5:
            throttle = 0.5
        if throttle >= 0.6:
            throttle = 0.6
        print('throttle', throttle)
        return throttle

# NOTE(r7vme): Just a copy of default_categorical with zoomed input layer.
def dt_categorical():
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
    angle_out = Dense(15, activation='softmax', name='angle_out')(x)        # Connect every input with every output and output 15 hidden units. Use Softmax to give percentage. 15 categories and find best one based off percentage 0.0-1.0

    #continous output of throttle
    throttle_out = Dense(1, activation='relu', name='throttle_out')(x)      # Reduce to 1 number, Positive number only

    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])
    model.compile(optimizer='adam',
                  loss={'angle_out': 'categorical_crossentropy',
                        'throttle_out': 'mean_absolute_error'},
                  loss_weights={'angle_out': 0.9, 'throttle_out': .001})

    return model
