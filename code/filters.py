import numpy as np
from numpy import pi, exp, sqrt
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import resize
import matplotlib.pyplot as plt
import cv2 


def apply_red_circle(image, circle, model):
    # check for 4-channel image & resize to appropriate size
    print(circle)
    if len(circle.shape) > 2 and circle.shape[2] == 4:
        circle = cv2.cvtColor(circle, cv2.COLOR_BGRA2BGR)
    circle = resize(circle, (10, 10, 3))
    image = np.expand_dims(image, axis=0)
    colored_img = np.squeeze(cv2.merge((image, image, image)))
    # print(colored_img.shape)
    # colored_img = cv2.cvtColor(colored_img, cv2.COLOR_BGR2GRA
    output = model.predict(image)
    nose_tip_x = int(48*output[0, 20]+48)
    nose_tip_y = int(48*output[0, 21]+48)
    colored_img[nose_tip_y, nose_tip_x, :] = [255, 0, 0]
    plt.imshow(colored_img)
    plt.show()

    for x in range(0, 10):
        for y in range(0, 10):
            pixel_g = circle[y][x][1]
            pixel_b = circle[y][x][2]
            if pixel_g<0.5 and pixel_b<0.5: # red part
                # print('applying filter at ', x, " and ", y)
                colored_img[nose_tip_y-5+y, nose_tip_x-5+x, :] = circle[y, x, :]

    plt.imshow(colored_img)
    plt.show()

    

