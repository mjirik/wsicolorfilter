import logging
import numpy as np
from skimage.color import rgb2hsv

from matplotlib import pyplot as plt

from wsicolorfilter.annotation import hue_to_continuous_2d
from wsicolorfilter.data_operations import load_image
from wsicolorfilter.svm_filter import SvmFilter
from matplotlib.widgets import Button

import random

logger = logging.getLogger('predict')

model = SvmFilter()
model.load_model()

img = load_image()
img_hhsv = rgb2hsv(img)
img_hhsv = hue_to_continuous_2d(img_hhsv)
img = img.astype('float32') / 255.0

result = model.predict(img_hhsv)

fig = plt.figure()

ax_plus = plt.axes([0.7, 0.05, 0.1, 0.075])
ax_minus = plt.axes([0.81, 0.05, 0.1, 0.075])
ax_back = plt.axes([0.2, 0.2, 0.6, 0.6])

plt.imshow(
    img * np.stack([result == 0, result == 0, result == 0], axis=-1) + np.stack([result != 0, result != 0, result != 0],
                                                                                axis=-1).astype('float32'))

btn_p = Button(ax_plus, "+")
btn_m = Button(ax_minus, "-")


def btn_plus(event):
    model.model.coef_[:, -1] *= 1.2

    result = model.predict(img_hhsv)

    plt.imshow(img * np.stack([result == 0, result == 0, result == 0], axis=-1) + np.stack(
        [result != 0, result != 0, result != 0],
        axis=-1).astype('float32'))

    plt.title(str(random.getrandbits(32)))
    plt.show()


def btn_minus(event):
    model.model.coef_[:, -1] *= 0.8

    result = model.predict(img_hhsv)

    plt.imshow(img * np.stack([result == 0, result == 0, result == 0], axis=-1) + np.stack(
        [result != 0, result != 0, result != 0],
        axis=-1).astype('float32'))

    plt.title(str(random.getrandbits(32)))
    plt.show()


if __name__ == '__main__':
    btn_p.on_clicked(btn_plus)
    btn_m.on_clicked(btn_minus)

    plt.title(str(random.getrandbits(32)))
    plt.show()
