"""Script for manual dataset creation"""
import tkinter as tk
from tkinter import filedialog

import numpy as np
import scaffan.image as scim
import sed3
from skimage.color import rgb2hsv
from data_operations import load_image, hue_to_continuous_2d


def annotate(img):
    """Use sed3 tool to annotate data from given image."""
    ed = sed3.sed3(img)
    ed.show()

    seeds = ed.seeds
    seeds = seeds[:, :, 0]
    seeds = seeds.astype('int8')

    return seeds


def run_annotation():
    """Run the annotation process."""
    img = load_image()
    seeds = annotate(img)

    img = rgb2hsv(img)
    img = hue_to_continuous_2d(img)

    np.save('seeds.npy', seeds)
    np.save('image.npy', img)


if __name__ == '__main__':
    run_annotation()
