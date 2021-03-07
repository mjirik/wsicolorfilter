import scaffan.image as scim
import tkinter as tk
from tkinter import filedialog
import numpy as np
import sed3


def zoom_img(img, stride=10):
    """Use sed3 tool to manually zoom to wanted image region.

    :parameter stride: for image view (Displaying image in full resolution is both resource and time consuming.)
    """
    # show sed3 tool for region selection
    ed = sed3.sed3(img[::stride, ::stride, :])
    ed.show()

    # save user selected region
    nzx, nzy, nzz = np.nonzero(ed.seeds)

    # crop original image
    img = img[
          np.min(nzx) * stride:np.max(nzx) * stride,
          np.min(nzy) * stride:np.max(nzy) * stride,
          :
          ]
    return img


def load_image(f_path=None):
    """Load and return .czi image from given path."""

    if f_path is None:
        root = tk.Tk()
        root.withdraw()

        f_path = filedialog.askopenfilename()

    anim = scim.AnnotatedImage(str(f_path))
    img = anim.get_full_view().get_region_image()

    img = zoom_img(img)

    return img


def hue_to_continuous_2d(img):
    """Takes hsv image and returns ´hhsv´ format, where hue is replaced by 2 values - sin(hue) and cos(hue)"""
    hue = np.expand_dims(img[:, :, 0], -1)
    hue_x = np.cos(hue * 2 * np.pi)
    hue_y = np.sin(hue * 2 * np.pi)
    img = np.concatenate((hue_x, hue_y, img[:, :, 1:]), axis=-1)
    return img


def is_valid_input(inpt):
    if not inpt.isdigit():
        return False
    inpt = int(inpt)
    return 0 < inpt < 3
