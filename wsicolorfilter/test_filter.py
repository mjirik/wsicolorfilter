import unittest

import numpy as np
import pandas as pd

from wsicolorfilter.nbg_filter import NaiveBayesFilter
from wsicolorfilter.data_operations import load_image
from wsicolorfilter.nearest_neighbor_filter import NearestNeighborFilter
from wsicolorfilter.svm_filter import SvmFilter
from wsicolorfilter.visualization import show_visual

models = [NearestNeighborFilter(), SvmFilter(), NaiveBayesFilter()]


class TestFilter(unittest.TestCase):
    """"""

    def model_test(self, model):
        """"""
        img = np.load('wsicolorfilter/image.npy')
        train_data = pd.read_csv('wsicolorfilter/train_data.csv')

        x = train_data[['cos(hue)', 'sin(hue)', 'saturation', 'value']]
        y = train_data['label']

        x = x.to_numpy()
        y = y.to_numpy().astype('int8')

        model.train_model(x, y)
        output = model.predict(img)
        model.save_model()
        model.load_model()

    def test_all_models(self):
        for model in models:
            self.model_test(model)

    def test_visual_comparsion(self):
        img = load_image()

        model_names = ['Nearest Neighbor', 'SVM', 'Naive Bayes Gaussian']

        for i in range(len(models)):
            show_visual(img, models[i], model_names[i])
