# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 22:12:19 2020

@author: hpmar
"""

import pyxro
import numpy as np


Sample = pyxro.MultilayerSample()


Substrate = {
    'Formula': 'SrTiO3',
    'Name': 'STO',
    'Density': 3.0,
}

Layer_top = {
    'Formula': 'C',
    'Name': 'C',
    'Density': 1.0,
    'Thickness': 4.0,
}

Layer_BFO = {
    'Formula': 'BiFeO3',
    'Name': 'BFO',
    'Density': 8.34,
    'Thickness': 23.88,
}

Layer_LSMO = {
    'Formula': 'La0.7Sr0.3MnO3',
    'Name': 'LSMO',
    'Density': 6.63,
    'Thickness': 23.22,
}

Sample.set_substrate(Substrate)
Sample.set_vacuum()

for i in np.arange(10):
    Sample.add_layer(Layer_BFO)
    Sample.add_layer(Layer_LSMO)
    
print(len(Sample.layers))