# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 21:45:43 2021

@author: Henrique
"""

import numpy as np
import pandas as pd
from dash.dependencies import Input, Output, State, MATCH, ALL
from dash.exceptions import PreventUpdate
import pkg_resources

from pydtd import Crystal

def register_callbacks(app):
    
    @app.callback(
        Output('scanmode_fixed_label', 'children'),
        Input('scanmode_select', 'value'),
    )
    def scanmode_set_fixed_label(value):
        if value == 'angle':
            return 'Energy (eV)'
        elif value == 'energy':
            return 'Angle (°)'
        else:
            raise PreventUpdate
            
    @app.callback(
        Output('sites_table', 'data'),
        [
            Input('cif_select', 'value'),
            Input({'type': 'hkl', 'index': ALL}, 'value')
        ],
    )
    def cifselect_fill_table(file, hkl):
        if file is None or not all(hkl):
            raise PreventUpdate
            
        filepath = pkg_resources.resource_filename('pydtd', 'data/cif/{}'.format(file))
        dtdCrystal = Crystal(filepath, hkl = hkl)
                
        return dtdCrystal.sites.to_dict('records')
