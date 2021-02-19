# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 21:45:43 2021

@author: Henrique
"""

import numpy as np
import pandas as pd
import pkg_resources
import dash
import dash_core_components as dcc
from dash.dependencies import Input, Output, State, MATCH, ALL
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from pydxd import Crystal

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
            
        hkl = [int(x) for x in hkl]
        filepath = pkg_resources.resource_filename('pydxd', 'data/cif/{}'.format(file))
        dtdCrystal = Crystal(filepath, hkl = hkl)
                
        return dtdCrystal.sites.to_dict('records')


    @app.callback(
        [
            Output('results_alert', 'is_open'),
            Output('results_alert', 'children'),
            Output('results_figure_refl', 'children'),
            Output('results_figure_elf', 'children'),
            Output({'type': 'results_parameter', 'index': ALL}, 'children')
        ],
        Input('results_calculate_button', 'n_clicks'),
        [
            State('cif_select', 'value'),
            State({'type': 'hkl', 'index': ALL}, 'value'),
            State('scanmode_select', 'value'),
            State('scanmode_fixed_value', 'value'),
            State('xrange_delta', 'value'),
            State('xrange_npts', 'value')
        ],
        # prevent_initial_call = True,
    )
    def fill_results(nc_calculate,
                     cif_file, hkl, scanmode, scanmode_fixed_value, xrange_delta, xrange_npts):
        
        len_parameters = len(dash.callback_context.outputs_list[-1])
        
        if not cif_file:
            return (
                True, "No crystal selected",
                dash.no_update, dash.no_update, [dash.no_update]*len_parameters,
            )
        
        if not hkl or None in hkl:
            return (
                True, "(hkl) not set",
                dash.no_update, dash.no_update, [dash.no_update]*len_parameters,
            )
        
        if scanmode_fixed_value is None:
            if scanmode == 'angle':
                return (
                    True, "Fixed photon energy not defined",
                    dash.no_update, dash.no_update, [dash.no_update]*len_parameters,
                )
            elif scanmode == 'energy':
                return (
                    True, "Fixed incident angle not defined",
                    dash.no_update, dash.no_update, [dash.no_update]*len_parameters,
                )
            else:
                raise PreventUpdate
        
        if xrange_delta is None or float(xrange_delta) <= 0:
            return (
                True, "Δx has to be defined and above zero",
                dash.no_update, dash.no_update, [dash.no_update]*len_parameters,
            )
        
        if xrange_npts is None or int(xrange_npts) <= 0:
            return (
                True, "Number of points in x-axis has to be defined and above zero",
                dash.no_update, dash.no_update, [dash.no_update]*len_parameters,
            )
        
        hkl = [int(x) for x in hkl]
        filepath = pkg_resources.resource_filename('pydxd', 'data/cif/{}'.format(cif_file))
        dtdCrystal = Crystal(filepath, hkl = hkl)
        
        if scanmode == 'angle':
            dtdCrystal.set_mode(mode = 'angular', energy = float(scanmode_fixed_value))
        elif scanmode == 'energy':
            dtdCrystal.set_mode(mode = 'energy', angle = float(scanmode_fixed_value))
        
        dtdCrystal.set_structure_factor(dtdCrystal.Bragg.energy)
        dtdCrystal.calc_reflectivity(delta = float(xrange_delta), npts = int(xrange_npts))
        
        #
        # Results parameters
        #
        results_parameter_list = [
            '{:.4f} Å'.format(dtdCrystal.d_hkl), 
            '{:.4f}°'.format(dtdCrystal.Bragg.angle), 
            '{:.4f} eV'.format(dtdCrystal.Bragg.energy), 
            '{:.4f} Å'.format(dtdCrystal.Bragg.wavelength), 
            
            '{:.4f}'.format(dtdCrystal.F_0), 
            '{:.4f}'.format(dtdCrystal.F_H), 
            '{:.4f}'.format(dtdCrystal.F_Hb), 
            '{:.4f} rad'.format(np.angle(dtdCrystal.F_H)), 
        ]
    
        #
        # Reflectivity
        #
        Refl_figure = make_subplots(specs=[[{"secondary_y": True}]])
        
        Refl_figure.add_trace(
            go.Scatter(x = dtdCrystal.data.x, y = dtdCrystal.data.Refl, name="Reflectivity", mode='lines'),
            secondary_y=False,
        )
        
        Refl_figure.add_trace(
            go.Scatter(x = dtdCrystal.data.x, y = dtdCrystal.data.Phase, name="Phase", mode='lines'),
            secondary_y=True,
        )
        
        Refl_figure.update_xaxes(showgrid = False, zeroline = False)
        if scanmode == 'angle':
            Refl_figure.update_xaxes(
                title_text="Relative incident angle (°)", 
            )
        elif scanmode == 'energy':
            Refl_figure.update_xaxes(
                title_text="Relative photon energy (eV)", 
            )
            
        Refl_figure.update_yaxes(
            title_text="Reflectivity", 
            showgrid = False, 
            zeroline = False, 
            secondary_y = False
        )
        Refl_figure.update_yaxes(
            title_text="Phase (rad)", 
            showgrid = False, 
            zeroline = False, 
            secondary_y = True
        )
        
        Refl_figure.update_layout(
            margin=dict(l=0, r=0, b=0, t=0),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.93
            ),
        )
        
        #
        # Electric Field
        #
        ELF_zrange = np.linspace(0, dtdCrystal.z_dist, 301)
        ELF_xrange = dtdCrystal.x_range
        ELF_data = pd.DataFrame(0, index = ELF_zrange, columns = ELF_xrange)
        
        for z in ELF_data.index.values:
            ELF_data.loc[z, :] = dtdCrystal.calc_part_RC(1.0, z / dtdCrystal.d_hkl)
                    
        ELF_figure = go.Figure()
        
        ELF_figure.add_trace(
            go.Heatmap(x = ELF_xrange, y = ELF_zrange, z = ELF_data.values, colorbar=dict(title='E-field'))
        )
        
        for i in range(int(dtdCrystal.z_dist / dtdCrystal.d_hkl)+1):
            ELF_figure.add_trace(
                go.Scatter(
                    x = [ELF_xrange.min(), ELF_xrange.max()],
                    y = [i*dtdCrystal.d_hkl]*2,
                    mode = 'lines',
                    line=dict(width=1, color='black'), 
                    showlegend = False,
                )
            )
            
        group_by_element = dtdCrystal.sites.groupby('name')
        for name in group_by_element.groups.keys():
            group = group_by_element.get_group(name)
            to_plot = list(group.zcoord.values)
            ELF_figure.add_trace(
                go.Scatter(x = [ELF_xrange.mean()]*len(to_plot), y = to_plot, mode = 'markers', name = name)
            )

        ELF_figure.update_layout(
            margin=dict(l=0, r=0, b=0, t=0),
            xaxis = dict(
                range = [ELF_xrange.min(), ELF_xrange.max()],
            ),
            yaxis = dict(
                range = [dtdCrystal.z_dist, 0],
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            ),
        )
        
        print(ELF_figure.data)
        
        if scanmode == 'angle':
            ELF_figure.update_xaxes(title_text="Relative incident angle (°)")
        elif scanmode == 'energy':
            ELF_figure.update_xaxes(title_text="Relative photon energy (eV)")
            
        ELF_figure.update_yaxes(title_text="Depth (Å)")
        
        
        
        return (
            dash.no_update, dash.no_update,
            dcc.Graph(figure = Refl_figure, responsive = True),
            dcc.Graph(figure = ELF_figure,  responsive = True),
            results_parameter_list,
        )