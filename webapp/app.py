# -*- coding: utf-8 -*-

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_table
import plotly.express as px
import pandas as pd
import pkg_resources
import glob

from callbacks import register_callbacks

app = dash.Dash(__name__, external_stylesheets = [dbc.themes.BOOTSTRAP])

app.enable_dev_tools(
        dev_tools_ui=True,
        dev_tools_serve_dev_bundles=True,
)

df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})
fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

topbar = dbc.Navbar(
    [
        dbc.Row(
            [
                dbc.Col(dbc.NavbarBrand("Dynamical X-ray Diffraction")),
            ],
            align="center",
            no_gutters=True,
            className = "mr-auto pr-2"
        ),
], className = "mb-2", sticky = "top")


CIF_LIST = pkg_resources.resource_listdir('pydtd', "data/cif")
select_cif = select = dbc.Select(
    id="cif_select",
    options=[{"label": x, "value": x} for x in CIF_LIST],
    value = 'Cu_mp-30_symmetrized.cif',
)

sites_table = dash_table.DataTable(
    columns = [
        {
            'id': 'Z',
            'name': 'Z',
        },
        {
            'id': 'name',
            'name': 'Atom',
        },
        {
            'id': 'label',
            'name': 'Label',
        },
        {
            'id': 'zcoord',
            'name': 'z',
            'type': 'numeric',
            'format': {'specifier': '.4f'},
        },
        {
            'id': 'Hdotr',
            'name': 'H·r',
            'type': 'numeric',
            'format': {'specifier': '.4f'},
        },
        {
            'id': 'cohpos',
            'name': 'CP',
            'type': 'numeric',
            'format': {'specifier': '.4f'},
        },
    ],
    data = [],
    editable = False,
    page_action = 'none',
    # style_table={'height': '210px', 'overflowY': 'scroll'},
    # style_cell={
    #     'font-family': 'sans-serif',
    #     'font-size': '9pt',
    #     'overflow': 'hidden',
    #     'textOverflow': 'ellipsis',
    #     'maxWidth': 0
    # },
    # style_table = {'font-family': 'sans-serif', 'font-size': '9pt'},#, 'height': '200px', 'overflowY': 'auto'},
    style_header= {'fontWeight': 'bold'},
    style_cell  = {'font-family': 'sans-serif', 'textAlign': 'center', 'whiteSpace': 'normal', 'font-size': '9pt'},
    # style_cell_conditional=[
    #     {'if': {'column_id': 'Z'}, 'width': '20px'},
    #     {'if': {'column_id': 'name'}, 'width': '30px'},
    #     {'if': {'column_id': 'label'}, 'width': '30px'},
    #     {'if': {'column_id': 'zcoord'}, 'width': '40px'},
    #     {'if': {'column_id': 'Hdotr'}, 'width': '30px'},
    #     {'if': {'column_id': 'cohpos'}, 'width': '30px'},
    # ],
    id='sites_table',
)

app.layout = dbc.Container(
    [
        topbar,
        
        html.Div([
            html.Hr(),
            dbc.Row([
                # Inputs
                dbc.Col([
                    dbc.Form([
                        dbc.FormGroup(
                            [
                                dbc.Label("CIF", width=3),
                                dbc.Col(select_cif, width=9),
                            ],
                            row=True,
                        ),
                    ]),
                    dbc.FormGroup(
                        [
                            dbc.Label("(hkl)", width=3),
                            dbc.Col([
                                dbc.InputGroup([
                                    dbc.Input(type="numeric", id={'type': 'hkl', 'index': 'h'}, placeholder="h", value=1),
                                    dbc.Input(type="numeric", id={'type': 'hkl', 'index': 'k'}, placeholder="k", value=1),
                                    dbc.Input(type="numeric", id={'type': 'hkl', 'index': 'l'}, placeholder="l", value=1),
                                ], size="sm")
                            ], className = "my-auto", width=9),
                        ],
                        row=True,
                    ),
                    dbc.FormGroup(
                        [
                            dbc.Label("Scan mode", width=3),
                            dbc.Col([
                                dbc.RadioItems(
                                    id="scanmode_select",
                                    options=[
                                        {"label": "Angle",  "value": 'angle'},
                                        {"label": "Energy", "value": 'energy'},
                                    ],
                                    value = 'angle',
                                ),
                            ], className = "my-auto", width=3),
                            dbc.Label("Energy (eV)", id = "scanmode_fixed_label", width=3),
                            dbc.Col([
                                dbc.Input(type="numeric", id="scanmode_fixed_value")
                            ], className = "my-auto", width=3),
                        ],
                        row=True,
                    ),
                    dbc.FormGroup(
                        [
                            dbc.Label("Δx", width=3),
                            dbc.Col([
                                dbc.Input(type="numeric", id="xrange_delta", value = 30),
                            ], className = "my-auto", width=3),
                            dbc.Label("# points", width=3),
                            dbc.Col([
                                dbc.Input(type="numeric", id="xrange_npts", value = 501)
                            ], className = "my-auto", width=3),
                        ],
                        row=True,
                    ),
                ], width = 6),
                # Sites table
                dbc.Col([
                    html.Div(sites_table, style = {'overflowY': 'auto', 'height': '210px'}),
                ]),
            ]),
            html.Hr(),
            dbc.Row([
                # Refl, Phase
                dbc.Col([
                    html.Div(id = 'figure_refl')
                ]),
                # EF
                dbc.Col([
                    html.Div(id = 'figure_elf')
                ]),
            ]),
            dbc.Row([
                # RC figure
                dbc.Col([
                ]),
                # RC controls
                dbc.Col([
                ]),
            ]),
        ])
    ]
)

register_callbacks(app)     
             
if __name__ == '__main__':
    app.run_server(debug=True, port=5666)