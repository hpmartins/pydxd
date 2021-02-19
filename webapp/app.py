# -*- coding: utf-8 -*-

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_table
import plotly.express as px
import pandas as pd
import pkg_resources

from callbacks import register_callbacks

meta_viewport = {"name": "viewport", "content": "width=device-width, initial-scale=1, shrink-to-fit=no"}
    

app = dash.Dash(__name__, 
                meta_tags=[meta_viewport],
                external_stylesheets = [dbc.themes.BOOTSTRAP]
)

# app.enable_dev_tools(
#         dev_tools_ui=True,
#         dev_tools_serve_dev_bundles=True,
# )

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


CIF_LIST = pkg_resources.resource_listdir('pydxd', "data/cif")
select_cif = select = dbc.Select(
    id="cif_select",
    options=[{"label": x, "value": x} for x in CIF_LIST],
    value = 'Cu_mp-30_symmetrized.cif',
    persistence = True, persistence_type = 'session',
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
    style_header= {'fontWeight': 'bold'},
    style_cell  = {'font-family': 'sans-serif', 'textAlign': 'center', 'whiteSpace': 'normal', 'font-size': '11pt'},
    id='sites_table',
)

app.layout = dbc.Container(
    [
        topbar,
                
        html.Div([
            html.Hr(),
            dbc.Container(
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
                                        dbc.Input(
                                            type="numeric", 
                                            id={'type': 'hkl', 'index': 'h'}, 
                                            placeholder="h", 
                                            value=1,
                                            persistence = True, persistence_type = 'session'
                                        ),
                                        dbc.Input(
                                            type="numeric", 
                                            id={'type': 'hkl', 'index': 'k'}, 
                                            placeholder="k", 
                                            value=1,
                                            persistence = True, persistence_type = 'session'
                                        ),
                                        dbc.Input(
                                            type="numeric", 
                                            id={'type': 'hkl', 'index': 'l'}, 
                                            placeholder="l", 
                                            value=1,
                                            persistence = True, persistence_type = 'session'
                                        ),
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
                                        persistence = True, persistence_type = 'session'
                                    ),
                                ], className = "my-auto", width=3),
                                dbc.Label("Energy (eV)", id = "scanmode_fixed_label", width=3),
                                dbc.Col([
                                    dbc.Input(
                                        type="numeric", 
                                        id="scanmode_fixed_value", 
                                        value = 15000,
                                        bs_size = "sm",
                                        persistence = True, persistence_type = 'session',
                                    )
                                ], className = "my-auto", width=3),
                            ],
                            row=True,
                        ),
                        dbc.FormGroup(
                            [
                                dbc.Label("Δx", width=3),
                                dbc.Col([
                                    dbc.Input(
                                        type="numeric", 
                                        id="xrange_delta", 
                                        value = 5,
                                        bs_size = "sm",
                                        persistence = True, persistence_type = 'session',
                                    ),
                                ], className = "my-auto", width=3),
                                dbc.Label("# points", width=3),
                                dbc.Col([
                                    dbc.Input(
                                        type="numeric", 
                                        id="xrange_npts", 
                                        value = 501,
                                        bs_size = "sm",
                                        persistence = True, persistence_type = 'session',
                                    )
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
            ),
            html.Hr(),
            dbc.Alert(
                id = 'results_alert',
                is_open = False,
                duration = 2000,
            ),
            dbc.Container([
                dbc.Row(
                    [
                        # Controls and calculated parameters
                        dbc.Col([
                            dbc.Button('Calculate', id = 'results_calculate_button'),
                        ], width = 4),
                        dbc.Col(
                            [
                                dbc.Row([
                                    dbc.Col(dcc.Markdown('d<sub>hkl</sub>', dangerously_allow_html = True), width = 2),
                                    dbc.Col(id = {'type': 'results_parameter', 'index': 'd_hkl'}, width=10),
                                ], className = 'rowresults'),
                                dbc.Row([
                                    dbc.Col(dcc.Markdown('Θ<sub>B</sub>', dangerously_allow_html = True), width = 2),
                                    dbc.Col(id = {'type': 'results_parameter', 'index': 'Bragg_angle'}, width=10),
                                ], className = 'rowresults'),
                                dbc.Row([
                                    dbc.Col(dcc.Markdown('E<sub>B</sub>', dangerously_allow_html = True), width = 2),
                                    dbc.Col(id = {'type': 'results_parameter', 'index': 'Bragg_energy'}, width=10),
                                ], className = 'rowresults'),
                                dbc.Row([
                                    dbc.Col(dcc.Markdown('λ<sub>B</sub>', dangerously_allow_html = True), width = 2),
                                    dbc.Col(id = {'type': 'results_parameter', 'index': 'Bragg_wavelength'}, width=10),
                                ], className = 'rowresults'),
                            ], 
                            width = 4,
                        ),
                        dbc.Col(
                            [
                                dbc.Row([
                                    dbc.Col(dcc.Markdown('F<sub>0</sub>', dangerously_allow_html = True), width = 2),
                                    dbc.Col(id = {'type': 'results_parameter', 'index': 'F_0'}, width=10),
                                ], className = 'rowresults'),
                                dbc.Row([
                                    dbc.Col(dcc.Markdown('F<sub>h</sub>', dangerously_allow_html = True), width = 2),
                                    dbc.Col(id = {'type': 'results_parameter', 'index': 'F_H'}, width=10),
                                ], className = 'rowresults'),
                                dbc.Row([
                                    dbc.Col(dcc.Markdown('F<sub>hb</sub>', dangerously_allow_html = True), width = 2),
                                    dbc.Col(id = {'type': 'results_parameter', 'index': 'F_Hb'}, width=10),
                                ], className = 'rowresults'),
                                dbc.Row([
                                    dbc.Col(dcc.Markdown('φ<sub>h</sub>', dangerously_allow_html = True), width = 2),
                                    dbc.Col(id = {'type': 'results_parameter', 'index': 'Angle_F_H'}, width=10),
                                ], className = 'rowresults'),
                            ], 
                            width = 4,
                        ),
                    ],
                    className = "pb-2",
                ),
                dbc.Row([
                    # Refl, Phase
                    dbc.Col([
                        html.Div(id = 'results_figure_refl')
                    ], width = 6),
                    # EF
                    dbc.Col([
                        html.Div(id = 'results_figure_elf')
                    ], width = 6),
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