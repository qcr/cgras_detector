# Copyright 2024 - Andrew Kwok Fai LUI,
# Robotics and Autonomous Systems Group, REF, RI
# and the Queensland University of Technology

__author__ = 'Andrew Lui'
__copyright__ = 'Copyright 2024'
__license__ = 'GPL'
__version__ = '1.0'
__email__ = 'ak.lui@qut.edu.au'
__status__ = 'Development'

import time
from collections import OrderedDict
import pandas as pd
# dash modules
import dash
from dash import html, dcc, Input, Output, State, dash_table, ctx
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from cgras_datatools.logging_tools import logger
from detector.model import DETECT_DAO
from detector.dao_detect import SampleStatusNames

class TileSampleSearchBlock():
    def __init__(self, app, prefix):
        self.app = app
        self.prefix = prefix = prefix + 'tss_'
        self.search_clicked_trigger_id = prefix + 'search_query_store'
        self.external_trigger_id = prefix + 'external_trigger_store'
        # define widgets
        tile_id_textbox = dcc.Input(id=prefix+'tile_id_input', type='text', placeholder='Tile ID',
                                    style={'width': '180px', 'height': '36px', 'padding': '4px 8px'})
        batch_id_textbox = dcc.Input(id=prefix+'batch_id_input', type='text', placeholder='Batch ID',
                                     style={'width': '180px', 'height': '36px', 'padding': '4px 8px'})
        # the species filter — zIndex 50: opens above all other dropdowns in this panel
        species_dropdown = dcc.Dropdown(id=prefix+'species_dropdown', multi=True,
                                        placeholder='All species', clearable=True,
                                        style={'minWidth': '180px', 'zIndex': 50})
        # the period filter
        self.period_options = [
            {'label': 'Past Day', 'value': -1},
            {'label': 'Past Week', 'value': -7},
            {'label': 'Past Month', 'value': -31},
            {'label': 'All Time', 'value': 0},
        ]
        period_dropdown = dcc.Dropdown(id=prefix+'period_dropdown',
                                       searchable=False, clearable=False,
                                       style={'width': '180px', 'zIndex': 40})
        # the status filter
        self.status_options = [
            {'label': 'All', 'value': SampleStatusNames.ALL.value},
            {'label': 'Rejected or Flagged', 'value': SampleStatusNames.REJECTED.value},
            {'label': 'Done', 'value': SampleStatusNames.DONE.value},
            # {'label': 'Flagged', 'value': SampleStatusNames.FLAGGED.value},
            # {'label': 'Rejected', 'value': SampleStatusNames.REJECTED.value},
        ]
        status_dropdown = dcc.Dropdown(options=self.status_options, id=prefix+'status_dropdown',
                                       searchable=False, clearable=False,
                                       style={'width': '180px', 'zIndex': 30})
        # the page size filter
        self.pagesize_options = [
            {'label': 'Recent 50', 'value': 50},
            {'label': 'Recent 100', 'value': 100},
            {'label': 'Recent 150', 'value': 150},
            {'label': 'Recent 250', 'value': 250},]
        self.DEFAULT_PAGE_SIZE = 100
        pagesize_dropdown = dcc.Dropdown(options=self.pagesize_options, id=prefix+'pagesize_dropdown', value=100,
                                         searchable=False, clearable=False, maxHeight=200,
                                         style={'width': '140px', 'zIndex': 20})

        filter_card = dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col([html.Label('Tile ID',         className='fw-semibold d-block mb-1'), tile_id_textbox],  className='col-auto'),
                dbc.Col([html.Label('Batch ID',        className='fw-semibold d-block mb-1'), batch_id_textbox], className='col-auto'),
                dbc.Col([html.Label('Species',         className='fw-semibold mb-1'),         species_dropdown],  className='col-auto'),
                dbc.Col([html.Label('Period / Season', className='fw-semibold mb-1'),         period_dropdown],   className='col-auto'),
                dbc.Col([html.Label('Status',          className='fw-semibold mb-1'),         status_dropdown],   className='col-auto'),
                dbc.Col([html.Label('Results',         className='fw-semibold mb-1'),         pagesize_dropdown], className='col-auto'),
                dbc.Col([
                    html.Label(' ', className='d-block mb-1'),
                    dbc.Button('Reset', id=prefix+'reset_filter_button', n_clicks=0,
                               color='secondary', size='sm', style={'width': '80px'}),
                ], className='col-auto'),
                dcc.Store(id=self.external_trigger_id),
            ], className='g-3 align-items-end'),
        ]), className='shadow-sm')

        self.tile_sample_search_panel = html.Div([
                dcc.Store(id=self.search_clicked_trigger_id),
                html.H4(dbc.Badge('SEARCH PROCESSED TILE SAMPLES', className='ms-1 me-2', color='white', text_color='secondary'),
                        className='mb-3'),
                filter_card,
            ], id=prefix+'main_panel', className='mx-auto', style={'padding': '12px', 'zIndex': 10})

        self.app.callback([Output(prefix+'tile_id_input', 'value', allow_duplicate=True),
                           Output(prefix+'batch_id_input', 'value', allow_duplicate=True),
                           Output(prefix+'period_dropdown', 'options', allow_duplicate=True),
                           Output(prefix+'period_dropdown', 'value', allow_duplicate=True),
                           Output(prefix+'pagesize_dropdown', 'value', allow_duplicate=True),
                           Output(prefix+'status_dropdown', 'value', allow_duplicate=True),
                           Output(prefix+'reset_filter_button', 'n_clicks'),
                           Output(prefix+'species_dropdown', 'options', allow_duplicate=True),
                           Output(prefix+'species_dropdown', 'value', allow_duplicate=True),
                           ],
                            [
                            Input(prefix+'reset_filter_button', 'n_clicks'),
                            Input(prefix+'main_panel', 'children'),
                            ], prevent_initial_call='initial_duplicate')(self._reset_filter_button_clicked())

        self.app.callback([Output(prefix+'search_query_store', 'data')],
                            [
                            Input(prefix+'tile_id_input', 'value'),
                            Input(prefix+'batch_id_input', 'value'),
                            Input(prefix+'period_dropdown', 'value'),
                            Input(prefix+'pagesize_dropdown', 'value'),
                            Input(prefix+'status_dropdown', 'value'),
                            Input(prefix+'species_dropdown', 'value'),
                            Input(self.external_trigger_id, 'data'),
                            ], prevent_initial_call=True)(self._refresh_table_clicked())


    def get_panel(self):
        return self.tile_sample_search_panel

    def register_trigger(self, trigger_id:str):
        # define callbacks for the datatable data
        self.app.callback([Output(self.external_trigger_id, 'data', allow_duplicate=True)],
            [Input(trigger_id, 'data')], prevent_initial_call=True, allow_duplicate=True)(self._external_triggered())

    def get_search_clicked_trigger_id(self):
        return self.search_clicked_trigger_id

    def _reset_filter_button_clicked(self):
        def reset_filter_button_clicked(n_clicks, _):
            if n_clicks <= 1:   # n_clicks is 0 if the callback is triggered by page loading
                period_options = self.period_options
                season_titles_list = DETECT_DAO.list_seasons_in_tile_sample()
                for season_title in season_titles_list:
                    period_options.append({'label': f'{season_title} Season', 'value': season_title})
                species_options = [{'label': s, 'value': s} for s in DETECT_DAO.list_species_in_tile_sample()]
                DEFAULT_PERIOD_VALUE = 0
                return ('', '', period_options, DEFAULT_PERIOD_VALUE, self.DEFAULT_PAGE_SIZE, SampleStatusNames.ALL.value, 0, species_options, None)
            time.sleep(1.0)
            return (dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, 0, dash.no_update, dash.no_update)
        return reset_filter_button_clicked

    def _refresh_table_clicked(self):
        def refresh_table_clicked(tile_id, batch_id, the_period, the_pagesize, the_status, species_list, store):
            button_id = ctx.triggered_id if ctx.triggered_id is not None else 'No clicks yet'
            # set default value at initialization
            if the_status == SampleStatusNames.REJECTED:
                the_status = [SampleStatusNames.REJECTED, SampleStatusNames.FLAGGED]
            else:
                the_status = None if the_status == SampleStatusNames.ALL.value or the_status is None else the_status
            the_pagesize = 10 if the_pagesize is None else the_pagesize
            if isinstance(the_period, str):
                season = the_period
                the_period = 0
            else:
                season = None
            # build the query structure; species_list is applied as a post-filter in TileSampleTable
            query = [season, the_status, tile_id, batch_id, the_period, the_pagesize, species_list]
            return (query,)
        return refresh_table_clicked

    def _external_triggered(self):
        def external_triggered(store):
            return (store,)
        return external_triggered
