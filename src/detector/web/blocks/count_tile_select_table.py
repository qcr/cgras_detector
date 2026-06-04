# Copyright 2024 - Andrew Kwok Fai LUI,
# Robotics and Autonomous Systems Group, REF, RI
# and the Queensland University of Technology

__author__ = 'Andrew Lui'
__copyright__ = 'Copyright 2024'
__license__ = 'GPL'
__version__ = '1.0'
__email__ = 'ak.lui@qut.edu.au'
__status__ = 'Development'

import numbers
import pandas as pd
# dash modules
import dash
from dash import html, dcc, Input, Output, State, dash_table, ctx
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from detector.model import DETECT_DAO
from cgras_datatools.logging_tools import logger

class CountTileSelectTable():
    def __init__(self, app, prefix, page_size=25):
        self.app = app
        self.prefix = prefix = prefix + 'ctst_'
        self.update_panel_store_id = self.prefix + 'update_panel_store'
        self.update_datatable_store_id = self.prefix + 'update_datatable_store'
        self.row_selected_trigger_id = self.prefix + 'row_selected_store'
        self._model = None
        self._operators = [['ge ', '>='],
                    ['le ', '<='],
                    ['lt ', '<'],
                    ['gt ', '>'],
                    ['ne ', '!='],
                    ['eq ', '='],
                    ['contains '],
                    ['datestartswith ']]
        self._columns = [{'name': 'Tile ID',    'id': 'tile_id',    'type': 'text',     'editable': False},
                         {'name': 'Species',    'id': 'species',    'type': 'text',     'editable': False},
                         {'name': 'Settled On', 'id': 'settle_time','type': 'datetime', 'editable': False},
                         ]
        self._datatable = dash_table.DataTable(
            id=prefix+'datatable', columns=self._columns, style_header={}, fill_width=True,
            page_current=0, page_size=page_size, page_action='custom',
            filter_action='custom', filter_query='', row_selectable=False,
            cell_selectable=True, row_deletable=False, style_cell={'fontSize': 14})

        # --- Filter bar ---
        species_dropdown = dcc.Dropdown(
            id=prefix+'species_filter', multi=True,
            placeholder='All species', clearable=True,
            style={'width': '100%'})

        season_dropdown = dcc.Dropdown(
            id=prefix+'season_filter', multi=True,
            placeholder='All seasons', clearable=True,
            style={'width': '100%'})

        tile_id_input = dcc.Input(
            id=prefix+'tile_id_filter', type='text',
            placeholder='Tile ID contains…', debounce=True,
            style={'width': '100%', 'height': '36px', 'padding': '4px 8px'})

        date_picker_start = dcc.DatePickerSingle(
            id=prefix+'date_start', placeholder='Start date',
            clearable=True, display_format='YYYY-MM-DD',
            style={'fontSize': '13px'})

        date_picker_end = dcc.DatePickerSingle(
            id=prefix+'date_end', placeholder='End date',
            clearable=True, display_format='YYYY-MM-DD',
            style={'fontSize': '13px'})

        filter_card = dbc.Card(dbc.CardBody([
            dbc.Row(dbc.Col([html.Label('Species',       className='fw-semibold mb-1 small'), species_dropdown],    width=12), className='mb-2'),
            dbc.Row(dbc.Col([html.Label('Season',        className='fw-semibold mb-1 small'), season_dropdown],     width=12), className='mb-2'),
            dbc.Row(dbc.Col([html.Label('Tile ID',       className='fw-semibold mb-1 small'), tile_id_input],       width=12), className='mb-2'),
            dbc.Row(dbc.Col([html.Label('Capture Start', className='fw-semibold mb-1 small'), date_picker_start],   width=12), className='mb-2'),
            dbc.Row(dbc.Col([html.Label('Capture End',   className='fw-semibold mb-1 small'), date_picker_end],     width=12)),
        ]), className='mb-2 shadow-sm', style={'fontSize': '13px'})

        self.the_panel = html.Div([
                html.H4([dbc.Badge('TILE BROWSER', color='white', text_color='secondary')]),
                filter_card,
                dbc.Row(html.Div(self._datatable)),
                dcc.Store(id=self.update_panel_store_id),
                dcc.Store(id=self.update_datatable_store_id),
                dcc.Store(id=self.row_selected_trigger_id),
            ], id=prefix+'main_panel', style={'margin-top': '24px'})

        # Populate filter dropdowns on page load
        self.app.callback(
            [Output(prefix+'species_filter', 'options'),
             Output(prefix+'season_filter',  'options')],
            [Input(prefix+'main_panel', 'children')],
            prevent_initial_call='initial_duplicate'
        )(self._cb_populate_filters())

        # Any filter change -> pack into store -> triggers datatable refresh
        self.app.callback(
            [Output(self.update_datatable_store_id, 'data', allow_duplicate=True)],
            [Input(prefix+'species_filter', 'value'),
             Input(prefix+'season_filter',  'value'),
             Input(prefix+'tile_id_filter', 'value'),
             Input(prefix+'date_start',     'date'),
             Input(prefix+'date_end',       'date')],
            prevent_initial_call='initial_duplicate'
        )(self._cb_filter_changed())

        # Datatable data — triggered by pagination/filter/sort or store change
        self.app.callback(
            [Output(self.prefix+'datatable', 'data'),
             Output(prefix+'datatable', 'selected_rows', allow_duplicate=True)],
            [Input(self.prefix+'datatable', 'page_current'),
             Input(self.prefix+'datatable', 'page_size'),
             Input(self.prefix+'datatable', 'sort_by'),
             Input(self.prefix+'datatable', 'filter_query'),
             Input(self.update_datatable_store_id, 'data')],
            prevent_initial_call='initial_duplicate'
        )(self._update_datatable())

        # Cell click -> row highlight + downstream tile-selected trigger
        self.app.callback(
            [Output(self.row_selected_trigger_id, 'data'),
             Output(prefix+'datatable', 'selected_cells',        allow_duplicate=True),
             Output(prefix+'datatable', 'active_cell',           allow_duplicate=True),
             Output(prefix+'datatable', 'style_data_conditional',allow_duplicate=True)],
            [Input(prefix+'datatable', 'active_cell'),
             State(self.prefix+'datatable', 'data'),
             State(self.prefix+'datatable', 'page_current'),
             State(self.prefix+'datatable', 'page_size')],
            prevent_initial_call=True
        )(self._cb_cell_selected())

    def get_panel(self):
        return self.the_panel

    def get_row_selected_trigger_id(self) -> str:
        return self.row_selected_trigger_id

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------
    def _get_filtered_tiles(self, species_list, season_list, tile_id_filter, date_start, date_end):
        df = DETECT_DAO.query_processed_tile_samples(the_period=0)
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=['tile_id', 'species', 'settle_time'])
        if species_list:
            df = df[df['species'].isin(species_list)]
        if season_list:
            df = df[df['season'].isin(season_list)]
        if tile_id_filter:
            df = df[df['tile_id'].str.contains(tile_id_filter, case=False, na=False)]
        if date_start:
            df = df[df['batch_time'].str[:10] >= date_start]
        if date_end:
            df = df[df['batch_time'].str[:10] <= date_end]
        tiles = df.groupby('tile_id').agg(
            species=('species', 'first'),
            settle_time=('settle_time', 'first'),
        ).reset_index()
        return tiles[['tile_id', 'species', 'settle_time']].sort_values(
            ['species', 'tile_id']).reset_index(drop=True)

    def split_filter_part(self, filter_part):
        for operator_type in self._operators:
            for operator in operator_type:
                if operator in filter_part:
                    name_part, value_part = filter_part.split(operator, 1)
                    name = name_part[name_part.find('{') + 1: name_part.rfind('}')]
                    value_part = value_part.strip()
                    v0 = value_part[0]
                    if (v0 == value_part[-1] and v0 in ("'", '"', '`')):
                        value = value_part[1: -1].replace('\\' + v0, v0)
                    else:
                        try:
                            value = int(value_part)
                        except ValueError:
                            value = value_part
                    return name, operator_type[0].strip(), value
        return [None] * 3

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _cb_populate_filters(self):
        def populate_filters(_):
            species = [{'label': s, 'value': s}
                       for s in DETECT_DAO.list_species_in_tile_sample()]
            seasons = [{'label': f'{s} Season', 'value': s}
                       for s in DETECT_DAO.list_seasons_in_tile_sample()]
            return species, seasons
        return populate_filters

    def _cb_filter_changed(self):
        def filter_changed(species_list, season_list, tile_id_filter, date_start, date_end):
            return ({'species': species_list, 'seasons': season_list,
                     'tile_id': tile_id_filter,
                     'date_start': date_start, 'date_end': date_end},)
        return filter_changed

    def _update_datatable(self):
        def update_datatable(page_current, page_size, sort_by, filter_query, store):
            species_list = season_list = tile_id_filter = date_start = date_end = None
            if isinstance(store, dict):
                species_list  = store.get('species')
                season_list   = store.get('seasons')
                tile_id_filter = store.get('tile_id')
                date_start    = store.get('date_start')
                date_end      = store.get('date_end')

            self._model = self._get_filtered_tiles(
                species_list, season_list, tile_id_filter, date_start, date_end)
            model = self._model

            # Apply Dash DataTable column-level filters
            for filter_part in (filter_query or '').split(' && '):
                col_name, operator, filter_value = self.split_filter_part(filter_part)
                if col_name is None:
                    continue
                if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
                    model = model.loc[getattr(model[col_name], operator)(filter_value)]
                elif operator == 'contains':
                    if isinstance(filter_value, numbers.Number):
                        filter_value = str(filter_value)
                    model = model.loc[model[col_name].str.contains(filter_value)]
                elif operator == 'datestartswith':
                    model = model.loc[model[col_name].str.startswith(filter_value)]

            page_start = page_current * page_size
            return (model.iloc[page_start: page_start + page_size].to_dict('records'), [])
        return update_datatable

    def _cb_cell_selected(self):
        def cb_cell_selected(active_cell, model, page_current, page_size):
            if active_cell is None:
                raise PreventUpdate
            row = active_cell['row']
            tile_id = model[row]['tile_id']
            style_data_conditional = [
                {'if': {'filter_query': "{{tile_id}} = '{}'".format(tile_id)},
                 'backgroundColor': 'yellow'}
            ]
            return (tile_id, [], None, style_data_conditional)
        return cb_cell_selected
