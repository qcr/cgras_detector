# Copyright 2024 - Andrew Kwok Fai LUI, 
# Robotics and Autonomous Systems Group, REF, RI
# and the Queensland University of Technology

__author__ = 'Andrew Lui'
__copyright__ = 'Copyright 2024'
__license__ = 'GPL'
__version__ = '1.0'
__email__ = 'ak.lui@qut.edu.au'
__status__ = 'Development'

import base64, io, yaml, traceback
import pandas as pd
import dash
from dash import html, dcc, callback, Input, Output, State, dash_table, ctx
import dash_bootstrap_components as dbc
# project modules
from dash.exceptions import PreventUpdate
from cgras_datatools.logging_tools import logger
from detector.model import DETECT_DAO


class TileSampleImportFileBlock():
    def __init__(self, app, prefix):
        self.app = app
        prefix = prefix + 'tsif_'
        # define widgets
        self._toast = dbc.Toast(id=prefix+'toast', is_open=False, duration=5000, icon='danger', header='Message',
                                style={'position': 'fixed', 'top': '15%', 'left': '50%', 'width': 640, 'transform': 'translate(-50%, -50%)'})
        # define tile sample spec import panel
        self.file_upload_area = dcc.Upload(id=prefix+'file_import_area', children=html.Div([
            'Drag and Drop or ', html.A('Select a tile sample yaml file')]), style={
            'width': '90%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
            'textAlign': 'center', 'margin': '10px'}, multiple=False)
        
        # define bulk folder import widgets (multi-file upload from client/outside container)
        self.folder_upload_area = dcc.Upload(id=prefix+'folder_import_area', children=html.Div([
            'Drag and Drop or ', html.A('Select multiple yaml files from a folder')]),
            style={'width': '90%', 'height': '60px', 'lineHeight': '60px',
                   'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                   'textAlign': 'center', 'margin': '10px'}, multiple=True)
        self.bulk_confirm_modal = dbc.Modal(id=prefix+'bulk_confirm_modal', children=[
                dbc.ModalHeader(dbc.ModalTitle('Bulk Import Yaml Files')),
                dbc.ModalBody(children=[
                    html.P(id=prefix+'bulk_confirm_textbox'),
                    html.Div(id=prefix+'bulk_confirm_button_panel', children=[
                        dbc.Button('Confirm', id=prefix+'bulk_confirm_button', n_clicks=0, className='me-3'),
                        dbc.Button('Cancel', id=prefix+'bulk_cancel_button', n_clicks=0, color='secondary'),
                    ], className='text-center mt-3', style={'display': 'block'}),
                ]),
            ], size='lg', is_open=False)

        # define tile sample confirm panel
        self.confirm_modal = dbc.Modal(id=prefix+'confirm_modal', children=[
                dbc.ModalHeader(dbc.ModalTitle(id=prefix+'confirm_modal_title')),
                dbc.ModalBody(children=[html.P(id=prefix+'confirm_modal_textbox', className='text-danger'),
                                        dash_table.DataTable(id=prefix+'confirm_modal_table', style_cell={'textAlign': 'left'},),
                                        html.Div(id=prefix+'confirm_modal_button_panel', children=[
                                            dbc.Button('Confirm', id=prefix+'confirm_button', n_clicks=0, className='me-3'), 
                                            dbc.Button('Cancel', id=prefix+'cancel_button', n_clicks=0, color='secondary'),], 
                                        className='text-center, mt-3', style={'display': 'block'}),
                                        ]),
                html.P('Note: The tile sample import process will run in the background and will take some time to complete.', className='mt-3 text-danger'),
            ], size='xl', is_open=False,)  
        
        self._panel = dbc.Col([
                dcc.Store(id=prefix+'imported_content'),
                dcc.Store(id=prefix+'bulk_imported_content'),
                html.H4(dbc.Badge('IMPORT TILE SAMPLE FROM SPEC YAML FILE', className='ms-1 me-2', color='white', text_color='secondary')),
                html.P('Select the yaml file that specifies the images of a tile sample.', style={'display': 'inline-block'}),
                self.file_upload_area,
                html.Hr(),
                html.P('Or select multiple yaml files at once (e.g. all files from a folder via Ctrl+A):', style={'display': 'inline-block'}),
                self.folder_upload_area,
                self._toast,
                self.confirm_modal,
                self.bulk_confirm_modal,
            ], className='mx-auto text-center')   
                 
        # --- setup callbacks
        # callback setup for the tile sample import area and confirm dialog
        self.app.callback([Output(prefix+'toast', 'is_open', allow_duplicate=True),
                           Output(prefix+'toast', 'children', allow_duplicate=True),
                           Output(prefix+'confirm_modal', 'is_open', allow_duplicate=True),],
                        [Input(prefix+'confirm_button', 'n_clicks'),
                        Input(prefix+'cancel_button', 'n_clicks'),
                        State(prefix+'imported_content', 'data')], 
            prevent_initial_call=True)(self._file_import_confirmed())

        # callback setup for the bulk multi-file import
        self.app.callback([Output(prefix+'bulk_confirm_modal', 'is_open', allow_duplicate=True),
                           Output(prefix+'bulk_confirm_textbox', 'children'),
                           Output(prefix+'bulk_confirm_button_panel', 'style'),
                           Output(prefix+'bulk_imported_content', 'data'),
                           Output(prefix+'folder_import_area', 'contents'),
                           Output(prefix+'folder_import_area', 'filename'),],
                        [Input(prefix+'folder_import_area', 'contents'),
                         State(prefix+'folder_import_area', 'filename'),],
            prevent_initial_call=True)(self._bulk_files_received())

        self.app.callback([Output(prefix+'toast', 'is_open', allow_duplicate=True),
                           Output(prefix+'toast', 'children', allow_duplicate=True),
                           Output(prefix+'bulk_confirm_modal', 'is_open', allow_duplicate=True),],
                        [Input(prefix+'bulk_confirm_button', 'n_clicks'),
                         Input(prefix+'bulk_cancel_button', 'n_clicks'),
                         State(prefix+'bulk_imported_content', 'data')],
            prevent_initial_call=True)(self._bulk_import_confirmed())
     
        self.app.callback([Output(prefix+'confirm_modal', 'is_open', allow_duplicate=True),
                           Output(prefix+'confirm_modal_title', 'children'),
                           Output(prefix+'confirm_modal_table', 'data'),
                           Output(prefix+'confirm_modal_textbox', 'children'),
                           Output(prefix+'confirm_modal_button_panel', 'style'),
                           Output(prefix+'imported_content', 'data'),
                            Output(prefix+'file_import_area', 'contents'),  # essential to clear the uploaded_content to accept another upload file
                            Output(prefix+'file_import_area', 'filename'),],   # essential to clear the uploaded_content to accept another upload file
            [   Input(prefix+'file_import_area', 'contents'),
                State(prefix+'file_import_area', 'filename'),
                State(prefix+'file_import_area', 'last_modified'),
                # State('tile_upload_worksheet_checklist', 'value')
            ], 
            prevent_initial_call=True)(self._file_import_received())
    
    def get_panel(self):
        return self._panel

    # define callback functions
    def _file_import_confirmed(self): 
        def file_import_confirmed(confirm_button, cancel_button, tile_sample_data):
            button_id = ctx.triggered_id if ctx.triggered_id is not None else 'No clicks yet'
            if button_id.endswith('confirm_button'):
              
                result = DETECT_DAO.import_tile_sample_yaml(tile_sample_data)
                if result:
                    message = 'Import tile sample successful'
                else:
                    message = 'Import tile sample failed'
                return (True, message, False) 
            elif button_id.endswith('cancel_button'):
                
                return (False, ' ', False) 
        return file_import_confirmed  
     
    # the callback for import tile sample yaml file
    def _file_import_received(self): 
        def file_import_received(contents, filename, last_modified):       
            uploaded = {'contents': contents, 'filename': filename, 'last_modified': last_modified}
            content_type, content_string = contents.split(',')
            try:
                decoded = base64.b64decode(content_string)
                tile_sample_data = yaml.load(io.BytesIO(decoded), Loader=yaml.Loader)
                
                # validate the input and obtain the model for the confirm dialog
                is_valid, model = DETECT_DAO.validate_tile_sample_import(tile_sample_data)
                if not is_valid:
                    message = 'One or more problems have been found in the tile sample spec yaml file.'
                    return (True, 'Error in the uploaded file', model.to_dict('records'), message, {'display': 'none'}, tile_sample_data, None, None,)  
                else:
                    return (True, 'Confirm to import this tile sample', model.to_dict('records'), None, {'display': 'block'}, tile_sample_data, None, None,) 
            except Exception as e:
                traceback.print_exc()
                return (True, 'Unrecognized import file format', None, None, {'display': 'none'}, None, None, None,) 
        return file_import_received

    def _bulk_files_received(self):
        def bulk_files_received(contents_list, filename_list):
            if not contents_list:
                raise PreventUpdate
            parsed = []
            fail_count = 0
            for contents, filename in zip(contents_list, filename_list):
                try:
                    _content_type, content_string = contents.split(',')
                    decoded = base64.b64decode(content_string)
                    tile_sample_data = yaml.load(io.BytesIO(decoded), Loader=yaml.Loader)
                    parsed.append(tile_sample_data)
                except Exception:
                    traceback.print_exc()
                    fail_count += 1
            if not parsed:
                return (True, 'No valid yaml files could be parsed.', {'display': 'none'}, None, None, None)
            message = (f'{len(parsed)} yaml file(s) ready to import'
                       + (f' ({fail_count} could not be parsed and will be skipped).' if fail_count else '.'))
            return (True, message, {'display': 'block'}, parsed, None, None)
        return bulk_files_received

    def _bulk_import_confirmed(self):
        def bulk_import_confirmed(confirm_button, cancel_button, bulk_data):
            button_id = ctx.triggered_id if ctx.triggered_id is not None else 'No clicks yet'
            if button_id.endswith('bulk_confirm_button'):
                if not bulk_data:
                    return (True, 'No data to import.', False)
                success_count, fail_count = 0, 0
                for tile_sample_data in bulk_data:
                    try:
                        is_valid, _ = DETECT_DAO.validate_tile_sample_import(tile_sample_data)
                        if not is_valid:
                            fail_count += 1
                            continue
                        result = DETECT_DAO.import_tile_sample_yaml(tile_sample_data)
                        if result:
                            success_count += 1
                        else:
                            fail_count += 1
                    except Exception:
                        traceback.print_exc()
                        fail_count += 1
                message = (f'Bulk import complete: {success_count} succeeded, {fail_count} failed '
                           f'out of {len(bulk_data)} file(s).')
                return (True, message, False)
            elif button_id.endswith('bulk_cancel_button'):
                return (False, ' ', False)
        return bulk_import_confirmed