# Copyright 2024 - Andrew Kwok Fai LUI, 
# Robotics and Autonomous Systems Group, REF, RI
# and the Queensland University of Technology

__author__ = 'Andrew Lui'
__copyright__ = 'Copyright 2024'
__license__ = 'GPL'
__version__ = '1.0'
__email__ = 'ak.lui@qut.edu.au'
__status__ = 'Development'

import os
import pandas as pd
# dash modules
import dash
from dash import html, dcc, Input, Output, State, dash_table, ctx, ALL
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from detector.model import DETECT_DAO, PERSISTENT_STORE_DAO, CONFIG, SystemConfigNames
from detector.dao_detect import SampleStatusNames
from detector.task_detection import DetectionTaskModel
from cgras_datatools.logging_tools import logger

class TileSampleTable():
    def __init__(self, app, prefix, allow_priority=True, allow_reprocess=False, allow_delete=False, allow_view=False, paginagate:int=None):
        self.app = app 
        self.prefix = prefix = prefix + 'tst_'
        self.allow_priority = allow_priority
        self.allow_reprocess = allow_reprocess
        self.update_table_store_id = prefix + 'update_table_store'
        self.user_action_store_id = prefix + 'user_action_store'
      
        # define a toast for feedback  
        self._toast = dbc.Toast(id=prefix+'toast', is_open=False, duration=5000, icon='danger', header='Message',
                                style={'position': 'fixed', 'top': '15%', 'left': '50%', 'width': 640, 'transform': 'translate(-50%, -50%)'})

        # define the modal for confirmation of user actions
        self._user_confirm_modal = dbc.Modal([
                    dbc.ModalHeader(dbc.ModalTitle(id=prefix+'confirm_modal_title')),
                    html.Div([html.P(id=prefix+'confirm_modal_message'),
                                dbc.Button('Confirm', id={'type': prefix+'action', 'index': 'confirm'},), 
                                dbc.Button('Cancel', id={'type': prefix+'action', 'index': 'cancel'}, color='secondary')
                            ]
                        , className='d-grid gap-2 col-8 mx-auto', style={'padding': '6px'})
                        ], id=prefix+'confirm_modal', is_open=False)

        self._columns = [{'name': 'Tile Sample ID', 'id': 'id', 'type': 'text', 'editable': False},
                         {'name': 'Capture Time', 'id': 'batch_time', 'type': 'datetime', 'editable': False},
                         {'name': 'Season', 'id': 'season', 'type': 'text', 'editable': False},
                         {'name': 'Species', 'id': 'species', 'type': 'text', 'editable': False},
                         {'name': 'Age', 'id': 'age', 'type': 'text', 'editable': False},
                         {'name': 'Settled On', 'id': 'settle_time', 'type': 'text', 'editable': False},
                         # {'name': 'Importer', 'id': 'importer_id', 'type': 'text', 'editable': False},
                         {'name': 'Import Time', 'id': 'create_time', 'type': 'text', 'editable': False},
                         {'name': 'Status', 'id': 'status', 'type': 'text', 'editable': False},
                         {'name': 'Remarks', 'id': 'remarks', 'type': 'text', 'editable': False},
                         ]
    
        self._style_cell_conditional = [
            {'if': {'column_id': 'id'},          'maxWidth': '160px'},
            {'if': {'column_id': 'batch_time'},  'maxWidth': '130px'},
            {'if': {'column_id': 'season'},      'maxWidth': '80px'},
            {'if': {'column_id': 'species'},     'maxWidth': '140px'},
            {'if': {'column_id': 'age'},         'maxWidth': '50px', 'textAlign': 'center'},
            {'if': {'column_id': 'settle_time'}, 'maxWidth': '95px'},
            {'if': {'column_id': 'create_time'}, 'maxWidth': '130px'},
            {'if': {'column_id': 'status'},      'maxWidth': '75px'},
            {'if': {'column_id': 'remarks'},     'maxWidth': '160px', 'fontSize': 14},
        ]
        _style_cell = {
            'overflow': 'hidden', 'textOverflow': 'ellipsis', 'whiteSpace': 'nowrap',
            'fontSize': 13, 'padding': '6px 8px', 'textAlign': 'left',
        }
        _style_table = {'overflowX': 'auto'}
        if paginagate is not None and isinstance(paginagate, int):
            self._datatable = dash_table.DataTable(id=prefix+'datatable', columns=self._columns, fill_width=True, row_selectable='multi',
                                               style_cell=_style_cell, style_cell_conditional=self._style_cell_conditional,
                                               style_table=_style_table, tooltip_delay=0, tooltip_duration=None,
                                               cell_selectable=allow_view, row_deletable=False, page_current=0, page_size=paginagate)
        else:
            self._datatable = dash_table.DataTable(id=prefix+'datatable', columns=self._columns, fill_width=True, row_selectable='multi',
                                               style_cell=_style_cell, style_cell_conditional=self._style_cell_conditional,
                                               style_table=_style_table, tooltip_delay=0, tooltip_duration=None,
                                               cell_selectable=allow_view, row_deletable=False)

        self._viewdata_modal = dbc.Modal([
                    dbc.ModalHeader(dbc.ModalTitle('View analysis results of the tile sample')),
                    html.Div([
                        html.B('', id=prefix+'view_modal_message'),
                        html.P(''),
                        dbc.Button('Reconstructed Tile', target='view_image', external_link=True, id=prefix+'view_reconstruct_tile_link', color='primary'),
                        dbc.Button('Annotated Tile', target='view_image', external_link=True, id=prefix+'view_annotated_tile_link', color='primary'),
                        # dbc.Button('Annotated Tile (Original Size)', target='view_image', external_link=True, id=prefix+'view_annotated_original_tile_link', color='primary'),
                        dbc.Button('Feature Matching Images', target='view_image', external_link=True, id=prefix+'view_feature_match_link', color='primary'),  
                        dbc.Button('Annotated Blobs', target='view_image', external_link=True, id=prefix+'view_annotated_blobs_link', color='primary'),                       
                        ]
                        , className='d-grid gap-2 col-8 mx-auto p-2 pb-3')
                    ], id=prefix+'view_modal', is_open=False)
        
        # define confirm panel in a modal
        self._reprocess_mode_radio = dcc.RadioItems(id=prefix+'reprocess_mode', options={
                                                '_whole': '  Redo the whole analysis (reconstruction, tile location, object detection and analysis)',
                                                '_redo_detect': '  Redo from detection (object detection and analysis)',
                                                '_redo_analysis': '  Redo analysis (only analysis)',
                                                '_redo_finalize': '  Redo finalize (include annotate image)'}, value='_redo_finalize'   )  # style={'display': 'flex'}
        
        self._confirm_reprocess_modal = dbc.Modal(id=prefix+'confirm_reprocess_modal', children=[
                dbc.ModalHeader(dbc.ModalTitle('Re-Process Tile Samples')),
                dbc.ModalBody(children=[html.P('Select the starting point of the re-process (i.e., involves removing different cache files)', className='text-secondary'),
                                        self._reprocess_mode_radio ,
                                            html.Div(id=prefix+'confirm_modal_button_panel', children=[
                                            dbc.Button('Confirm Re-Process', id=prefix+'confirm_redo_button', n_clicks=0, className='me-3'), 
                                            dbc.Button('Cancel', id=prefix+'cancel_redo_button', n_clicks=0, color='secondary'),], 
                                        className='text-center, mt-3', style={'display': 'block'}, ),
                                        ]),
            ], size='xl', is_open=False,)  
        
        self._deletedata_modal = dbc.Modal([
                    dbc.ModalHeader(dbc.ModalTitle('Deleted a Scan Session')),
                    html.Div( [
                        html.P(id=prefix+'deletedata_modal_message'),
                            ]
                        , className='d-grid gap-2 col-8 mx-auto', style={'padding': '6px'})
                        ], id=prefix+'deletedata_modal', is_open=False)

        self._div_panel_children = [
            dbc.Button('Select All', id=prefix+'table_selectall_button', color='light', className='mb-1 me-5', size='sm', style={'width': '100px'}),
        ]
        if allow_reprocess:
            self._div_panel_children.append(
                dbc.Button('Redo',  id={'type': prefix+'table', 'index': 'redo'}, color='primary',className='ms-2', size='sm', style={'width': '80px'}))
        if allow_priority:
            self._div_panel_children.append(
                dbc.Button('Prioritize',  id={'type': prefix+'table', 'index': 'priority'}, color='primary', className='ms-2', size='sm', style={'width': '80px'}))

        self._div_panel_children.extend([         
            dbc.Button('Reject',  id={'type': prefix+'table', 'index': 'reject'}, color='danger', className='ms-2', size='sm', style={'width': '80px'}),
        ])
        
        if allow_delete:
            self._div_panel_children.append(
                dbc.Button('Delete',  id={'type': prefix+'table', 'index': 'delete'}, color='danger', className='ms-2', size='sm', style={'width': '80px'}))
            
        self._datatable_panel_children = [html.Div(self._div_panel_children, style={'display':'flex'}), 
                                          self._datatable]

        self.the_panel = html.Div([
                dbc.Row(html.Div(self._datatable_panel_children, className='p-2', style={'background-color': 'rgb(225, 225, 225)'})),
                dcc.Store(id=self.update_table_store_id),
                dcc.Store(id=self.user_action_store_id),
                dcc.Store(id=prefix+'row_priority_store'),
                dcc.Store(id=prefix+'row_redo_store'),
                dcc.Store(id=prefix+'row_view_store'),
                self._confirm_reprocess_modal,
                self._deletedata_modal,    
                self._viewdata_modal,
                self._toast,
                self._user_confirm_modal,     
                ], id=prefix+'main_panel', style={'margin-top':'24px'})
        
        # define callback for selecting a scan and open the modal window
        self.app.callback([ Output(prefix+'view_modal', 'is_open'),
                            Output(prefix+'view_modal_message', 'children'), 
                            Output(prefix+'view_reconstruct_tile_link', 'href'),
                            Output(prefix+'view_reconstruct_tile_link', 'disabled'),
                            Output(prefix+'view_annotated_tile_link', 'href'),
                            Output(prefix+'view_annotated_tile_link', 'disabled'),   
                            # Output(prefix+'view_annotated_original_tile_link', 'href'),
                            # Output(prefix+'view_annotated_original_tile_link', 'disabled'),                                                        
                            Output(prefix+'view_feature_match_link', 'href'),
                            Output(prefix+'view_feature_match_link', 'disabled'),
                            Output(prefix+'view_annotated_blobs_link', 'href'),
                            Output(prefix+'view_annotated_blobs_link', 'disabled'),],
                        [Input(prefix+'row_view_store', 'data')], prevent_initial_call=True)(self._view_row_confirmed())          

        self.app.callback([Output(prefix+'confirm_reprocess_modal', 'is_open', allow_duplicate=True)],
            [Input(prefix+'row_redo_store', 'data')], prevent_initial_call=True)(self._redo_row_requested())     

        self.app.callback([Output(prefix+'toast', 'is_open', allow_duplicate=True),
                           Output(prefix+'toast', 'children', allow_duplicate=True),
                           Output(prefix+'confirm_reprocess_modal', 'is_open'),
                           Output(self.update_table_store_id, 'data', allow_duplicate=True)],
                        [Input(prefix+'confirm_redo_button', 'n_clicks'),
                        Input(prefix+'cancel_redo_button', 'n_clicks'),
                        State(prefix+'reprocess_mode', 'value'),
                        State(prefix+'row_redo_store', 'data'),
                        State(self.update_table_store_id, 'data'),], prevent_initial_call=True)(self._redo_row_confirmed())      
             
        self.app.callback([Output(prefix+'toast', 'is_open', allow_duplicate=True),
                           Output(prefix+'toast', 'children', allow_duplicate=True),
                           Output(prefix+'datatable', 'page_current')],
                            [Input(prefix+'row_priority_store', 'data'),], prevent_initial_call=True)(self._priority_row_confirmed())  
        
        self.app.callback([Output(prefix+'confirm_modal', 'is_open', allow_duplicate=True),
                           Output(prefix+'toast', 'is_open', allow_duplicate=True),
                           Output(prefix+'toast', 'children', allow_duplicate=True),
                           Output(self.update_table_store_id, 'data', allow_duplicate=True)],
                            [State(self.user_action_store_id, 'data'),
                            State(self.update_table_store_id, 'data'),
                            Input({'type': prefix+'action', 'index': ALL}, 'n_clicks'),], prevent_initial_call=True)(self._cb_confirm_modal_pressed())            

        input_list = [State(prefix+'datatable', 'selected_rows'), 
                      Input({'type': prefix+'table', 'index': ALL}, 'n_clicks'),]
        self.app.callback([Output(prefix+'confirm_modal', 'is_open', allow_duplicate=True),
                            Output(prefix+'confirm_modal_title', 'children', allow_duplicate=True),
                            Output(prefix+'confirm_modal_message', 'children', allow_duplicate=True),
                            Output(prefix+'row_redo_store', 'data'),
                            Output(prefix+'row_priority_store', 'data'),
                            Output(self.user_action_store_id, 'data'),
                            Output(prefix+'datatable', 'selected_rows', allow_duplicate=True),
                           ], input_list, prevent_initial_call=True)(self._table_button_pressed())     

        self.app.callback(Output(prefix+'datatable', 'style_data_conditional'),
                            [Input(prefix+'datatable', 'derived_viewport_selected_rows'),
                             State(prefix+'datatable', 'data')])(self._style_selected_rows())
        
        self.app.callback([Output(self.prefix+'datatable', 'data'),
                           Output(self.prefix+'datatable', 'tooltip_data')],
            [Input(self.update_table_store_id, 'data')], prevent_initial_call=True, allow_duplicate=True)(self._update_datatable())       
        
        self.app.callback([Output(self.prefix+'datatable', 'selected_rows')],
            [Input(self.prefix+'table_selectall_button', 'n_clicks'),
             State(self.prefix+'datatable', 'data'),
             State(self.prefix+'datatable', 'selected_rows')], prevent_initial_call=True, allow_duplicate=True)(self._selectall_button_pressed())  
                             
        self.app.callback([Output(prefix+'row_view_store', 'data'),                 
                            Output(prefix+'datatable', 'active_cell', allow_duplicate=True),
                            Output(prefix+'datatable', 'selected_cells', allow_duplicate=True),],
                            [Input(prefix+'datatable', 'active_cell'),
                             State(prefix+'datatable', 'data')], prevent_initial_call=True, allow_duplicate=True)(self._cb_cell_selected())     
    
    def register_trigger(self, trigger_id:str):
        # define callbacks for the datatable data
        self.app.callback([Output(self.update_table_store_id, 'data', allow_duplicate=True)],
            [Input(trigger_id, 'data')], prevent_initial_call=True, allow_duplicate=True)(self._update_panel())

    def get_panel(self):
        return self.the_panel
    
    def get_default_datatable_model(self):
        model = DETECT_DAO.list_tile_samples(season_title=None, status=SampleStatusNames.QUEUED.value)
        return model
    
    def refine_datatable_model(self, model, show_column_top=True, show_column_refresh=False):
        model['status'] = model['status'].apply(lambda x: SampleStatusNames(x).name)
        model = model[['id', 'batch_time', 'season', 'species', 'age', 'settle_time', 'create_time', 'status', 'remarks']]
        return model
    
    # the callback for updating the datatable
    def _update_datatable(self):
        def update_datatable(store):
            # if the store contains a list, a search query triggers table refresh
            if isinstance(store, list):
                species_filter = store[6] if len(store) > 6 else None
                self._model = DETECT_DAO.query_processed_tile_samples(*store[:6])
                if species_filter:
                    self._model = self._model[self._model['species'].isin(species_filter)]
            else:
                self._model = self.get_default_datatable_model()
            self._model = self.refine_datatable_model(self._model, self.allow_priority, self.allow_reprocess)
            records = self._model.to_dict('records')
            tooltip_data = [
                {col: {'value': str(val) if val is not None else '', 'type': 'text'}
                 for col, val in row.items()}
                for row in records
            ]
            return (records, tooltip_data)
        return update_datatable
    
    def _table_button_pressed(self): 
        def table_button_pressed(selected_rows:list, *args):
            if selected_rows is None or len(selected_rows) == 0:
                raise PreventUpdate
            row_index_list = list(selected_rows)
            button_id = ctx.triggered_id if ctx.triggered_id is not None else {}
            button_index = button_id.get('index', None)
            # handle each button pressed
            if button_index.endswith('redo'):
                return (False, None, None, row_index_list, None, None, [])
            elif button_index.endswith('priority'):
                return (False, None, None, None, row_index_list, None, [])
            elif button_index.endswith('reject'):
                title = 'Reject the Selected Samples'
                message = 'The selected tile sample(s) will be rejected and their findings cleared! Are you sure?'
                action_data = {'selected_list': row_index_list, 'action': 'reject'}
                return (True, title, message, None, None, action_data, []) 
            elif button_index.endswith('delete'):
                title = 'Delete the Selected Samples'
                message = 'The selected tile sample(s) and their findings will be permanantly deleted! The tile samples may be imported again. Are you sure?'
                action_data = {'selected_list': row_index_list, 'action': 'delete'}
                return (True, title, message, None, None, action_data, [])                      
            return (False, None, None, None, None, None, [])
        return table_button_pressed 

    # callback when a cell is clicked, which should trigger the 
    def _cb_cell_selected(self):
        def cb_cell_selected(active_cell, model):
            if not active_cell:
                raise PreventUpdate
            row, column = active_cell['row'], active_cell['column']
            return ([row], None, [],)
        return cb_cell_selected

    def _cb_confirm_modal_pressed(self): 
        def cb_confirm_modal_pressed(action_data, store, *args):
            button_id = ctx.triggered_id if ctx.triggered_id is not None else {}
            button_index = button_id.get('index', None)
            if action_data is None or 'action' not in action_data:
                raise PreventUpdate
            # if the confirm buttons is pressed
            if button_index == 'confirm':
                row_index_list = action_data['selected_list']
                message_info = []
                for row_index in row_index_list:
                    if row_index >= len(self._model):
                        continue
                    tile_sample_id = self._model.iloc[row_index]['id']
                    if action_data['action'] == 'reject':
                        DETECT_DAO.update_tile_sample_status(tile_sample_id, SampleStatusNames.REJECTED.value)
                        DETECT_DAO.clear_tile_sample_data(tile_sample_id)
                        DetectionTaskModel.delete_cache_files(tile_sample_id, delete_reco=True, delete_object_list=True, delete_object_detection_model=True)
                    elif action_data['action'] == 'delete':
                        DETECT_DAO.clear_tile_sample_data(tile_sample_id)
                        DetectionTaskModel.delete_cache_folder(tile_sample_id)
                        DETECT_DAO.delete_tile_sample(tile_sample_id)
                    # DETECT_DAO.delete_tile_sample(id)
                    message_info.append(tile_sample_id)
                message = f'{action_data["action"].upper()} of the tile sample(s) {message_info} is successful'
                return (False, True, message, store)
            return (False, False, '', store)
        return cb_confirm_modal_pressed 

    def _redo_row_requested(self): 
        def redo_row_requested(row_index_list):
            if row_index_list is None:
                raise PreventUpdate        
            return (True,)  
        return redo_row_requested 

    def _redo_row_confirmed(self): 
        def redo_row_confirmed(confirm_button, cancel_button, mode, row_index_list, store):
            button_id = ctx.triggered_id if ctx.triggered_id is not None else 'No clicks yet'
            if button_id.endswith('confirm_redo_button'):
                message_info = []
                tile_sample_id_list = []
                for row_index in row_index_list:
                    tile_sample_id = self._model.iloc[row_index]['id']
                    tile_sample_id_list.append(tile_sample_id)
                
                for tile_sample_id in tile_sample_id_list:
                    DETECT_DAO.update_tile_sample_status(tile_sample_id, SampleStatusNames.QUEUED.value, '')
                    DETECT_DAO.clear_tile_sample_data(tile_sample_id)
                    message_info.append(tile_sample_id)
                    # remove the cache files
                    if mode == '_whole':
                        DetectionTaskModel.delete_cache_files(tile_sample_id, delete_reco=True, delete_object_list=True, delete_object_detection_model=True)
                    elif mode == '_redo_detect':
                        DetectionTaskModel.delete_cache_files(tile_sample_id, delete_reco=False, delete_object_list=True, delete_object_detection_model=True)
                    elif mode == '_redo_analysis':
                        DetectionTaskModel.delete_cache_files(tile_sample_id, delete_reco=False, delete_object_list=False, delete_object_detection_model=True)
                    elif mode == '_redo_finalize':
                        ...                      
                        
                message = f'The tile sample(s) {message_info} have been moved to the queue pending analysis'
                return (True, message, False, store)
            return (False, ' ', False, store) 
        return redo_row_confirmed 

    def _priority_row_confirmed(self): 
        def priority_row_confirmed(row_index_list):
            if row_index_list is None:
                raise PreventUpdate
            message_info = []
            for row_index in row_index_list:
                id = self._model.iloc[row_index]['id']
                DETECT_DAO.set_top_priority(id)
                message_info.append(id)
            message = f'The tile sample(s) {message_info} have been moved to the top priority'
            return (True, message, 0) 
        return priority_row_confirmed 
    
    def _view_row_confirmed(self):
        def view_row_confirmed(row_index_list):
            if not row_index_list:
                raise PreventUpdate
            # retrieve the tile_sample_id of the selected row
            tile_sample_id = self._model.iloc[row_index_list[0]]['id']
            logdata_folder = DetectionTaskModel.get_cache_folder(tile_sample_id)
            if logdata_folder is None:
                raise PreventUpdate
            # build the href based on the aux server configuration
            href = f'http://{CONFIG.get(SystemConfigNames.AUX_WEB_HOST, "localhost")}:{CONFIG.get(SystemConfigNames.AUX_WEB_PORT, "8024")}'
            # generate the title
            modal_title = f'Tile Sample ID: {tile_sample_id}'
            # obtain the partial path to the cache folder
            partial_cache_folder = DetectionTaskModel.get_partial_cache_folder(tile_sample_id)
            view_reconstruct_href = view_annotated_href = view_feature_match_href = view_annotated_blobs_href = None
            if partial_cache_folder is not None:
                # evalate if the file exists
                view_reconstruct_path = os.path.join(logdata_folder, DetectionTaskModel.WHOLE_RECO_HTML_FILENAME)
                if os.path.isfile(view_reconstruct_path):
                    view_reconstruct_href = f'{href}/{partial_cache_folder}/{DetectionTaskModel.WHOLE_RECO_HTML_FILENAME}' 
                    
                # evalate if the file exists
                view_annotated_path = os.path.join(logdata_folder, DetectionTaskModel.ANNOTATED_WHOLE_RECO_HTML_FILENAME)
                if os.path.isfile(view_annotated_path):
                    view_annotated_href = f'{href}/{partial_cache_folder}/{DetectionTaskModel.ANNOTATED_WHOLE_RECO_HTML_FILENAME}'                     

                # # evalate if the file exists
                # view_annotated_original_path = os.path.join(logdata_folder, DetectionTaskModel.ROTATED_ANNOTATED_ORIGINAL_SCALE_HTML_FILENAME)
                # if os.path.isfile(view_annotated_original_path):
                #     view_annotated_original_href = f'{href}/{partial_cache_folder}/{DetectionTaskModel.ROTATED_ANNOTATED_ORIGINAL_SCALE_HTML_FILENAME}'   

                # evaluate if the file exists
                view_feature_match_path = os.path.join(logdata_folder, DetectionTaskModel.FEATURE_MATCH_HTML_FILENAME)
                if os.path.isfile(view_feature_match_path):
                    view_feature_match_href = f'{href}/{partial_cache_folder}/{DetectionTaskModel.FEATURE_MATCH_HTML_FILENAME}' 
                    
                # evaluate if the file exists
                view_annotated_blobs_path = os.path.join(logdata_folder, DetectionTaskModel.ANNOTATED_BLOBS_INDEX_HTML_FILENAME)
                if os.path.isfile(view_annotated_blobs_path):
                    view_annotated_blobs_href = f'{href}/{partial_cache_folder}/{DetectionTaskModel.ANNOTATED_BLOBS_INDEX_HTML_FILENAME}' 
                                
            return (True, modal_title, view_reconstruct_href, view_reconstruct_href==None, view_annotated_href, view_annotated_href==None, 
                    # view_annotated_original_href, view_annotated_original_href==None,
                    view_feature_match_href, view_feature_match_href==None, 
                    view_annotated_blobs_href, view_annotated_blobs_href==None)
        return view_row_confirmed
    
    def _style_selected_rows(self):
        def style_selected_rows(row_index_list, model):
            if row_index_list is None:
                return dash.no_update
            style_data_conditional = [
                {"if": {"filter_query": "{{id}} ={}".format(model[i]['id'])}, "backgroundColor": "yellow",}
                for i in row_index_list
            ]
            style_data_conditional.append({'if': {
                    'filter_query': '{status} contains "REJECTED"',
                    'column_id': 'status'
                }, 'backgroundColor': '#ffffff', 'color': 'rgb(255, 0, 0)'} )
            style_data_conditional.append({'if': {
                    'filter_query': '{status} contains "FLAGGED"',
                    'column_id': 'status'
                }, 'backgroundColor': '#ffffff', 'color': 'rgb(255, 0, 0)'} )            
            return style_data_conditional
        return style_selected_rows
    
    def _selectall_button_pressed(self): 
        def selectall_button_pressed(selectall_button, model, selected_rows):
            if selectall_button is None:
                raise PreventUpdate
            if selected_rows is not None and len(selected_rows) == len(model):
                selected_rows = []
            else:
                selected_rows = [index for index in range(len(model))]
            return (selected_rows,)  
        return selectall_button_pressed     
    
    def _update_panel(self): 
        def update_panel(store):
            if store is None:
                raise PreventUpdate        
            return (store,)  
        return update_panel 