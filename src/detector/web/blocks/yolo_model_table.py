# Copyright 2024 - Andrew Kwok Fai LUI, 
# Robotics and Autonomous Systems Group, REF, RI
# and the Queensland University of Technology

__author__ = 'Andrew Lui'
__copyright__ = 'Copyright 2024'
__license__ = 'GPL'
__version__ = '1.0'
__email__ = 'ak.lui@qut.edu.au'
__status__ = 'Development'

import shutil
import pandas as pd
# dash modules
import dash
from dash import html, dcc, Input, Output, State, dash_table, ctx, ALL
import dash_bootstrap_components as dbc
from dash.dash_table.Format import Format, Padding
from dash.exceptions import PreventUpdate
from detector.model import DETECT_DAO, CONFIG, SystemConfigNames, ClassHierarchyCoral
from cgras_datatools.logging_tools import logger

class YoloModelTable():
    def __init__(self, app, prefix):
        self.app = app 
        self.prefix = prefix = prefix + 'ymet_'
        self.default_max_end_day = CONFIG.get(SystemConfigNames.MAX_CORAL_AGE)
        self.updated_success_trigger_id = prefix + 'updated_datatable_trigger'
        # model variable
        # define the modal for confirmation of user actions
        self._user_confirm_modal = dbc.Modal([
                    dbc.ModalHeader(dbc.ModalTitle(id=prefix+'confirm_modal_title')),
                    html.Div([html.P(id=prefix+'confirm_modal_message'),
                                dbc.Button('Confirm', id={'type': prefix+'action', 'index': 'confirm'},), 
                                dbc.Button('Cancel', id={'type': prefix+'action', 'index': 'cancel'}, color='secondary')
                            ]
                        , className='d-grid gap-2 col-8 mx-auto', style={'padding': '6px'})
                        ], id=prefix+'confirm_modal', is_open=False)
        
        # define a toast for feedback  
        self._toast = dbc.Toast(id=prefix+'toast', is_open=False, duration=5000, icon='danger', header='Message',
                                style={'position': 'fixed', 'top': '15%', 'left': '50%', 'width': 640, 'transform': 'translate(-50%, -50%)'})
        
        self._model, self._model_column = self.get_default_datatable_model()
        self._datatable = dash_table.DataTable(data=self._model, columns=self._model_column,
                                               id=prefix+'datatable', style_header={}, fill_width=True,
                                               cell_selectable=False, row_selectable='multi')
        
        model_name_label = dbc.Row([dbc.Label('Model Name:', width=2), dbc.Label(id=prefix+'name_label', width=10), ])
        model_file_label = dbc.Row([dbc.Label('Model File:', width=2), dbc.Label(id=prefix+'file_label', width=10), ])
        classes_map_label =  dbc.Row([dbc.Label('Class Map:', width=2),         
                                      dbc.ListGroup([dbc.ListGroupItem(id=prefix+'class_map_1', action=False, style={'border': 'none'}),
                                                       dbc.ListGroupItem(id=prefix+'class_map_2', action=False, style={'border': 'none'}),
                                                       dbc.ListGroupItem(id=prefix+'class_map_3', action=False, style={'border': 'none'}),
                                                       dbc.ListGroupItem(id=prefix+'class_map_4', action=False, style={'border': 'none'}),  
            ]
        , className='col-10'),])  

        yolo_predict_params_label = dbc.Row([dbc.Label('YOLO Predict Params:', width=2), dbc.Label(id=prefix+'predict_params_label', width=10), ])
        
        keep_object_filter_label = dbc.Row([dbc.Label('Keep Object Filters:', width=2), dbc.Label(id=prefix+'keep_object_filter_label', width=10), ])

        # model_coral_class_label = dbc.Row([dbc.Label('Coral Classes:', width=2), dbc.Label(id=prefix+'coral_classes_label', width=10), ])
        # model_dead_coral_class_label = dbc.Row([dbc.Label('Dead Coral Classes:', width=2), dbc.Label(id=prefix+'dead_coral_classes_label', width=10), ])
        
        species_input = dbc.Row([
            dbc.Label('Species', html_for=prefix+'species_input', width=2),
            dbc.Col(dbc.Input(type='text', id=prefix+'species_input', placeholder='Enter coral species name'), width=10),
        ], className='mb-3',)
        
        range_input = dbc.Row([
            dbc.Label('Applicable Period', html_for=prefix+'range_input', width=2),
            dbc.Col(dcc.RangeSlider(0, self.default_max_end_day, value=[0, self.default_max_end_day], id=prefix+'range_input', marks={
                        0: {'label': 'Start', 'style': {'color': '#77b0b1'}},
                        21: {'label': '21'},
                        42: {'label': '42'},
                        63: {'label': '63'},
                        self.default_max_end_day: {'label': 'End', 'style': {'color': '#f50'}}
                    }, tooltip={"placement": "bottom", "always_visible": True}))
        ])
        
        is_active_input = dbc.Row([
            dbc.Label('Active', html_for=prefix+'is_active_input', width=2),
            dbc.Col(dbc.Switch(id=prefix+'is_active_input', value=True, label='Model is used for coral object detection'), width=10),
        ], className='mb-3')

        define_yolo_model_form = dbc.Form([model_name_label, model_file_label, classes_map_label, yolo_predict_params_label, keep_object_filter_label, species_input, range_input, is_active_input])
                
        self._editdata_modal = dbc.Modal(id=prefix+'edit_modal', children=[
                dbc.ModalHeader(dbc.ModalTitle(children='Edit Model Attributes',)),
                dbc.ModalBody(children=[html.P(id=prefix+'edit_modal_textbox', className='text-danger'),
                                        html.P('Make the changes and Press Confirm', className='mb-3', style={}),
                                        define_yolo_model_form,
                                        html.Div(children=[
                                            dbc.Button('Confirm', id={'type': prefix+'edit_action', 'index': 'confirm'}, n_clicks=0, className='me-3'), 
                                            dbc.Button('Cancel', id={'type': prefix+'edit_action', 'index': 'cancel'}, n_clicks=0, color='secondary'),], 
                                        className='text-center, mt-3', style={'display': 'block'}),
                                        ]),
            ], size='xl', is_open=False,)  

        self._datatable_title = dbc.Row(html.Div([
                        html.Span('⬇️', className='col-6 text-center'),
                        html.H6([dbc.Badge('YOLO MODELS', color='white', text_color='primary'), 
                        html.Span(id=prefix+'datatable_title')]),
                    ], className='mx-auto text-center')),

        self.the_panel = html.Div([
                html.H4(dbc.Badge('CURRENT COD MODELS', className='ms-1 me-2', color='white', text_color='secondary')),
                dbc.Row(html.Div([
                    dbc.Button('Edit', id={'type': prefix+'table', 'index': 'update'}, n_clicks=0, color='secondary', className='mb-1 me-1', size='sm'), 
                    dbc.Button('Delete', id={'type': prefix+'table', 'index': 'delete'}, n_clicks=0, color='danger', className='mb-1', size='sm'),
                    self._datatable], className='p-3 text-start', style={'background-color': 'rgb(225, 225, 225)'})
                ),
                dcc.Store(id=prefix+'row_edit_store'),
                dcc.Store(id=prefix+'row_delete_store'),
                dcc.Store(id=prefix+'update_datatable_trigger'),
                dcc.Store(id=self.updated_success_trigger_id),
                self._toast,
                self._editdata_modal,    
                self._user_confirm_modal,            
                ], className='text-center')
                  
    
        self.app.callback([Output(prefix+'edit_modal', 'is_open', allow_duplicate=True),
                           Output(prefix+'name_label', 'children'),
                           Output(prefix+'file_label', 'children'),
                           Output(prefix+'predict_params_label', 'children'),
                           Output(prefix+'keep_object_filter_label', 'children'),
                           Output(prefix+'class_map_1', 'children'),
                           Output(prefix+'class_map_2', 'children'),
                           Output(prefix+'class_map_3', 'children'),
                           Output(prefix+'class_map_4', 'children'),
                           Output(prefix+'species_input', 'value'),
                           Output(prefix+'range_input', 'value'),
                           Output(prefix+'is_active_input', 'value'),],
            [Input(prefix+'row_edit_store', 'data')], prevent_initial_call=True)(self._edit_row_received())    

        self.app.callback([Output(prefix+'toast', 'is_open', allow_duplicate=True),
                           Output(prefix+'toast', 'children', allow_duplicate=True),
                           Output(prefix+'edit_modal', 'is_open', allow_duplicate=True),
                           Output(self.prefix+'update_datatable_trigger', 'data', allow_duplicate=True),
                           Output(self.updated_success_trigger_id, 'data', allow_duplicate=True),],
                        [State(prefix+'species_input', 'value'),
                        State(prefix+'range_input', 'value'),
                        State(prefix+'is_active_input', 'value'),
                        State(prefix+'row_edit_store', 'data'),
                        Input({'type': prefix+'edit_action', 'index': ALL}, 'n_clicks')], prevent_initial_call=True)(self._edit_row_confirmed()) 
        
        self.app.callback([Output(prefix+'confirm_modal', 'is_open', allow_duplicate=True),
                           Output(prefix+'toast', 'is_open', allow_duplicate=True),
                           Output(prefix+'toast', 'children', allow_duplicate=True),
                            Output(prefix+'datatable', 'data', allow_duplicate=True),
                            Output(self.updated_success_trigger_id, 'data', allow_duplicate=True),
                           ],
                            [State(prefix+'datatable', 'data_previous'),
                            State(prefix+'datatable', 'data'),
                            State(prefix+'row_delete_store', 'data'),
                            Input({'type': prefix+'action', 'index': ALL}, 'n_clicks')], prevent_initial_call=True)(self._delete_row_confirmed())   
        
        self.app.callback([Output(self.prefix+'datatable', 'data'),
                           Output(self.prefix+'datatable', 'columns'),
                           Output(self.updated_success_trigger_id, 'data', allow_duplicate=True),],
            [Input(self.prefix+'update_datatable_trigger', 'data')], prevent_initial_call=True, allow_duplicate=True)(self._update_datatable())
          
        
        self.app.callback([Output(prefix+'confirm_modal', 'is_open', allow_duplicate=True),
                            Output(prefix+'confirm_modal_title', 'children'),
                            Output(prefix+'confirm_modal_message', 'children'),
                           Output(prefix+'row_edit_store', 'data'),
                           Output(prefix+'row_delete_store', 'data'),
                            Output(prefix+'datatable', 'selected_rows', allow_duplicate=True),
                           ],
                        [State(prefix+'datatable', 'data'),
                        State(prefix+'datatable', 'selected_rows'),
                        Input({'type': prefix+'table', 'index': ALL}, 'n_clicks'),], prevent_initial_call=True)(self._table_button_pressed())   

        self.app.callback([Output(prefix+'datatable', 'style_data_conditional'),
                          Output(prefix+'datatable', 'selected_rows', allow_duplicate=True)],
                            [Input(prefix+'datatable', 'selected_rows'),
                             State(prefix+'datatable', 'data')], prevent_initial_call=True)(self._style_selected_rows())


    def register_update_table_trigger(self, trigger_id:str):
        self.app.callback([Output(self.prefix+'update_datatable_trigger', 'data'),],
            [Input(trigger_id, 'data')], prevent_initial_call=False, allow_duplicate=True)(self._trigger_update_datatable())
        
    def get_updated_success_trigger_id(self):
        return self.updated_success_trigger_id
        
    def get_panel(self):
        return self.the_panel
    
    def get_default_datatable_model(self):
        model = DETECT_DAO.list_yolo_model()
        model['Period'] = model.apply(lambda row: DETECT_DAO.get_period_str(row['start_day'], row['end_day']), axis=1, result_type='reduce')
        model['Input Image Size'] = model.apply(lambda row: f'{row["input_image_width"]} x {row["input_image_height"]}', axis=1, result_type='reduce')
        model['Active'] = model['is_active'].apply(lambda v: 'Yes' if v else 'No')
        model = model[['id', 'name', 'species', 'Period', 'Input Image Size', 'Active']]
        model.columns = ['ID', 'Model Name', 'Species', 'Period', 'Input Image Size', 'Active']
        columns = [{'name': col, 'id': col, 'type': 'text', 'editable': False} for col in model.columns]
        return model.to_dict('records'), columns
        
    # the callback for updating the datatable
    def _update_datatable(self):
        def update_datatable(store):
            self._model, self._model_column = self.get_default_datatable_model()
            return (self._model, self._model_column, True)
        return update_datatable
    
    def _trigger_update_datatable(self):
        def trigger_update_datatable(store):
            return (store,)
        return trigger_update_datatable
            
    def _table_button_pressed(self): 
        def table_button_pressed(model, selected_rows:list, *args):
            if selected_rows is None or len(selected_rows) == 0:
                raise PreventUpdate
            row_index = selected_rows[0]
            name = model[row_index]['Model Name']
            data = DETECT_DAO.get_yolo_model(name)
            button_id = ctx.triggered_id if ctx.triggered_id is not None else {}
            button_index = button_id.get('index', None)
            if button_index.endswith('update'):
                return (False, None, None, data, None, [])  
            elif button_index.endswith('delete'):   
                title = 'Delete the YOLO model'
                message = 'The selected YOLO model will be deleted. Are you sure?'
                return (True, title, message, None, row_index, [])        
            return (False, None, None, None, None, [])
        return table_button_pressed 
    
    def _edit_row_received(self):
        def edit_row_received(row):
            if row is None:
                raise PreventUpdate
            start_day, end_day = row['start_day'], row['end_day']
            if end_day is None or end_day < 0:
                end_day = self.default_max_end_day
            is_active = bool(row.get('is_active', 1))
            try:
                class_map_str_1 = f'POLYP_SINGLE: {row["classes_map"][ClassHierarchyCoral.POLYP_SINGLE.value]}'
                class_map_str_2 = f'POLYP_MULTI: {row["classes_map"][ClassHierarchyCoral.POLYP_MULTI.value]}'
                class_map_str_3 = f'POLYP_KEYPART: {row["classes_map"][ClassHierarchyCoral.POLYP_KEYPART.value]}'
                class_map_str_4 = f'DEAD_CORAL: {row["classes_map"][ClassHierarchyCoral.DEAD_CORAL.value]}'
                predict_params = row.get('predict_params', '')
                keep_object_filter = row.get('keep_object_filter', '')
                return (True, row['name'], row['model_file_path'], str(predict_params), str(keep_object_filter), class_map_str_1, class_map_str_2, class_map_str_3, class_map_str_4, row['species'], (start_day, end_day,), is_active,)
            except:
                return (True, row['name'], row['model_file_path'], None, None, None, None, None, None, row['species'], (start_day, end_day,), is_active,)
        return edit_row_received     

    def _edit_row_confirmed(self):
        def edit_row_confirmed(species, range, is_active, row, *args):
            button_id = ctx.triggered_id if ctx.triggered_id is not None else {}
            button_index = button_id.get('index', None)
            if button_index.endswith('confirm'):
                start_day, end_day = range
                end_day = -1 if end_day >= self.default_max_end_day else end_day
                result = DETECT_DAO.update_yolo_model(row['name'], species, start_day, end_day)
                DETECT_DAO.set_yolo_model_active(row['name'], 1 if is_active else 0)
                if result:
                    return (True, 'Update yolo model successful', False, True, True)
                else:
                    return (True, 'Update yolo model failed', False, True, False)
            else:
                return (False, ' ', False, True, False)
        return edit_row_confirmed  

    def _delete_row_confirmed(self): 
        def delete_row_confirmed(rows_previous, model_dict, row_index, *args):
            button_id = ctx.triggered_id if ctx.triggered_id is not None else {}
            button_index = button_id.get('index', None)
            if button_index == 'confirm':
                DETECT_DAO.delete_yolo_model(model_dict[row_index]['Model Name'])
                del model_dict[row_index]
                return (False, True, 'The YOLO model has been deleted', model_dict, True)
            else:
                return (False, False, '', model_dict, False)
        return delete_row_confirmed 

    def _style_selected_rows(self):
        def style_selected_rows(selected_rows, model):
            if selected_rows is None:
                return dash.no_update
            if len(selected_rows) >= 2:
                selected_rows.pop(0)   # assume that the new row is added to the end of the selected_row list

            style_data_conditional = [
                {"if": {"filter_query": "{{ID}} = '{}'".format(model[i]['ID'])}, "backgroundColor": "yellow",}
                for i in selected_rows
            ]
            return (style_data_conditional, selected_rows,)
        return style_selected_rows

