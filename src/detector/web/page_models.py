# Copyright 2024 - Andrew Kwok Fai LUI, 
# Robotics and Autonomous Systems Group, REF, RI
# and the Queensland University of Technology

__author__ = 'Andrew Lui'
__copyright__ = 'Copyright 2024'
__license__ = 'GPL'
__version__ = '1.0'
__email__ = 'ak.lui@qut.edu.au'
__status__ = 'Development'

import pandas as pd
import dash
from dash import html, dcc, callback, Input, Output, State, ctx, dash_table
import dash_bootstrap_components as dbc
# project modules
from dash.exceptions import PreventUpdate
from cgras_datatools.logging_tools import logger
from detector.web.blocks import YoloModelFileImportBlock, YoloModelTable, YoloModelRangeChartBlock, HealthModelFileImportBlock, HealthModelTable, LocTileModelSelectBlock

dash.register_page(__name__)

# -- define the GUI components of this page
class ModelsPage():
    def __init__(self, app):
        self.app = app
        prefix = 'models_'
        # the yolo model blocks
        self.yolo_model_file_import_panel = YoloModelFileImportBlock(app, prefix)
        self.yolo_model_table_panel = YoloModelTable(app, prefix)
        self.yolo_model_range_chart_panel = YoloModelRangeChartBlock(app, prefix)
        # the health model blocks
        # self.health_model_file_import_panel = HealthModelFileImportBlock(app, prefix)
        # self.health_model_table_panel = HealthModelTable(app, prefix)
        # the loctile model selection block
        self.loctile_model_select_panel = LocTileModelSelectBlock(app, prefix)
        self._define_page()
    
    def layout(self, validate=False):
        return self._layout

    def _define_page(self):
        # connect the yolo import and other yolo blocks
        self.yolo_model_table_panel.register_update_table_trigger(self.yolo_model_file_import_panel.get_import_success_trigger_id())
        self.yolo_model_range_chart_panel.register_update_chart_trigger(self.yolo_model_table_panel.get_updated_success_trigger_id())
        # connect the import health model and the table
        # self.health_model_table_panel.register_update_table_trigger(self.health_model_file_import_panel.get_import_success_trigger_id())

        # putting the components together 
        rows = html.Div(id='scan-body', children =[
            dbc.Row(html.H3(children = 'Coral Object Detection Model Manager', className='mt-3 mb-3')),
            # dbc.Row(html.H4(children = 'Model', className='text-center mt-5 mb-3')),
            
            dbc.Row([dbc.Col(self.yolo_model_file_import_panel.get_panel(), className='col-3 border'), 
                     dbc.Col(self.yolo_model_table_panel.get_panel(), className='ps-3 pe-3 col-9 border'), 
                     ], className='mx-auto mt-5'),
           
            # dbc.Row(html.H5(children = 'Scopes of Yolo Models', className='text-center mt-5 mb-3')),
            dbc.Row([dbc.Col(self.yolo_model_range_chart_panel.get_panel(), className='col-12 border'), 
                     ], className='mx-auto mt-3'), 
            
            # dbc.Row(html.H4(children = 'Import Health Model', className='text-center mt-5 mb-3')),
            # dbc.Row([dbc.Col(self.health_model_file_import_panel.get_panel(), className='col-12 border'),
            #          ], className='mx-auto'),
            # dbc.Row(html.H4(children = 'Health Models', className='text-center mt-5 mb-3')),
            # dbc.Row([dbc.Col(self.health_model_table_panel.get_panel(), className='col-12 border'),
            #          ], className='mx-auto'),

            dbc.Row(html.H3(children='Tile Corner Detection Model Manager', className='mt-5 mb-3')),
            dbc.Row([dbc.Col(self.loctile_model_select_panel.get_panel(), className='col-12 border'),
                     ], className='mx-auto mt-3'),
                                 
        ], className='mx-auto col-10 mb-5')
        self._layout = dbc.Container(rows, fluid=True)


