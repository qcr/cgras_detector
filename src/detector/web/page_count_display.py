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
from detector.web.blocks import CountTileSelectTable, CoralCountTileInfoBlock, CountTileTrendBlock, CountHeatmapBlock, CountResultDownloadBlock, CountScatterMapBlock, CountImagePreviewBlock

dash.register_page(__name__)

# -- define the GUI components of this page
class CountViewerPage():
    def __init__(self, app):
        self.app = app
        prefix = self.prefix = 'count_display'
        self.update_trigger_id = self.prefix+'update_trigger'
        self.tile_browse_table = CountTileSelectTable(app, prefix)    # the table on the left menu
        self.tile_detect_info = CoralCountTileInfoBlock(app, prefix)
        self.coral_count_trend = CountTileTrendBlock(app, prefix)
        self.count_heatmap = CountHeatmapBlock(app, prefix)
        self.count_scatter_plot = CountScatterMapBlock(app, prefix)
        self.count_image_preview = CountImagePreviewBlock(app, prefix, container_id=prefix+'_panel')
        self.download_menu_panel = CountResultDownloadBlock(app, prefix)
        self._define_page()
    
    def layout(self, validate=False):
        return self._layout
    
    def _define_page(self):
        # define widgets

        # putting the components together 
        rows = html.Div(children = [
            dcc.Store(id=self.update_trigger_id), 
            dbc.Row(html.H3(children = 'Coral Detection Findings Browser', className='mt-3 mb-3')),
            dbc.Row([
                dbc.Col([self.tile_browse_table.get_panel(),], className='col-3'),
                dbc.Col([
                        self.tile_detect_info.get_panel(),
                        dbc.Row(className='mt-4'),
                        self.download_menu_panel.get_panel(),
                        dbc.Row(className='mt-4'),                        
                        self.coral_count_trend.get_panel(),
                        dbc.Row(className='mt-4'),                        
                        self.count_scatter_plot.get_panel(),
                        dbc.Row([self.count_heatmap.get_panel()], className='mx-auto col-12, mt-4'),
                        dbc.Row(className='mt-4'),
                        self.count_image_preview.get_panel(),
                ], id=self.prefix+'_panel', className='col-9', style={'visibility': 'hidden'}), 
            ], className='mx-auto col-12'),
        ])
        
        self._layout = dbc.Container(rows, fluid=True)
        # link up trigger events of the components
        tile_browse_panel_trigger_id = self.tile_browse_table.get_row_selected_trigger_id()

        self.tile_detect_info.register_trigger(self.update_trigger_id)
        self.coral_count_trend.register_trigger(self.update_trigger_id)
        self.count_heatmap.register_trigger(self.update_trigger_id)
        self.count_scatter_plot.register_trigger(self.update_trigger_id)
        self.count_image_preview.register_trigger(self.update_trigger_id)
        self.download_menu_panel.register_trigger(self.update_trigger_id)
        # pass the relevant figure ids to the menu panel for download
        self.download_menu_panel.set_download_figures_funcs(self.coral_count_trend.get_figures_as_list, self.count_heatmap.get_figures_as_list,
                                                            self.count_scatter_plot.get_figures_as_list,
                                                            self.count_heatmap.generate_all_figures_for_tile)
        
        self.app.callback([Output(self.update_trigger_id, 'data'),
                           Output(self.prefix+'_panel', 'style'),],
            [Input(tile_browse_panel_trigger_id, 'data')], prevent_initial_call=True, allow_duplicate=True)(self._update())   

    def _update(self):
        def update(tile_id):
            if tile_id is None:
                raise PreventUpdate
            return (tile_id, {},)
        return update