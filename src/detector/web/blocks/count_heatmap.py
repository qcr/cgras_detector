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
import pandas as pd
# dash modules
import dash
from dash import html, dcc, Input, Output, State, dash_table, ctx
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate
from cgras_datatools.logging_tools import logger
from detector.model import DETECT_DAO, CONFIG, SystemConfigNames
from detector.models.visualize import CoralObjectMapModel, CoralObjectMapModelHelper
from detector.models.heatmap_tools import HeatmapHelper

class CountHeatmapBlock():
    def __init__(self, app, prefix:str):
        self.app = app 
        self.prefix = prefix = prefix + 'hcb_'
        self.update_trigger_id = 'tile_sample_id_update_trigger'
        # default charting parameters
        self.default_style = {'visibility': 'hidden'}
        self.default_config = {'staticPlot': True}       
        self.reversed_heatmaps = False
        self.default_count_range = HeatmapHelper.DEFAULT_COUNT_RANGE
        # model variables
        # self.current_tile_id = None
        self.coral_trend_model = self.output_model = None
        self.heatmap_figures_list = None
        self.class_options = None
        self.current_filter_class = None
        self.current_show_count_threshold = 0
        # define widgets
        heatmap_show_label_slider_max = CONFIG.get(SystemConfigNames.HEATMAP_SHOW_LABEL_SLIDER_MAX, 30)
        _sample_select_datatable = dash_table.DataTable(id=prefix+'sample_select_datatable', row_selectable=False, cell_selectable=True, style_cell={'fontSize': 14})
        _count_threshold_slider = dcc.Slider(0, heatmap_show_label_slider_max, 1, value=0, id=prefix+'count_threshold_slider',
                                               marks = {i: f'{i}' for i in range(0, heatmap_show_label_slider_max, 5)}, tooltip={"always_visible": False,},className='mt-5')
        # define the main panel
        self._panel = html.Div(id=prefix+'top_panel', children=[
                dcc.Store(self.update_trigger_id),
                html.H4(dbc.Badge('SPATIAL DISTRIBUTION OF AN OBJECT CLASS ON THE TILE', className='ms-1 me-2', color='white', text_color='secondary')),
                dbc.Row([
                    dbc.Col([
                            html.P('Select object class', className='mt-5'),
                            dcc.Dropdown(id=prefix+'class_list_dropdown', searchable=False, clearable=False, style={'fontSize': 14}),
                            html.P('Count range not to be shown', className='mt-5'),
                            _count_threshold_slider,
                            html.P('Click and select a sample below to compare with the latest sample', style={'margin-top': '40px'}),
                            _sample_select_datatable,
                            dbc.Button('Reverse Order', id=prefix+'reverse_button', color='secondary', className='mt-5'),
                        ], className='col-2', style={'background-color': '#dddddd'}),
                    dbc.Col(
                        html.Div(id=prefix+'chart_panel')
                        , className='col-10'),                    
                ], className='mx-auto col-12'),            
            ], className='text-center')
        
        # define callbacks
        self.app.callback([Output(prefix+'chart_panel', 'children', allow_duplicate=True),
                           Output(prefix+'reverse_button', 'n_clicks'),],
                        [Input(prefix+'reverse_button', 'n_clicks'),
                         State(prefix+'chart_panel', 'children'),], 
            prevent_initial_call=True)(self._reverse_button_pressed()) 
        
        self.app.callback([Output(self.prefix+f'chart_panel', 'children', allow_duplicate=True)],
                        [Input(self.prefix+'class_list_dropdown', 'value'),
                         Input(self.prefix+'sample_select_datatable', 'active_cell'),
                         Input(self.prefix+'count_threshold_slider', 'value')], prevent_initial_call=True)(self._update_chart_panel())   
    
    # function called by the parent to link up a trigger from another block to update the heatmap block 
    def register_trigger(self, trigger_id:str):
        # build the output list
        output_list = [Output(self.prefix+f'top_panel', 'style', allow_duplicate=True),  
                    Output(self.prefix+'sample_select_datatable', 'data', allow_duplicate=True),
                    Output(self.prefix+'sample_select_datatable', 'active_cell'),
                    Output(self.prefix+'sample_select_datatable', 'selected_cells'),
                    ]
        input_list = [Input(trigger_id, 'data'),]
        # define callbacks for the datatable data
        self.app.callback(output_list, input_list, prevent_initial_call=True, allow_duplicate=True)(self._update_panel())
        
        self.app.callback([Output(self.prefix+'class_list_dropdown', 'options', allow_duplicate=True),
                            Output(self.prefix+'class_list_dropdown', 'value', allow_duplicate=True)],
                         [Input(trigger_id, 'data'),], prevent_initial_call=True)(self._update_class_dropdown())         
    
    # return the main panel of this block
    def get_panel(self):
        return self._panel
    
    # return the heatmap figures as a list for export
    def get_figures_as_list(self):
        return self.heatmap_figures_list

    # generate heatmaps for all tile samples belonging to tile_id (used by ZIP download)
    def generate_all_figures_for_tile(self, tile_id):
        coral_trend_model, _ = self._get_coral_trend_model(tile_id)
        if len(coral_trend_model) == 0:
            return []
        if self.class_options is None:
            self.class_options = CoralObjectMapModelHelper.get_class_options_list()
        if not self.class_options:
            return []
        filter_class = self.current_filter_class or self.class_options[0]['value']
        latest_index = len(coral_trend_model) - 1
        _, figures_list = self._generate_figures_list(coral_trend_model, filter_class, compare_to_index=latest_index,
                                                      show_count_threshold=self.current_show_count_threshold)
        return figures_list

    # generate the coral count trend model
    def _get_coral_trend_model(self, tile_id):
        coral_trend_model = DETECT_DAO.get_coral_count_trend_as_df(tile_id) 
        coral_trend_model['batch_time'] = pd.to_datetime(coral_trend_model['batch_time']).dt.date
        coral_trend_model['batch_time'] = coral_trend_model.apply(lambda row: f'{row["batch_time"]} ({row["age"]} days old)', axis=1)
        output_model = coral_trend_model[['batch_time']].copy()
        output_model.columns = ['Sample Date']
        if len(output_model) > 0:
            output_model.loc[len(output_model)] = ['Whole History']
        return coral_trend_model, output_model

    def _generate_heatmap(self, tile_sample_id, filter_class, title:str=None, show_count_threshold:int=None):
        count_range = (show_count_threshold, ) if show_count_threshold is not None else self.default_count_range
        # generate the object count maps as dataframes
        vt_model = CoralObjectMapModel(tile_sample_id)
        # count_map stores the counts as integers and count_label_map is the string representation
        count_map, count_label_map = vt_model.compute_object_count_map(filter_class, count_range=count_range)
        # find the sum of all cells
        count_total = count_map.sum()
        title += f' ({count_total})'
        # generate the heatmap for the given tile_sample_id
        color_scale = CONFIG.get(SystemConfigNames.HEATMAP_COLOUR_SCALE, None)
        fig = HeatmapHelper.generate_plotly_heatmap(count_map, count_label_map, title=title, fig_size=(640, 600,), count_range=count_range, color_scale=color_scale)
        return fig, count_map
    
    # internal function for generating the heatmaps based on the given filter_class, the indices of tile samples to show in additional to the most recent one, and the count show threshold 
    def _generate_figures_list(self, coral_trend_model, filter_class, compare_to_index:str=None, show_count_threshold:int=None):
        figures_graph_list = []
        figures_list = []
        latest_index = len(coral_trend_model) - 1
        max_count = 0
        for index in range(latest_index, -1, -1):
            if index == len(coral_trend_model) - 1 or (compare_to_index is not None and (compare_to_index == index or compare_to_index >= latest_index)):
                the_sample = coral_trend_model.iloc[index]
                the_sample_id = the_sample['tile_sample_id']
                title = f'Captured on {the_sample["batch_time"]}'
                if index == latest_index:
                    title += ' (Latest)'    
                if self.class_options is not None:
                    for option in self.class_options:
                        if option['value'] == filter_class:
                            title += f': # {option["label"]}s'
                            break
                    
                fig, count_map = self._generate_heatmap(the_sample_id, filter_class, title=title, show_count_threshold=show_count_threshold) 
                # find the scale
                max_count = max(max_count, count_map.max())                   
                fig.update_layout(
                    margin=dict(l=5, r=5, b=10, t=50, pad=0),
                    title=dict(font=dict(size=18, weight='bold'), yref='paper'),
                )
                figures_list.append(fig)
                # children_list.append(html.H4(dbc.Badge(title, className='ms-1 me-2', color='white', text_color='primary')))
                figures_graph_list.append(dcc.Graph(figure=fig, config=self.default_config, style={'visibility': 'visible'}),)
        # use the max_count to set the maximum value for the scale of the heatmap
        for fig in figures_list:
            fig.update_layout(coloraxis=dict(cmax=max_count))
        return figures_graph_list, figures_list
    
    def _update_class_dropdown(self):
        def update_class_dropdown(tile_id):
            # get options for the dropdown
            self.class_options = CoralObjectMapModelHelper.get_class_options_list()
            value = self.class_options[0]['value'] if self.class_options is not None and len(self.class_options) > 0 else None
            self.current_filter_class = value
            return (self.class_options, value,)
        return update_class_dropdown

    def _update_panel(self):
        def update_panel(tile_id):
            if tile_id is None:
                raise PreventUpdate
            # the update is due to a new tile_id selected, or query again if the coral trend model has no data
            # if self.current_tile_id is None or tile_id != self.current_tile_id or self.coral_trend_model is None or len(self.coral_trend_model) == 0:
            #     self.current_tile_id = tile_id
            # update the coral_trend_model
            self.coral_trend_model, self.output_model = self._get_coral_trend_model(tile_id)
            # reset the reverse heatmap display state
            self.reversed_heatmaps = False

                
            if len(self.coral_trend_model) > 0:
                return [{}, self.output_model.to_dict('records'), None, []]   
            else:
                return [{'visibility': 'hidden'}, self.output_model.to_dict('records'), None, []]                 
        return update_panel
    
    def _update_chart_panel(self):
        def update_chart_panel(filter_class, active_cell, slider_value):
            graph_list = None
            compare_to_index = None
            # update the latest figure
            if len(self.coral_trend_model) > 0:
                if active_cell is not None:
                    compare_to_index = active_cell['row']
                # persist current UI state for use by export
                self.current_filter_class = filter_class
                self.current_show_count_threshold = slider_value if slider_value is not None else 0
                # set the slider value as the show count threshold
                show_count_threshold = slider_value
                # generate history of heatmaps
                graph_list, self.heatmap_figures_list = self._generate_figures_list(self.coral_trend_model, filter_class, compare_to_index, show_count_threshold)
                # if the state of heatmap display list is reversed, reverse the figures now
                if self.reversed_heatmaps:
                    graph_list.reverse()
                    self.heatmap_figures_list.reverse()
                return [graph_list]   
            else:
                return [None,]           
        return update_chart_panel    

    # callback for the reverse button
    def _reverse_button_pressed(self):
        def reverse_button_pressed(n_clicks, graph_list):
            if n_clicks == 1 and graph_list is not None:
                # update the state variable
                self.reversed_heatmaps = not self.reversed_heatmaps
                # reverse the current figures in the list to update the display
                graph_list.reverse()
                self.heatmap_figures_list.reverse()
                return (graph_list, 0,)
            time.sleep(1.0)
            return (dash.no_update, 0,)
        return reverse_button_pressed