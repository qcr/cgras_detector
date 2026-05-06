# Copyright 2024 - Andrew Kwok Fai LUI, 
# Robotics and Autonomous Systems Group, REF, RI
# and the Queensland University of Technology

__author__ = 'Andrew Lui'
__copyright__ = 'Copyright 2024'
__license__ = 'GPL'
__version__ = '1.0'
__email__ = 'ak.lui@qut.edu.au'
__status__ = 'Development'

import shutil, zipfile, io
from base64 import b64encode
import pandas as pd
# dash modules
import dash
from dash import html, dcc, Input, Output, State, dash_table, ctx
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate
from cgras_datatools.logging_tools import logger
from detector.model import DETECT_DAO, SampleStatusNames
from detector.models.visualize import CoralObjectMapModel, CoralObjectMapModelHelper

class CountResultDownloadBlock():
    def __init__(self, app, prefix):
        self.app = app 
        self.prefix = prefix = prefix + 'tdm_'
        self.tile_id_store_id = prefix+'tile_id_store'
        # default charting parameters      
        self.default_count_range = None
        # model variables
        self.get_trend_figure_func = None
        self.get_heatmap_figures_list_func = None
        self.get_scatterplot_figures_list_func = None
        self.get_all_heatmap_figures_for_tile_func = None
        # define widgets
        self._count_show_button = dbc.Button('Coral Count Report', id=prefix+'popup_button', color='light', 
                                             href='', external_link=True, target='count_view')
        dropdown = dbc.DropdownMenu(label='Download Data and Report', children=[            
            html.P('Select the format of the data or report to download', className="text-muted px-4 mt-4"),
            dbc.DropdownMenuItem('Data', header=True),
            dbc.DropdownMenuItem('Count Data (Excel)', id=prefix+'download_excel'),
            dbc.DropdownMenuItem('Figure Images (ZIP)', id=prefix+'download_figures'),
            dbc.DropdownMenuItem(divider=True),
            dbc.DropdownMenuItem('Print', header=True),
            self._count_show_button,
        ])
        
        self._panel = html.Div([
            dcc.Store(id=self.tile_id_store_id),
            dropdown,
            dcc.Download(id=prefix+'download_figures_zip_file'),
            ], 
            id=prefix+'main_panel', className='mx-auto')
        
    def register_trigger(self, trigger_id:str):
        # define callback
        self.app.callback([Output(self.tile_id_store_id, 'data'),
                           Output(self.prefix+'popup_button', 'href'),
                            Output(self.prefix+'main_panel', 'style'),],
            [Input(trigger_id, 'data')], prevent_initial_call=True, allow_duplicate=True)(self._update_tile_id()) 
        
    def set_download_figures_funcs(self, get_trend_figure_func, get_heatmap_figures_list_func, get_scatterplot_figures_list_func, get_all_heatmap_figures_for_tile_func=None):
        self.get_trend_figure_func = get_trend_figure_func
        self.get_heatmap_figures_list_func = get_heatmap_figures_list_func
        self.get_scatterplot_figures_list_func = get_scatterplot_figures_list_func
        self.get_all_heatmap_figures_for_tile_func = get_all_heatmap_figures_for_tile_func
        # define callback
        self.app.callback([Output(self.prefix+'download_figures_zip_file', 'data', allow_duplicate=True)],
            [Input(self.prefix+'download_figures', 'n_clicks'),
             State(self.tile_id_store_id, 'data')], prevent_initial_call=True, allow_duplicate=True)(self._download_figures_selected())   
        
        self.app.callback([Output(self.prefix+'download_figures_zip_file', 'data')],
            [Input(self.prefix+'download_excel', 'n_clicks'),
             State(self.tile_id_store_id, 'data')], prevent_initial_call=True, allow_duplicate=True)(self._download_excel_selected())           
        
        
    def get_panel(self):
        return self._panel
    
    # callback for the diskspace table
    def _update_tile_id(self):
        def update_tile_id(tile_id):
            if tile_id is None:
                raise PreventUpdate
            coral_trend_model = DETECT_DAO.get_coral_count_trend_as_df(tile_id) 
            href = f'/popup_viewer?tile_id={tile_id}'
            if len(coral_trend_model) == 0:
                return (tile_id, None, {'visibility': 'hidden'},)
            return (tile_id, href, {'visibility': 'visible'},)
        return update_tile_id

    # callback
    def _download_figures_selected(self):
        def update_datatable(n_clicks, tile_id):
            if not n_clicks or not tile_id:
                raise PreventUpdate
            try:
                image_filebytes_list = []
                image_filename_list = []
                # retrieve figures from the trend panel
                if self.get_trend_figure_func is not None:
                    fig_list = self.get_trend_figure_func()
                    if fig_list:
                        trend_fig = fig_list[0]
                        if trend_fig is not None:
                            img_bytes = trend_fig.to_image(format='png')
                            image_filebytes_list.append(img_bytes)
                            image_filename_list.append(f'{tile_id}_trend_chart.png')
                # retrieve figures from the heatmap panel — generate all samples fresh if possible
                if self.get_all_heatmap_figures_for_tile_func is not None:
                    fig_list = self.get_all_heatmap_figures_for_tile_func(tile_id)
                elif self.get_heatmap_figures_list_func is not None:
                    fig_list = self.get_heatmap_figures_list_func()
                else:
                    fig_list = None
                if fig_list:
                    for index, heatmap_fig in enumerate(fig_list):
                        if heatmap_fig is None:
                            continue
                        img_bytes = heatmap_fig.to_image(format='png')
                        image_filebytes_list.append(img_bytes)
                        image_filename_list.append(f'{tile_id}_heatmap_{index}.png')
                # retrieve figures from the scatter plot panel
                if self.get_scatterplot_figures_list_func:
                    fig_list = self.get_scatterplot_figures_list_func()
                    if fig_list:
                        for index, scatterplot_fig in enumerate(fig_list):
                            if scatterplot_fig is None:
                                continue
                            img_bytes = scatterplot_fig.to_image(format='png')
                            image_filebytes_list.append(img_bytes)
                            image_filename_list.append(f'{tile_id}_scatterplot_{index}.png')
                zip_encoded = self.generate_zip(image_filebytes_list, image_filename_list)
                zip_encoded = b64encode(zip_encoded).decode()
                data = dict(content=zip_encoded, filename=f'{tile_id}_charts.zip', base64=True)
                return (data,)
            except Exception:
                logger.exception('Error generating figures ZIP')
                raise PreventUpdate
        return update_datatable
    
    def generate_zip(self, image_filebytes_list, image_filename_list):
        sink = io.BytesIO()
        with zipfile.ZipFile(sink, mode='w') as zf:
            for image_filebyte, image_filename in zip(image_filebytes_list, image_filename_list):
                zf.writestr(image_filename, image_filebyte)
        zf.close()
        return sink.getvalue()

    # callback
    def _download_excel_selected(self):
        def download_excel_selected(n_clicks, tile_id):
            if not n_clicks or not tile_id:
                raise PreventUpdate
            # build multiple sheets excel 
            with io.BytesIO() as output:
                writer = pd.ExcelWriter(output, engine='xlsxwriter')
                # retrieve the tile info
                tile_info_df = DETECT_DAO.get_tile_info_from_tile_sample(tile_id, to_dataframe=True)
                tile_info_df.to_excel(writer, sheet_name='TileInfo', index=False)
                
                tile_sample_df = DETECT_DAO.list_tile_samples_of_tile_id(tile_id, status=SampleStatusNames.DONE.value)
                tile_sample_df.drop(columns=['spawn_time', 'status', 'priority', 'metadata'], inplace=True)
                tile_sample_df.to_excel(writer, sheet_name='TileSamples', index=False)
                
                coral_count_trend_df = DETECT_DAO.get_coral_count_trend_as_df(tile_id)
                coral_count_trend_df.to_excel(writer, sheet_name='CoralCountTrend', index=False)
                
                # create complex index for tab indices
                tab_index_list = None

                # go through each tile sample id and retrieve the coral detection
                for index, row in coral_count_trend_df.iterrows():
                    tile_sample_id = row['tile_sample_id']
                    age = row['age']
                    # extract tile sample info
                    tile_sample_row = tile_sample_df[tile_sample_df['id'] == tile_sample_id]

                    batch_time = tile_sample_row.iloc[0]['batch_time']
                    # generate detected_object worksheet for the tile_sample_id
                    detected_objects_df = DETECT_DAO.query_detected_objects(tile_sample_id)
                    detected_objects_df.drop(columns=['metadata'], inplace=True)
                    detected_objects_df.insert(loc=1, column='tile_id', value=tile_id)
                    detected_objects_df.insert(loc=2, column='batch_time', value=batch_time)
                    detected_objects_df.insert(loc=3, column='age', value=age)
                    species = tile_info_df.loc[0, 'species']
                    detected_objects_df.insert(loc=4, column='species', value=species)
                    # compute the tab index
                    map_size = (tile_info_df.loc[0, 'tab_ncols'], tile_info_df.loc[0, 'tab_nrows'],)
                    detected_objects_df['tab_xindex'] = detected_objects_df['centre_x'] * map_size[0]
                    detected_objects_df['tab_yindex'] = detected_objects_df['centre_y'] * map_size[1]
                    detected_objects_df['tab_xindex'].clip(upper = map_size[0] - 1, inplace=True)
                    detected_objects_df['tab_yindex'].clip(upper = map_size[1] - 1, inplace=True)
                    detected_objects_df = detected_objects_df.astype({'tab_xindex': int, 'tab_yindex': int})
                    # save to excel
                    detected_objects_df.to_excel(writer, sheet_name=f'Detect-{row["batch_time"][:10]}', index=False)

                    # logger.warning(f'detected_tab_df: {detected_tab_df.index}')
                    # detected_tab_df.index.set_names(['tab_xindex', 'tab_yindex'])
                    # generate group by present_class
                    present_count_df = detected_objects_df.groupby(by=['tab_xindex', 'tab_yindex', 'present_class'], group_keys=False).size()
                    present_count_df = present_count_df.reset_index()
                    present_count_df = present_count_df.pivot(index=['tab_xindex', 'tab_yindex'], columns='present_class')
                    # generate group by coral_class
                    coral_count_df = detected_objects_df.groupby(by=['tab_xindex', 'tab_yindex', 'coral_class'], group_keys=False).size()
                    coral_count_df = coral_count_df.reset_index()
                    coral_count_df = coral_count_df.pivot(index=['tab_xindex', 'tab_yindex'], columns='coral_class')                   
                    # merge the two sets
                    merged_df = pd.merge(present_count_df, coral_count_df, left_index=True, right_index=True)
                    # fix the multiindex of the merged columns
                    new_columns = []
                    merged_df.columns = merged_df.columns.to_flat_index()
                    for index in range(len(merged_df.columns)):
                        new_columns.append(merged_df.columns[index][1])
                    merged_df.columns = new_columns 
                    # add the missing row index
                    if tab_index_list is None:
                        tab_index_list = [(x, y) for y in range(map_size[1]) for x in range(map_size[0])]
                    merged_df = merged_df.reindex(tab_index_list)
                    
                    merged_df = merged_df.fillna(0).astype(int)
                    merged_df = merged_df.reset_index()
                    merged_df.insert(loc=0, column='species', value=species)
                    merged_df.insert(loc=1, column='tile_id', value=tile_id)
                    merged_df.insert(loc=2, column='batch_time', value=batch_time)
                    merged_df.insert(loc=3, column='age', value=age)
                    merged_df.to_excel(writer, sheet_name=f'Detect-Tabs-{row["batch_time"][:10]}', index=False)
                    
                    # generate count map for the tile sample id
                    vt_model = CoralObjectMapModel(tile_sample_id)
                    count_map, count_label_map = vt_model.compute_object_count_map(CoralObjectMapModelHelper.VISCLASS_CORAL['value'], count_range=self.default_count_range)
                    count_map_df = pd.DataFrame(data=count_map)
                    count_map_df.to_excel(writer, sheet_name=f'CM-ALIVE-{row["batch_time"][:10]}', index=False)
                    # iterate through each coral class assigned to the tile_sanmple_id
                    coral_classes_list = DETECT_DAO.list_coral_classes(tile_sample_id=tile_sample_id)
                    for coral_class in coral_classes_list:
                        count_map, count_label_map = vt_model.compute_object_count_map(coral_class, count_range=self.default_count_range)
                        count_map_df = pd.DataFrame(data=count_map)
                        count_map_df.to_excel(writer, sheet_name=f'CM-{coral_class}-{row["batch_time"][:10]}', index=False)                        
                
                # latest_tile_sample_id = coral_count_trend_df.iloc[-1]['tile_sample_id']
                # vt_model = CoralObjectMapModel(latest_tile_sample_id)
                # count_map, count_label_map = vt_model.compute_object_count_map(CoralObjectMapModelHelper.VISCLASS_CORAL['value'], count_range=self.default_count_range)
                # count_map_df = pd.DataFrame(data=count_map)
                # count_map_df.to_excel(writer, sheet_name=f'CountMap-{coral_count_trend_df.iloc[-1]["batch_time"][:11]}', index=False)
                writer.close()
                output.seek(0)
                # wrap up the excel file and send for download
                output_encoded = b64encode(output.getvalue()).decode()
                data = dict(content=output_encoded, filename=f'{tile_id}_data.xlsx', base64=True)            
                return (data,)
            
        return download_excel_selected