# Copyright 2024 - Andrew Kwok Fai LUI, 
# Robotics and Autonomous Systems Group, REF, RI
# and the Queensland University of Technology

__author__ = 'Andrew Lui'
__copyright__ = 'Copyright 2024'
__license__ = 'GPL'
__version__ = '1.0'
__email__ = 'ak.lui@qut.edu.au'
__status__ = 'Development'

import os, math, yaml, numbers, pickle
from enum import Enum
from collections import defaultdict
from datetime import datetime
import cv2
import numpy as np

import plotly.express as px

from detector.models.detect import CoralObject
from detector.models import logger

class HeatmapHelper():
    """ HeatmapHelper provides functions to help create heatmaps from a list of CoralObjects

    """
    DEFAULT_COUNT_RANGE = (0,)

    @staticmethod
    def compute_object_count_map(object_list:list, map_size:tuple, class_filter=None, include_invalidated=False, count_range:tuple=None) -> np.ndarray:
        """ compute a 2d object count map as a numpy array from the locatios of the CoralObject in the input object_list

        :param object_list: the input list of CoralObjects
        :type object_list: list
        :param map_size: the dimension (xdim, ydim) of the map 
        :type map_size: tuple
        :param class_filter: a list of classes or a single class to be included in the count, defaults to None
        :type class_filter: Any, optional
        :param include_invalidated: include invalidated objects in the counting, defaults to False
        :type include_invalidated: bool, optional
        :param count_range: a 1-tuple (min) or 2-tuple (min, max) defines the range of count values that are displayed in the heatmap
        :type count_range: tuple, optional
        :return: the count map as a 2d numpy array
        :rtype: np.ndarray
        """
        if class_filter is not None and type(class_filter) == str:
            class_filter = [class_filter]
        # initialize a numpy array of the dimension for the heatmap
        count_map_array = np.zeros(shape=(map_size[1], map_size[0]), dtype=np.uint16)
        count_map_label_array = np.full(shape=(map_size[1], map_size[0]), fill_value=' ', dtype=object)
        # iterates through the object list
        coral_object:CoralObject
        for coral_object in object_list:
            if coral_object.invalidated and not include_invalidated:
                continue
            if class_filter is not None and coral_object.yolo_class not in class_filter:
                continue
            # filter out the out of bound objects
            if coral_object.centre_normalized[0] < 0 or coral_object.centre_normalized[1] < 0 or coral_object.centre_normalized[0] >= 1.0 or coral_object.centre_normalized[1] >= 1.0:
                continue
            # compute the cell index based on the object normalized location
            x, y = int(coral_object.centre_normalized[0] * map_size[0]), int(coral_object.centre_normalized[1] * map_size[1])
            x, y = min(x, map_size[0] - 1), min(y, map_size[1] - 1)
            count_map_array[y, x] += 1
            if count_range is not None and type(count_range) in (tuple, list) and len(count_range) >= 1:
                try:
                    if count_map_array[y, x] > count_range[0]:
                        count_map_label_array[y, x] = str(count_map_array[y, x])
                except:
                    ...
            else:
                count_map_label_array[y, x] = str(count_map_array[y, x])
        return count_map_array, count_map_label_array
    
    @staticmethod
    def generate_plotly_heatmap(count_map_array:np.ndarray, count_map_label_array:np.ndarray=None, title:str=None, fig_size:tuple=None, count_range:tuple=None, color_scale:str=None, show_fig:bool=False, output_file:str=None):
        """ returns a plotly figure object containing the heatmap generated from the given object count map

        :param count_map_array: the coral object count map to be converted into a graphical heatmap
        :type count_map_array: np.ndarray
        :param fig_size: the size (xdin, ydim) of the graphical heatmap in pixels
        :type fig_size: tuple
        :param show_fig: to call fig.show() at the end of the function, defaults to False
        :type show_fig: bool, optional
        :param output_file: the path where the figure is saved to an image file if provided, defaults to None
        :type output_file: str, optional
        :return: a plotly fig object
        :rtype: plotly.graph_objs._figure.Figure
        """
        title = ' ' if title is None else title
        # set the range of count to be considered in colour indexing
        z_min, z_max = 0, 100000000
        if count_range is not None:
            try:
                z_min = count_range[0]
                z_max = count_range[1]
            except:
                ...
            
        # generate the heatmap as a figure
        # fig = px.imshow(count_map_array, text_auto=True, title=title, color_continuous_scale=px.colors.sequential.Viridis, zmin=z_min, zmax=z_max)
        
        color_scale = px.colors.sequential.Viridis if color_scale is None else color_scale
        fig = px.imshow(count_map_array, text_auto=True, title=title, color_continuous_scale=color_scale, zmax=z_max)  
        # set the labels of the cells in the heatmap
        # if count_map_label_array is not None:
        #   fig.update_traces(text=count_map_label_array, texttemplate="%{text}", textfont_size=6)
        if count_map_label_array is not None:
            fig.update_traces(text=count_map_label_array, texttemplate="%{text}")
        # adjust the coloraxis
        # fig.update_layout(coloraxis=dict(cmin=z_min, cmax=z_max))  
                  
        # adjust the figure size if given fig_size
        if fig_size is not None and type(fig_size) in (list, tuple) and len(fig_size) >= 2:
            fig.update_layout(width=fig_size[0], height=fig_size[1])
        
        # write the figure to an image file if required
        if output_file is not None:
            fig.write_image(output_file)
        if show_fig:
            fig.show()
        return fig

# ----------------------------------------------------------------------------------
# Test functions
    
# def test_build_spatial_distribute_model():
#     """ Test loading a CoralOjbectDetectModel from a yaml file
#     """
#     logdata_folder = '/home/qcr/cgras_data/detector/detect/'
#     logger.info('Loading CoralObjectDetect Model from a yaml file')
#     cod_model = test_load_coral_object_detect_model()
#     cod_model.print_info()
#     return cod_model

# if __name__ == '__main__':
#     logdata_folder = '/home/qcr/cgras_data/detector/detect/'
#     output_file = os.path.join(logdata_folder, 'coral_count_heatmap.jpg')
    
#     cod_model = test_build_spatial_distribute_model()
#     object_list = cod_model.get_object_list()

#     map_size = (120, 40)
#     fig_size = (3600, 1200)
#     count_map_array, count_map_label_array = HeatmapHelper.compute_object_count_map(object_list, map_size, None)
#     fig = HeatmapHelper.generate_plotly_heatmap(count_map_array, count_map_label_array, fig_size=(1200, 800), output_file=output_file)
#     input('Press Enter to Quit ')