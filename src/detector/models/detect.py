# Copyright 2024 - Andrew Kwok Fai LUI, 
# Robotics and Autonomous Systems Group, REF, RI
# and the Queensland University of Technology

__author__ = 'Andrew Lui'
__copyright__ = 'Copyright 2024'
__license__ = 'GPL'
__version__ = '1.0'
__email__ = 'ak.lui@qut.edu.au'
__status__ = 'Development'

import os, sys, math, yaml, numbers, random, time, traceback
from enum import Enum
from collections import defaultdict
from datetime import datetime
import cv2
import numpy as np

from detector.models.yolo_detector import YoloObjectDetector, YoloResult, ObjectType
from detector.models import logger, ModelsConfigNames, DetectorFailed, DetectorAborted, DetectorCancelled, DetectorExceptionCodes
from detector.dao_detect import CoralObject, ClassHierarchyPresentation, ClassHierarchyCoral

from cgras_datatools.opencv_tools import CompareTools


class CoralObjectDetectModel():
    # constant
    ANNOTATED_WHOLE_RECO_IMAGE_FILENAME = 'rotated_whole_reco_image_annotated.jpg'
    ANNOTATED_WHOLE_RECO_ORIGINAL_SCALE_IMAGE_FILENAME = 'rotated_whole_reco_original_image_annotated.jpg'
    """ CoralDetectorModel uses an object detector to extract a list of objects detected in a 2d grid of images that represent a full coral aquaculture tile. The images which may overlap with one another are
        arranged in a 2d grid that implies the location with reference to the tile. The class uses an ImageReconstructionModel to map locations on individual images to  
    """
    def __init__(self, images_2d_list:list, yolo_detect_model_list:list, map_bbox_image_fn, map_normalize_bbox_tile_fn, tile_size:tuple, progress_cb=None, **kwargs):
        """ the constructor

        :param images_2d_list: A list of lists of images, each of which can be image paths (str typed) or image pixels (np.ndarray), arranged in a 2D grid
        :type images_2d_list: list
        :param yolo_detect_model_list: One or more YoloObjectDetector model to be used, which should be suitable for the coral species found in the images
        :type yolo_detect_model_list: list of YoloObjectDetector
        :param map_bbox_image_fn: A function that maps a bbox in (image_grid_x, image_grid_y, bbox) to the space reconustructed image
        :type map_bbox_image_fn: A function definition (image_grid_x, image_grid_y, bbox:list) -> bbox:list
        :param map_normalize_bbox_tile_fn: A function that maps a bbox in reconstructed image space to the tile space normalized to (0, 1)
        :type map_normalize_bbox_tile_fn: A function definition (bbox:list) -> bbox:list  
        :param tile_size: The pixel size of the tile
        :type tile_size: 2-tuple 
        """
        # ignore the constructor if the object is loaded from yaml file
        if images_2d_list is None:
            return
        # setup progress tracking variables
        self.progress_cb = progress_cb
        self.num_images = len(images_2d_list) * len(images_2d_list[0])
        self.count_images_completed = 0
        # keep input parameters
        self.images_2d_list = images_2d_list
        self.yolo_detect_model_list = yolo_detect_model_list
        self.map_bbox_image_fn = map_bbox_image_fn
        self.map_normalize_bbox_tile_fn = map_normalize_bbox_tile_fn
        self.tile_size = tile_size
        self.params = kwargs

        # extract other keyword parameters - operational
        self.blob_overlap_pix = kwargs.get(ModelsConfigNames.COD_BLOB_OVERLAP_PIX.value, 0)
        self.coral_child_min_overlap_ratio = kwargs.get(ModelsConfigNames.COD_CORAL_CHILD_MIN_OVERLAP_RATIO.value, 0.25)
        self.debug_blob_images = kwargs.get(ModelsConfigNames.COD_DEBUG_BLOB_IMAGES.value, True)
        self.jpeg_quality = kwargs.get(ModelsConfigNames.OUTPUT_JPEG_QUALITY.value, 95)

        # extract other keyword parameters - output cached data and debug information
        self.logdata_folder = kwargs.get(ModelsConfigNames.LOGDATA_FOLDER.value, None)
        self.cod_model_cache_filename = kwargs.get(ModelsConfigNames.COD_MODEL_FILENAME.value, f'coral_object_detect_model.yaml')

        # extract keyword parameters - whether to label POLYP_KEYPART as ALIVE or MASK
        self.mask_polyp_keypart = kwargs.get(ModelsConfigNames.COD_MASK_POLYP_KEYPART.value, False)
        # init model parameters
        self.object_list = None     # the master list of detected objects
        self.image_grid_size = (len(self.images_2d_list[0]), len(self.images_2d_list))
        self.object_list_of_images = dict()  # the list of detected objects of each image
        self.object_class_names = None
        self.annotated_blob_filename_dict_lists = dict()        # indexed lists of annotated blob filenames with each list indexed by the image location
        self.annotated_image_filename_dict_list = []            # list of annotated image filenames 
        # init the current COD model
        self.cod_model = None
        # model parameters: abort
        self.to_cancel = False
        
    def build(self):
        # step 1: iterate through each image in the 2d list of images
        for row_index, row_1d_image_list in enumerate(self.images_2d_list):
            for col_index, image in enumerate(row_1d_image_list):                
                # if the attribute progress_cb is set, call the progress_cb to record the progress
                if hasattr(self, 'progress_cb') and self.progress_cb is not None:
                    self.progress_cb((self.count_images_completed, self.num_images))
                time.sleep(0.1)    # sleep for a short while to release cpu
                if self.to_cancel:  # stop processing if abort signal is recieved
                    raise DetectorCancelled(DetectorExceptionCodes.CANCELLED_BY_SYSTEM, 'Received an cancel command from the system')
                # compute the coral object detect model (long process)
                self.cod_model = CoralObjectDetectImageModel(image, col_index, row_index, self.yolo_detect_model_list, self.map_bbox_image_fn, self.map_normalize_bbox_tile_fn, **self.params)
                self.cod_model.build()
                
                index = (col_index, row_index)
                # extract the list of detected objects from the cod_model
                self.object_list_of_images[index] = self.cod_model.get_object_list(include_invalidated=True)
                # record the class names if not already known
                if self.object_class_names is None:
                    self.object_class_names = self.cod_model.get_object_class_names() 
                # record the list of filenames of the annotated images that have been generated
                self.annotated_image_filename_dict_list.append(self.cod_model.get_annotated_image_filename_dict())
                self.annotated_blob_filename_dict_lists[index]  = self.cod_model.get_annotated_blob_filename_dict_list()
                # increase the counter
                self.count_images_completed += 1 
        # make the last progress callback
        if hasattr(self, 'progress_cb') and self.progress_cb is not None:
            self.progress_cb((self.count_images_completed, self.num_images))
        # step 3: resolve duplicate objects in the overlapping regions between images
        logger.info(f'DUPLICATE OBJECT REMOVAL between images of the tile') 
        self.num_invalidated_objects = self._invalidate_duplicate_objects(self.object_list_of_images, self.image_grid_size)
        # step 2: resolve the presentation classes of the objects
        logger.info(f'RESOLVE PRESENTATION CLASS of valid objects on the whole tile') 
        self._resolve_presentation_class_of_objects(self.object_list_of_images, self.image_grid_size)
        # step 4: extract the objects from individual lists of images into a single list
        self.object_list = self._merge_object_lists(include_invalidated=True)

        # step 5: clear data if not needed for model inference
        self.images_2d_list = None
        self.yolo_detect_model_list = None  

    def _merge_object_lists(self, include_invalidated=False) -> list:
        """ internal function to return as a single list all the coral objects detected in the 2d grid of images, the duplicated objects due to overlapping regions between neighbouring images are flagged invalidated.
            The parameter include_invalidated controls whether to also return the invalidated object in the list
        """
        # iterate through lists of detected objects from all the images
        final_objects_list = []
        for index in self.object_list_of_images.keys():
            # if invlidated objects are included, just merge two lists
            if include_invalidated:
                final_objects_list.extend(self.object_list_of_images[index])
            else:
                # iterate through each object
                for coral_object in self.object_list_of_images[index]:
                    if not coral_object.invalidated:
                        final_objects_list.append(coral_object)
        return final_objects_list
    
    def cancel_build(self):
        """ call to abort the computing of this CoralObjectDetectModel
        """
        self.to_cancel = True
        if hasattr(self, 'cod_model'):
            if self.cod_model:
                self.cod_model.cancel_build()

    def get_progress(self) -> tuple:
        """ returns the progress of computing this CoralObjectDetectModel

        :return: a tuple of two integers, (the number of images complete, total number of images to analyze)
        :rtype: tuple
        """
        if hasattr(self, 'num_images') and hasattr(self, 'count_images_completed'):
            return (self.count_images_completed, self.num_images)

    def get_annotated_blob_filename_dict_lists_as_dict(self) -> dict:
        """ Returns a dict contains lists of file names of images showing annotation of blobs for debug purpose and the dict is indexed by image location (col, row)

        :return: the list of image file names
        """
        return self.annotated_blob_filename_dict_lists
    
    def get_annotated_image_filename_dict_list(self) -> list:
        """ Returns the list of file names of images showing annotation of images for debug purpose

        :return: the file name
        """
        return self.annotated_image_filename_dict_list   


    def get_object_list(self, include_invalidated=False) -> list:
        """ returns the list of CoralObject objects

        :return: the finalized list of CoralObject, which may include invalidated object
        :rtype: list
        """
        if include_invalidated:
            return self.object_list
        validate_objects_list = []
        coral_object:CoralObject
        for coral_object in self.object_list:
            if not coral_object.invalidated:
                validate_objects_list.append(coral_object)
        return validate_objects_list
    
    def get_num_objects(self) -> int:
        """ return the number of objects, which may include invalidated objects

        :return: the number of objects, which may include invalidated objects
        :rtype: int
        """
        return len(self.object_list)
    
    def get_num_invalidated_objects(self) -> int:
        """ returns the number of objects marked as invalidated, due to being a duplicate

        :return: the number of objects marked as invalidated, due to being a duplicate
        :rtype: int
        """
        return self.num_invalidated_objects
    
    def get_object_class_names(self) -> dict:
        """ returns a map of class_id, class_name for different classes of coral objects as specified by the object detection model

        :return: a map of class_id, class_name
        :rtype: dict
        """
        return self.object_class_names
    
    def get_tile_size(self) -> tuple:
        """ returns the size (xdim, ydim) of the tile in pixels

        :return: the size (xdim, ydim) of the tile in pixels
        :rtype: tuple
        """
        return self.tile_size

    def print_info(self) -> None:
        """ display the key parameters of the CoralObjectDetectModel
        """
        logger.info(f'Number of objects: {len(self.object_list)}')
        logger.info(f'Number of invalidated objects: {self.num_invalidated_objects}')
        logger.info(f'Number of unique objects: {len(self.object_list) - self.num_invalidated_objects}')
        logger.info(f'Tile size: {self.tile_size}')
        logger.info(f'Object Class Names: {self.object_class_names}')

    def _save_object_list_of_images(self, cache_file:str):
        """ Save the data associated with the detected coral objects to a yaml file

        :param cache_file: path to the target yaml file
        :type cache_file: str
        """
        data = {
            # 'object_list_of_images': self.object_list_of_images,
            'image_grid_size': self.image_grid_size,
            'resolved_object_list': self.object_list,
            'num_resolved_objects': len(self.object_list),
            'num_duplicate_objects': self.num_invalidated_objects,
            'object_class_names': self.object_class_names,
            'tile_size': self.tile_size,
            'annotated_blob_filename_dict_lists': self.annotated_blob_filename_dict_lists,
            'annotated_image_filename_dict_list': self.annotated_image_filename_dict_list,
        }
        with open(cache_file, 'w') as outfile:
            yaml.dump(data, outfile, Dumper=yaml.Dumper)
    
    def _load_object_list_of_images(self, cache_file:str) -> list:
        """ Load from a yaml file the data associated with the detected coral objects and restore them to the data structures of this object

        :param cache_file: path to the target yaml file
        :type cache_file: str
        """
        try:
            with open(cache_file, 'r') as infile:
                data = yaml.load(infile, Loader=yaml.Loader)
            # self.object_list_of_images = data['object_list_of_images']
            self.image_grid_size = data['image_grid_size']
            self.object_list = data['resolved_object_list']
            self.num_invalidated_objects = data['num_duplicate_objects']
            self.object_class_names = data['object_class_names']
            self.tile_size = data['tile_size']
            self.annotated_blob_filename_dict_lists =data['annotated_blob_filename_dict_lists']
            self.annotated_image_filename_dict_list = data['annotated_image_filename_dict_list']
        except (Warning, Exception) as e:
            # logger.warning(f'{type(self).__name__}: Failed to load object list cache file {cache_file}\n{e}')
            raise e
    
    def annotate_whole_reco_image_with_objects(self, rotated_reco_image:np.ndarray, image_scale:float, tile_origin_in_px:tuple, tile_size_in_px:tuple, 
                                               output_image_file:str, line_width:int=1, font_size:float=0.6, draw_coral_class:bool=False) -> bool:
        # define the colours for object annotation
        present_color_table = {
            ClassHierarchyPresentation.ALIVE_CORAL.value: (128, 256, 128,),
            ClassHierarchyPresentation.DEAD_CORAL.value: (64, 64, 255),
            ClassHierarchyPresentation.OTHER.value: (255, 32, 64),
        }
        coral_color_table = {
            ClassHierarchyCoral.POLYP_MULTI.value: (32, 32, 32,),
            ClassHierarchyCoral.POLYP_SINGLE.value: (64, 64, 192),
            ClassHierarchyCoral.POLYP_KEYPART.value: (64, 64, 192), 
            ClassHierarchyCoral.DEAD_CORAL.value: (0, 0, 0),
            ClassHierarchyCoral.OTHER.value: (32, 192, 32),   
        } 
        # draw the color legend
        y, ystep = 20, 30
        for coral_class in present_color_table:
            cv2.rectangle(rotated_reco_image, (10, y), (30, y + 20), present_color_table[coral_class], line_width)
            cv2.putText(rotated_reco_image, f'{coral_class}', (40, y + 10), cv2.FONT_HERSHEY_PLAIN, font_size, present_color_table[coral_class], 1)
            y += ystep

        # draw the images grid lines
        n_cols, n_rows = 20, 20  # default for debug
        grid_size_x, grid_size_y = tile_size_in_px[0] / n_cols, tile_size_in_px[1] / n_rows
        for row in range(n_rows):
            for col in range(n_cols):
                start_x, start_y = tile_origin_in_px[0] + col * grid_size_x, tile_origin_in_px[1] + row * grid_size_y, 
                end_x, end_y = start_x + grid_size_x - 1, start_y + grid_size_y - 1
                start_x, start_y = int(start_x * image_scale), int(start_y * image_scale)
                end_x, end_y = int(end_x * image_scale) + 1, int(end_y * image_scale) + 1
                cv2.rectangle(rotated_reco_image, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (0, 0, 255), line_width)
                centre_x, centre_y = int((start_x + end_x)/2), int((start_y + end_y)/2)
                cv2.putText(rotated_reco_image, f'{col},{row}',
                            (centre_x, centre_y), cv2.FONT_HERSHEY_PLAIN, font_size, (0, 0, 255), int(font_size + 0.5))   
        # iterate through the coral objects and annotate each on the rotated_reco_image
        coral_object:CoralObject 
        for coral_object in self.object_list:
            if not coral_object.invalidated:
                bbox_color = present_color_table.get(coral_object.present_class, None)
                if bbox_color is None:
                    continue
                bbox_in_tile = list(coral_object.bbox_in_tile)
                bbox_in_tile[0] = int((bbox_in_tile[0] + tile_origin_in_px[0]) * image_scale)
                bbox_in_tile[1] = int((bbox_in_tile[1] + tile_origin_in_px[1]) * image_scale)
                bbox_in_tile[2] = int((bbox_in_tile[2] + tile_origin_in_px[0]) * image_scale)
                bbox_in_tile[3] = int((bbox_in_tile[3] + tile_origin_in_px[1]) * image_scale)
                cv2.rectangle(rotated_reco_image, (int(bbox_in_tile[0]), int(bbox_in_tile[1])), (int(bbox_in_tile[2]), int(bbox_in_tile[3])), bbox_color, line_width)          
                text_color = coral_color_table.get(coral_object.coral_class, None)
                if bbox_color is None:
                    continue
                text_pos = (int(bbox_in_tile[0]) + random.randint(-30, 30), int(bbox_in_tile[3]) + random.randint(15, 30))
                # text_to_draw = coral_object.index_str.replace(' ', '')
                text_to_draw = coral_object.yolo_class
                text_to_draw = f'{coral_object.centre_normalized[0]:.3f},{coral_object.centre_normalized[1]:.3f} ({coral_object.confidence:.1f})'
                # text_to_draw the text
                cv2.putText(rotated_reco_image, f'{text_to_draw}', text_pos,
                            cv2.FONT_HERSHEY_PLAIN, max(font_size * 0.6, 0.6), (0, 0, 0), int(font_size + 0.5))   
                if draw_coral_class:
                    cv2.putText(rotated_reco_image, f'{coral_object.coral_class}', (text_pos[0], int(text_pos[1] + font_size * 15)),
                                cv2.FONT_HERSHEY_PLAIN, max(font_size * 0.6, 0.6), (0, 0, 0), int(font_size + 0.5))                           
                            
        if not cv2.imwrite(output_image_file, rotated_reco_image, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]):
            return False
            # raise DetectorExceptionCodes(DetectorExceptionCodes.OS_ERROR, f'Failed to save rotated annotated image to {output_image_file}')
        return True
    
    @staticmethod
    def draw_tab_grid_on_image(image: np.ndarray, image_scale: float, tile_origin_in_px: tuple, tile_size_in_px: tuple,
                               n_cols: int, n_rows: int, output_image_file: str, line_width: int = 1, font_size: float = 0.6,
                               jpeg_quality: int = 95) -> bool:
        grid_size_x = tile_size_in_px[0] / n_cols
        grid_size_y = tile_size_in_px[1] / n_rows
        for row in range(n_rows):
            for col in range(n_cols):
                start_x = tile_origin_in_px[0] + col * grid_size_x
                start_y = tile_origin_in_px[1] + row * grid_size_y
                end_x, end_y = start_x + grid_size_x - 1, start_y + grid_size_y - 1
                sx, sy = int(start_x * image_scale), int(start_y * image_scale)
                ex, ey = int(end_x * image_scale) + 1, int(end_y * image_scale) + 1
                cv2.rectangle(image, (sx, sy), (ex, ey), (0, 0, 255), line_width)
                cx, cy = (sx + ex) // 2, (sy + ey) // 2
                cv2.putText(image, f'{col},{row}', (cx, cy), cv2.FONT_HERSHEY_PLAIN, font_size, (0, 0, 255), int(font_size + 0.5))
        return cv2.imwrite(output_image_file, image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

    @classmethod
    def from_yaml_file(cls, object_file:str):
        """ Create a CoralObjectDetectModel object from a yaml file

        :param object_file: the path to the yaml file
        :type object_file: str
        :return: the new CoralObjectDetectModel object
        :rtype: CoralObjectDetectModel
        """
        cod_model = cls(None, None, None, None, None)
        cod_model._load_object_list_of_images(object_file)
        return cod_model
    
    def _filter_valid_objects(self, object_list:list):
        return [obj for obj in object_list if not obj.invalidated]

    def _invalidate_duplicate_objects(self, object_list_of_images:dict, images_grid_size:tuple) -> int:
        """ a generic function for invalidating objects associated with every image in the 2d grid of images that are found to be duplicates.

        :param object_list_of_images: a 2d grid of object lists, each of which stores objects found from the corresponding image in the 2d grid of images
        :type object_list_of_images: dict
        :param images_grid_size: the dimension of the 2d grid of object list, which equals to the 2d grid of images
        :type images_grid_size: tuple
        :param max_displacement: the threshold distance beyond which two objects can be considered as duplicates
        :type max_displacement: float
        :return: the total number of objects marked as invalidated by this function
        :rtype: int
        """
        total_duplicates_removed = 0
        overlap_sets_list = []
        # iterate through each row and then each grid locations along a row
        for row_index in range(images_grid_size[1]):
            for col_index in range(images_grid_size[0]):
                # abort the process
                if self.to_cancel:
                    raise DetectorCancelled(DetectorExceptionCodes.CANCELLED_BY_SYSTEM, 'Received an cancel command from the system')
                if col_index < images_grid_size[0] - 1:
                    # resolve diplicate between (col_index, row_index) and (col_index + 1, row_index)
                    object_list_index_1, object_list_index_2 = (col_index, row_index), (col_index + 1, row_index)
                    CoralObjectListHelper.extract_overlap_objects_same_class(self._filter_valid_objects(object_list_of_images[object_list_index_1]), 
                                                                                self._filter_valid_objects(object_list_of_images[object_list_index_2]), overlap_sets_list)
                    if row_index > 0:
                        # resolve diplicate between (col_index, row_index) and (col_index + 1, row_index - 1)
                        object_list_index_1, object_list_index_2 = (col_index, row_index), (col_index + 1, row_index - 1)
                        CoralObjectListHelper.extract_overlap_objects_same_class(self._filter_valid_objects(object_list_of_images[object_list_index_1]), 
                                                                                    self._filter_valid_objects(object_list_of_images[object_list_index_2]), overlap_sets_list)
                if row_index >= images_grid_size[1] - 1:
                    continue
                # resolve diplicate between (col_index, row_index) and (col_index, row_index + 1)
                object_list_index_1, object_list_index_2 = (col_index, row_index), (col_index, row_index + 1)
                CoralObjectListHelper.extract_overlap_objects_same_class(self._filter_valid_objects(object_list_of_images[object_list_index_1]), 
                                                                            self._filter_valid_objects(object_list_of_images[object_list_index_2]), overlap_sets_list)
                
                if col_index < images_grid_size[0] - 1:
                    # resolve diplicate between (col_index, row_index) and (col_index + 1, row_index + 1)
                    object_list_index_1, object_list_index_2 = (col_index, row_index), (col_index + 1, row_index + 1)
                    CoralObjectListHelper.extract_overlap_objects_same_class(self._filter_valid_objects(object_list_of_images[object_list_index_1]), 
                                                                                self._filter_valid_objects(object_list_of_images[object_list_index_2]), overlap_sets_list)
                    
  
        # go through the overlap sets one at a time
        for overlap_set in overlap_sets_list:
            # validate if all objects in the set are of the same class
            # find the one with the largest size and mark all other invalidate
            obj:CoralObject
            largest_object:CoralObject = max(overlap_set, key=lambda x: x.size[0] * x.size[1])
            for obj in overlap_set:
                if obj == largest_object:
                    obj.invalidated = False
                else:
                    obj.invalidated = True
                    total_duplicates_removed += 1    
        logger.info(f'Total number of duplicates removed from overlapping regions between images: {total_duplicates_removed}')    
        return total_duplicates_removed     
    
    def _resolve_presentation_class_of_objects(self, object_list_of_images:dict, images_grid_size:tuple) -> int:
        """ annotate the objects in the parameter object_list with a presentation class from the coral classes
        
            the algorithm:
                go through each image to collect inside sets composing of valid objects
                    if one object is inside another object, and the two objects are one the following cases, add to inside_sets_list and set inside_of to True
                    - POLYP_SINGLE inside POLYP_MULTI
                    - POLYP_KEYPART inside POLYP_SINGLE
                    - POLYP_KEYPART inside POLYP_MULTI
                    - DEAD_CORAL inside any of the above                    
                    
                iterate through the inside sets:
                    - POLYP_SINGLE inside POLYP_MULTI: set the child object to ClassHierarchyPresentation.NIL and the parent to ClassHierarchyPresentation.ALIVE_CORAL
                    - POLYP_KEYPART inside POLYP_SINGLE: set the child object to ClassHierarchyPresentation.NIL and the parent to ClassHierarchyPresentation.ALIVE_CORAL
                    - POLYP_KEYPART inside POLYP_MULTI: set the child object to ClassHierarchyPresentation.NIL and the parent to ClassHierarchyPresentation.ALIVE_CORAL
                    - DEAD_CORAL inside any of the above: set the child object to ClassHierarchyPresentation.NIL and the parent to ClassHierarchyPresentation.DEAD_CORAL
                
                iterate through all objects without assigned present_class (None)
                    - POLYP_MULTI: ClassHierarchyPresentation.ALIVE_CORAL
                    - POLYP_SINGLE: ClassHierarchyPresentation.ALIVE_CORAL
                    - POLYP_KEYPART: ClassHierarchyPresentation.ALIVE_CORAL
                    - DEAD_CORAL: ClassHierarchyPresentation.DEAD_CORAL

        :param object_list_of_images: a 2d grid of object lists, each of which stores objects found from the corresponding image in the 2d grid of images
        :type object_list_of_images: dict
        :param images_grid_size: the dimension of the 2d grid of object list, which equals to the 2d grid of images
        :type images_grid_size: tuple
        :return: the total number of objects marked as invalidated by this function
        :rtype: int
        """
        parent_children_table = {}
        all_objects_list = []
        # iterate through each row and then each grid locations along a row
        for row_index in range(images_grid_size[1]):
            for col_index in range(images_grid_size[0]):
                # abort the process
                if self.to_cancel:
                    raise DetectorCancelled(DetectorExceptionCodes.CANCELLED_BY_SYSTEM, 'Received an cancel command from the system')
                # add the object list to the all object list
                all_objects_list.extend(self._filter_valid_objects(object_list_of_images[(col_index, row_index,)]))
                # consider each pairs of objects of neigbouring images
                if col_index < images_grid_size[0] - 1:
                    # resolve diplicate between (col_index, row_index) and (col_index + 1, row_index)
                    object_list_index_1, object_list_index_2 = (col_index, row_index), (col_index + 1, row_index)
                    CoralObjectDetectModel.extract_inside_Of_object_sets(self._filter_valid_objects(object_list_of_images[object_list_index_1]), 
                                                                                self._filter_valid_objects(object_list_of_images[object_list_index_2]), parent_children_table,
                                                                                self.coral_child_min_overlap_ratio)
                    
                    if row_index > 0:
                        # resolve diplicate between (col_index, row_index) and (col_index + 1, row_index - 1)
                        object_list_index_1, object_list_index_2 = (col_index, row_index), (col_index + 1, row_index - 1)
                        CoralObjectDetectModel.extract_inside_Of_object_sets(self._filter_valid_objects(object_list_of_images[object_list_index_1]), 
                                                                                    self._filter_valid_objects(object_list_of_images[object_list_index_2]), parent_children_table,
                                                                                    self.coral_child_min_overlap_ratio)
                if row_index >= images_grid_size[1] - 1:
                    continue
                # resolve diplicate between (col_index, row_index) and (col_index, row_index + 1)
                object_list_index_1, object_list_index_2 = (col_index, row_index), (col_index, row_index + 1)
                CoralObjectDetectModel.extract_inside_Of_object_sets(self._filter_valid_objects(object_list_of_images[object_list_index_1]), 
                                                                            self._filter_valid_objects(object_list_of_images[object_list_index_2]), parent_children_table,
                                                                            self.coral_child_min_overlap_ratio)
                
                if col_index < images_grid_size[0] - 1:
                    # resolve diplicate between (col_index, row_index) and (col_index + 1, row_index + 1)
                    object_list_index_1, object_list_index_2 = (col_index, row_index), (col_index + 1, row_index + 1)
                    CoralObjectDetectModel.extract_inside_Of_object_sets(self._filter_valid_objects(object_list_of_images[object_list_index_1]), 
                                                                                self._filter_valid_objects(object_list_of_images[object_list_index_2]), parent_children_table,
                                                                                self.coral_child_min_overlap_ratio)

        # implementation of the hierchical coral class framework
        # process the parent children table and annotate the parents as DEAD if all the children are dead
        parent_object:CoralObject
        child_object:CoralObject
        for parent_object in parent_children_table:
            children_set = parent_children_table[parent_object]
            if parent_object.present_class == None: # the parent object is not yet annotated, there is work to do
                # if the parent object is identified as a POLYP_MULTI, the types of children needs to be worked out
                if parent_object.coral_class == ClassHierarchyCoral.POLYP_MULTI.value:
                    counter = [0, 0, 0]     # alive, dead, other
                    for child_object in children_set:
                        if child_object.coral_class in (ClassHierarchyCoral.POLYP_KEYPART.value, ClassHierarchyCoral.POLYP_SINGLE.value):
                            counter[0] += 1
                        elif child_object.coral_class == ClassHierarchyCoral.DEAD_CORAL.value:
                            counter[1] += 1
                        elif child_object.coral_class == ClassHierarchyCoral.OTHER.value:
                            counter[2] += 1 
                        maxcount = max(counter)
                        if maxcount == counter[0]:
                            parent_object.present_class = ClassHierarchyPresentation.ALIVE_CORAL.value                        
                        elif maxcount == counter[1]:
                            parent_object.present_class = ClassHierarchyPresentation.DEAD_CORAL.value                   
                        else:
                            parent_object.present_class = ClassHierarchyPresentation.OTHER.value
                # parent is ALIVE_CORAL if its coral class is POLYP_KEYPART and not mask_polyp_keypart, which is a system config
                elif parent_object.coral_class == ClassHierarchyCoral.POLYP_KEYPART.value:
                    if not self.mask_polyp_keypart:
                        parent_object.present_class = None
                    else:
                        parent_object.present_class = ClassHierarchyPresentation.MASKED.value 
                # parent is ALIVE_CORAL if its coral class is POLYP_SINGLE
                elif parent_object.coral_class == ClassHierarchyCoral.POLYP_SINGLE.value:
                    parent_object.present_class = ClassHierarchyPresentation.ALIVE_CORAL.value 
                # parent is DEAD_CORAL if its coral class is DEAD_CORAL
                elif parent_object.coral_class == ClassHierarchyCoral.DEAD_CORAL.value:
                    parent_object.present_class = ClassHierarchyPresentation.DEAD_CORAL.value
                # parent is OTHER if its coral class is OTHER
                elif parent_object.coral_class == ClassHierarchyCoral.OTHER.value:
                    parent_object.present_class = ClassHierarchyPresentation.OTHER.value   
                # mask all the child objects
                for child_object in children_set:
                    child_object.present_class = ClassHierarchyPresentation.MASKED.value                   

        # traverse all valid objects again and set the parent class of not already done so
        obj:CoralObject
        for obj in all_objects_list:
            if obj.present_class is None:
                if obj.coral_class is not None:
                    # POLYP_MULTI and POLYP_SINGLE are always considered ALIVE_CORAL
                    if obj.coral_class in (ClassHierarchyCoral.POLYP_MULTI.value, ClassHierarchyCoral.POLYP_SINGLE.value): 
                        obj.present_class = ClassHierarchyPresentation.ALIVE_CORAL.value
                    # POLYP_KEYPART depends on mask_polyp_keypart, which is a system config, 
                    # to determine if it is considered ALIVE because sometimes the parent POLYP_MULTI is not detected
                    elif obj.coral_class == ClassHierarchyCoral.POLYP_KEYPART.value:
                        if not self.mask_polyp_keypart:
                            obj.present_class = ClassHierarchyPresentation.ALIVE_CORAL.value 
                        else:
                            obj.present_class = ClassHierarchyPresentation.MASKED.value 
                    # DEAD_CORAL is mapped to DEAD_CORAL
                    elif obj.coral_class == ClassHierarchyCoral.DEAD_CORAL.value:
                        obj.present_class = ClassHierarchyPresentation.DEAD_CORAL.value   
                    # OTHER is mapped to OTHER
                    elif obj.coral_class == ClassHierarchyCoral.OTHER.value:
                        obj.present_class = ClassHierarchyPresentation.OTHER.value    
                    # UNDEFINED is mapped to MASKED
                    elif obj.coral_class == ClassHierarchyCoral.UNDEFINED.value:
                        obj.present_class = ClassHierarchyPresentation.MASKED.value                                             
                else:
                    obj.present_class = ClassHierarchyPresentation.MASKED.value                
    
    @staticmethod
    def first_inside_second_bbox_min_ratio(bbox1:tuple, bbox2:tuple, min_ratio:float=1.0) -> bool:
        xA = max(bbox1[0], bbox2[0])
        yA = max(bbox1[1], bbox2[1])
        xB = min(bbox1[2], bbox2[2])
        yB = min(bbox1[3], bbox2[3])
        interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
        if interArea == 0:
            return False
        boxAArea = abs((bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]))
        ratio = interArea / float(boxAArea)
        return ratio >= min_ratio

    @staticmethod            
    def extract_inside_Of_object_sets(object_list_1:list, object_list_2:list, parent_children_table:dict, verbose=False, coral_child_min_overlap_ratio:float=0.25) -> None:
        """ a generic function for invalidating objects from two lists if they are found to co-locate in the tile space, subject to a maximum distance, 
        using the greedy algorithm. The two objects may be of different class because one of them may be partial that results in a different class

        :param object_list_1: a list of CoralObject objects
        :type object_list_1: list
        :param object_list_2: another list of CoralObject objects
        :type object_list_2: list
        :param overlap_sets_list: a list of sets of overlapped objects of the same class
        :type overlap_sets_list: list
        """
        object_1:CoralObject
        object_2:CoralObject
        child_object:CoralObject
        parent_object:CoralObject
        # combine the two lists into one
        combined_object_list = list(object_list_1)
        combined_object_list.extend(object_list_2)
        # compare the objects from two lists of objects
        for index_1, object_1 in enumerate(combined_object_list):
            for index_2, object_2 in enumerate(combined_object_list): 
                if index_1 == index_2:
                    continue 
                object_1.area = object_1.size[0] * object_1.size[1]
                object_2.area = object_2.size[0] * object_2.size[1]
                # if the two objects do not locate inside one another, ignore them
                if CoralObjectDetectModel.first_inside_second_bbox_min_ratio(object_1.bbox, object_2.bbox, coral_child_min_overlap_ratio) and object_2.area > object_1.area:
                    child_object, parent_object = object_1, object_2
                elif CoralObjectDetectModel.first_inside_second_bbox_min_ratio(object_2.bbox, object_1.bbox, coral_child_min_overlap_ratio) and object_1.area > object_2.area:
                    child_object, parent_object = object_2, object_1
                else:
                    continue
                # add the child object to the set of the parent
                if parent_object in parent_children_table:
                    parent_children_table[parent_object].add(child_object)
                else:
                    parent_children_table[parent_object] = set((child_object,))                
       
class CoralObjectDetectImageModel():
    """ CoralObjectDetectImageModel is a delegated class of CoralObjectDetectModel to handle one image. The class returns a list of objects detected within the image.
        The class divides the image into a 2D grid of blobs. There may be overlapped regions between neighouring blobs, specified by the parameter cod_blob_overlap_pix. 
        The class removes any duplicate resulting from the overlappng regions before resolving the final list of objects. 
    """
    def __init__(self, image:np.ndarray, image_col_index:int, image_row_index:int, yolo_detect_model_list:YoloObjectDetector, map_bbox_image_fn, map_normalize_bbox_tile_fn, **kwargs): 
        """ the constructor
        
        :param image: the source numpy image
        :type image: np.ndarray
        :param image_col_index:the column (x) index of the image, used for reference with the ImageReconstructModel and for data logging
        :type image_col_index: int
        :param image_row_index: the row index (y) of the image, used for reference with the ImageReconstructModel and for data 
        :type image_row_index: int
        :param yolo_detect_model_list: The list of applicable YoloObjectDetector models to be used, which should be suitable for the coral species found in the images
        :type yolo_detect_model_list: list of YoloObjectDetector        
        :param map_bbox_image_fn: A function that maps a bbox in (image_grid_x, image_grid_y, bbox) to the space reconustructed image
        :type map_bbox_image_fn: A function definition (image_grid_x, image_grid_y, bbox:list) -> bbox:list
        :param map_normalize_bbox_tile_fn: A function that maps a bbox in reconstructed image space to the tile space normalized to (0, 1)
        :type map_normalize_bbox_tile_fn: A function definition (bbox:list) -> bbox:list  
        """
        # model variable for abort
        self.to_cancel = False
        # input parameters
        self.image = image
        self.image_col_index, self.image_row_index = image_col_index, image_row_index

        self.map_bbox_image_fn = map_bbox_image_fn
        self.map_normalize_bbox_tile_fn = map_normalize_bbox_tile_fn
        self.params = kwargs
        # check type of yolo_detect_model_list
        if type(yolo_detect_model_list) == YoloObjectDetector:
            self.yolo_detect_model_list = [yolo_detect_model_list]
        else:
            self.yolo_detect_model_list = yolo_detect_model_list
        if len(self.yolo_detect_model_list) == 0:
            raise AssertionError(f'{type(self).__name__}: Parameter yolo_detect_model_list does not contain at least one YoloObjectDetector')
        # other keyword parameters - the blob size of all yolo_detect_model should be the same
        self.blob_size = None
        for yolo_detect_model in self.yolo_detect_model_list:
            if self.blob_size is None:
                self.blob_size = yolo_detect_model.get_blob_size()
            else:
                blob_size_of_model = yolo_detect_model.get_blob_size()
                if blob_size_of_model[0] != self.blob_size[0] or blob_size_of_model[1] != self.blob_size[1]:
                    raise DetectorAborted(DetectorExceptionCodes.YOLO_MODEL_UNDEFINED, f'Multiple yolo models: blob sizes are not consistent') 
        if self.blob_size is None:
            raise AssertionError(f'{type(self).__name__}: A YoloObjectDetector does not contain a valid blob size')
        
        # the object categories as defined by the yolo model
        # self.coral_classes = kwargs.get(ModelsConfigNames.OBJECT_CLASSES_CORAL.value, [])
        # self.dead_coral_classes = kwargs.get(ModelsConfigNames.OBJECT_CLASSES_DEAD_CORAL.value, [])
        # extract parameters for blob creation and duplicate removal
        self.blob_overlap_pix = kwargs.get(ModelsConfigNames.COD_BLOB_OVERLAP_PIX.value, 0)
        # extract other keyword parameters - output cache and debug information
        self.logdata_folder = kwargs.get(ModelsConfigNames.LOGDATA_FOLDER.value, None)
        # other keyword parameters - use cached detection object list instead of actual detection using the yolo_model
        self.use_cached_object_detection = kwargs.get(ModelsConfigNames.COD_USE_CACHED_OBJECT_DETECTION.value, False)
        self.debug_blob_images = kwargs.get(ModelsConfigNames.COD_DEBUG_BLOB_IMAGES.value, True)
        self.jpeg_quality = kwargs.get(ModelsConfigNames.OUTPUT_JPEG_QUALITY.value, 95)
        # multiple yolo models
        self.merge_mutli_yolo_models = kwargs.get(ModelsConfigNames.COD_MERGE_MULTI_MODELS.value, False)
        # extract init model variables
        self.object_class_names:set = set()                   # list of class names of the detection model
        self.metadata_of_blobs = dict()                # metadata of the blobs including detection 
        self.raw_object_list_of_blobs = dict()         # data structure for the duplicate removal process
        self.resolved_object_list = None                  # final object list after duplicate removal
        self.annotated_blob_filename_dict_list = []       # list of annotated blob filenames
        self.annotated_image_filename_dict = None              # the filename for the annotated image 
        
    def build(self):
        # attempt to load from cache if the flag is true
        if self.use_cached_object_detection and self.logdata_folder is not None:
            cached_file = os.path.join(self.logdata_folder, f'object_list_{self.image_col_index}_{self.image_row_index}.yaml')
            self._load_raw_object_list_of_blobs(cached_file)
        # step 1: load the image if not already loaded
        if type(self.image) == str:
            try:
                self.image = cv2.imread(self.image)
            except (Warning, Exception) as e:
                raise DetectorFailed(DetectorExceptionCodes.INPUT_DATA_INVALID,f'Unable to read image file {self.image}', e=e)
        if self.image is None or type(self.image) is not np.ndarray:
            raise DetectorFailed(DetectorExceptionCodes.INPUT_DATA_INVALID,f'Unable to read image file {self.image}: file not exists or not an image')

        # step 2: traverse through the image coordinates to build a list of logical image blobs 
        image_size = self.image.shape[:2][::-1]
        step_x, step_y = self.blob_size[0] - self.blob_overlap_pix, self.blob_size[1] - self.blob_overlap_pix
        self.image_blob_grid_size = math.ceil(image_size[0] / step_x), math.ceil(image_size[1] / step_y)
        self.blobs_count = 0
        object_count = 0
        self.to_update_cache = False
        # start_x and start_y are the top left corner of an image blob
        for start_x in range(0, image_size[0], step_x):
            for start_y in range(0, image_size[1], step_y):
                if self.to_cancel:
                    raise DetectorCancelled(DetectorExceptionCodes.CANCELLED_BY_SYSTEM, 'Received the abort command from the system')
                # compute the blob index, the top left and the bottom right corner of an image blob
                blob_col_index, blob_row_index = start_x // step_x, start_y // step_y 
                corner = (start_x, start_y,)
                end_x, end_y = start_x + self.blob_size[0], start_y + self.blob_size[1]
                end_x, end_y = min(end_x, image_size[0]), min(end_y, image_size[1]) 
                # extract the image blob from the numpy image
                image_blob = self.image[start_y:end_y, start_x:end_x].copy()
                image_blob_size = image_blob.shape[:2][::-1]
                # compute the cache index
                cache_index = (self.image_col_index, self.image_row_index, blob_col_index, blob_row_index,)
                # generate the raw object list or refer to the cache if loaded successfully earlier
                if cache_index not in self.raw_object_list_of_blobs:
                    try:
                        self.to_update_cache = True
                        # if the object list of the cache index does not exist (not using the cache file), invoke the yolo object detector and extract objects as a list
                        # detect objects in the image_blob using the yolo_model
                        # logger.info(f'OBJECT DETECTION in image ({self.image_col_index, self.image_row_index}) blob ({blob_col_index, blob_row_index}): {start_x, start_y} {end_x, end_y} {image_blob_size}') 
                        speed_as_dict = None
                        object_list_of_blob = None
                        yolo_detect_model:YoloObjectDetector
                        for yolo_detect_model in self.yolo_detect_model_list:

                            yolo_result:YoloResult = yolo_detect_model.detect(image_blob)
                            self.object_class_names = self.object_class_names.union(yolo_result.get_class_names())
                            # extract the processing speed information and update the metadata about processing this blob
                            speed_as_dict = yolo_result.get_processes_speed_as_dict(speed_as_dict) 
                            # get the classes_map
                            classes_map = yolo_detect_model.get_classes_map()
                            # extract the detected objects from the output of the yolo model of this blob
                            object_list = self._extract_objects_from_result(yolo_result, classes_map, self.image_col_index, self.image_row_index, corner, blob_col_index, blob_row_index, 
                                                                            self.map_bbox_image_fn, self.map_normalize_bbox_tile_fn)  
                            # mark geometrically invalid objects
                            keep_object_filter = yolo_detect_model.get_keep_object_filter()
                            to_apply_keep_object_filter = keep_object_filter.get('apply', False)
                            if to_apply_keep_object_filter:
                                aspect_ratio_max = keep_object_filter.get('aspect_ratio_max', 4.5)
                                area_min = keep_object_filter.get('area_min', 100)                      # pixels
                                obj: CoralObject
                                for obj in object_list:
                                    if obj.invalidated:
                                        continue
                                    aspect_ratio = max(obj.size[0] / obj.size[1], obj.size[1] / obj.size[0])
                                    object_area = obj.size[0] * obj.size[1]
                                    if aspect_ratio > aspect_ratio_max or object_area < area_min:
                                        obj.invalidated = True
                                
                            # combine the object lists from the yolo detect models
                            if object_list_of_blob is None:
                                object_list_of_blob = object_list
                            elif self.merge_mutli_yolo_models:
                                object_list_of_blob = self._merge_results_of_two_yolo_models(object_list_of_blob, object_list)
                            else:
                                # select the results of yolo models that contains more objects
                                object_list_of_blob = object_list if len(object_list) > len (object_list_of_blob) else object_list_of_blob

                        blob_metdata = {
                                'cod_blob_size': image_blob_size,
                                'cod_blob_bbox': [start_x, start_y, end_x, end_y] 
                        }
                        blob_metdata.update(speed_as_dict)
                        self.metadata_of_blobs[cache_index] = blob_metdata   # the data structure is for logdata
                        object_count += len(object_list_of_blob)
                        self.raw_object_list_of_blobs[cache_index] = object_list_of_blob
                    except Exception as e:
                        traceback.print_exc()
                        raise DetectorAborted(DetectorExceptionCodes.YOLO_MODEL_ERROR, f'Error happened when the YoloModel is applied on an image blob ({e})', e=e)
                    
                    # if the self.debug_blob_images is True, then generate the annotated image for this image blob and save to the logdata folder
                    if self.debug_blob_images and self.logdata_folder is not None:
                        annotated_image = yolo_result.draw_detection(image_blob, True)  # making a copy before annotation so that the original image is intact
                        image_file_name = f'annotated_blob_{self.image_col_index}_{self.image_row_index}_{blob_col_index}_{blob_row_index}.jpg'
                        image_dict = {'title': f'Annotated blob at image ({self.image_col_index} {self.image_row_index}) blob ({blob_col_index} {blob_row_index})', 'src': image_file_name}
                        self.annotated_blob_filename_dict_list.append(image_dict)
                        target_image_file = os.path.join(self.logdata_folder, image_file_name)
                        if not cv2.imwrite(target_image_file, annotated_image, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]):
                            raise DetectorExceptionCodes(DetectorExceptionCodes.OS_ERROR, f'Failed to save annotated image to {target_image_file}')
                else:
                    # if the object_list for the cache_index exists, just get it from the data structure
                    object_list_of_blob = self.raw_object_list_of_blobs[cache_index]
                    object_count += len(object_list_of_blob)
                    
                # print the information about the coral objects
                # for coral_object in object_list:
                #     logger.info(coral_object)
                self.blobs_count += 1

        logger.info(f'YOLO MODEL FOUND {object_count} objects in the image ({self.image_col_index, self.image_row_index}):')
        # step 3: iterate through each pair of neighbour blobs
        total_duplicates_removed = self._invalidate_duplicate_objects(self.raw_object_list_of_blobs, self.image_col_index, self.image_row_index, self.image_blob_grid_size)
        logger.info(f'DUPLICATE OBJECT REMOVAL between image blobs in the image ({self.image_col_index, self.image_row_index}): {total_duplicates_removed}') 

        # save the raw_object_list_of_blobs to cache file
        if self.to_update_cache or (not self.use_cached_object_detection and self.logdata_folder is not None):
            cache_data_file = os.path.join(self.logdata_folder, f'object_list_{self.image_col_index}_{self.image_row_index}.yaml')        # save the object list and metadata to the cache file
            logger.info(f'{type(self).__name__}: SAVE object list and metadata for {self.blobs_count} image blobs to {cache_data_file}')
            self._save_raw_object_list_of_blobs(cache_data_file)    
    
        # step 4: merge object lists of every blob into final object list
        self.resolved_object_list = self._merge_into_image_object_list()
        
        # step 5: generate and save the image annotated with the resolved list of objects if the self.debug_blob_images is True
        if self.debug_blob_images and self.logdata_folder is not None:
            # generate the annotated image for the whole image and save to the logdata folder
            annotated_image = CoralObjectListHelper.annotate_image_with_objects(self.resolved_object_list, self.image, print_name=True, include_invalidated=False)
            annotated_image = CoralObjectListHelper.annotate_image_with_blob_bbox(self.metadata_of_blobs, annotated_image)
            image_file_name = f'annotated_image_{self.image_col_index}_{self.image_row_index}.jpg'
            image_dict = {'title': f'Annotated image ({self.image_col_index} {self.image_row_index})', 'src': image_file_name}
            self.annotated_image_filename_dict = image_dict
            target_image_file = os.path.join(self.logdata_folder, image_file_name)
            if not cv2.imwrite(target_image_file, annotated_image, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]):
                raise DetectorExceptionCodes(DetectorExceptionCodes.OS_ERROR, f'Failed to save annotated image to {target_image_file}')
            
        # step 6: clear data if not needed for model inference
        self.image = None
        self.yolo_detect_model_list = None               
    
    def _merge_into_image_object_list(self) -> list:
        """ internal function to merge the object lists from the blobs into the overall list

        :return: a list contains the coral objects detected in the image, probably including invalidated objects
        :rtype: list
        """
        all_object_list = []
        for index in self.raw_object_list_of_blobs.keys():
            all_object_list.extend(self.raw_object_list_of_blobs[index])    
        return all_object_list    
    
    def _merge_results_of_two_yolo_models(self, object_list_1:list, object_list_2:list) -> list:
        overlap_sets_list = []
        CoralObjectListHelper.extract_overlap_objects_same_class(object_list_1, object_list_2, overlap_sets_list, min_overlap_ratio=0.5)
        # go through the overlap sets one at a time
        for overlap_set in overlap_sets_list:
            # find the one with the largest size and mark all other invalidate
            largest_object:CoralObject = max(overlap_set, key=lambda x: x.size[0] * x.size[1])
            for obj in overlap_set:
                if obj == largest_object:
                    obj.invalidated = False
                else:
                    obj.invalidated = True
        # combine the valid objects into a list
        combined_object_list = [obj for obj in object_list_1 if not obj.invalidated]
        combined_object_list.extend([obj for obj in object_list_2 if not obj.invalidated])
        return combined_object_list
    
    def cancel_build(self):
        """ call to abort the computing of this CoralObjectDetectImageModel
        """
        self.to_cancel = True

    def get_annotated_blob_filename_dict_list(self) -> list:
        """ Returns the list of file names of images showing annotation of blobs for debug purpose

        :return: the list of image file names
        """
        return self.annotated_blob_filename_dict_list
    
    def get_annotated_image_filename_dict(self) -> str:
        """ Returns the file name of images showing annotation for debug purpose

        :return: the file name
        """
        return self.annotated_image_filename_dict    

    def get_object_list(self, include_invalidated=False) -> list:
        """ returns the coral objects of the image as a list

        :param include_invalidated: the list includes the invalidated objects, defaults to False
        :type include_invalidated: bool, optional
        :return: the coral objects of the image as a list
        :rtype: list
        """
        if include_invalidated:
            return self.resolved_object_list
        validate_objects_list = []
        for coral_object in self.resolved_object_list:
            if not coral_object.invalidated:
                validate_objects_list.append(coral_object)
        return validate_objects_list
    
    def get_object_class_names(self) -> dict:
        """ returns a map of class_id, class_name for different classes of coral objects as specified by the object detection model

        :return: a map of class_id, class_name
        :rtype: dict
        """
        return self.object_class_names

    def _save_raw_object_list_of_blobs(self, cache_file:str):
        """ Save the data associated with the detected coral objects to a yaml file

        :param cache_file: path to the target yaml file
        :type cache_file: str
        """
        data = {
            'raw_object_list_of_blobs': self.raw_object_list_of_blobs,
            'metadata_of_blobs': self.metadata_of_blobs,
            'image_blob_grid_size': self.image_blob_grid_size,
            'blobs_count': self.blobs_count,
        }
        try:
            with open(cache_file, 'w') as outfile:
                yaml.dump(data, outfile, Dumper=yaml.Dumper)
        except (Warning, Exception) as e:
            logger.warning(f'{type(self).__name__}: Failed to write raw_object_list_of_blobs to {cache_file}')
            raise
                
    def _load_raw_object_list_of_blobs(self, cache_file:str):
        """ Load from a yaml file the data associated with the detected coral objects and restore them to the data structures of this object

        :param cache_file: path to the target yaml file
        :type cache_file: str
        """
        try:
            with open(cache_file, 'r') as infile:
                data = yaml.load(infile, Loader=yaml.Loader)
            self.raw_object_list_of_blobs = data['raw_object_list_of_blobs']
            self.metadata_of_blobs = data['metadata_of_blobs']
            return True
        except (Warning, Exception) as e:
            # logger.warning(f'{type(self).__name__}: Failed to load object list of blobs cache file {cache_file}\n{e}')
            ...
        return False

    def _extract_objects_from_result(self, yolo_result:YoloResult, classes_map:dict, image_col_index:int, image_row_index:int, corner:tuple, blob_col_index:int, blob_row_index:int, 
                                     map_bbox_image_fn, map_normalize_bbox_tile_fn) -> list:
        """ build a list of coral objects (CoralObject class) from the result of yolo model

        :param yolo_result: The result object received from prediction using a yolo model
        :type yolo_result: YoloResult
        :param image_col_index: The column (x) index of the image in the 2d grid of images
        :type image_col_index: int
        :param image_row_index: The row (y) index of the image in the 2d grid of images
        :type image_row_index: int
        :param corner: The topleft corner of the image blob location in the original image
        :type corner: tuple
        :param blob_col_index: The column (x) index of the blob in the 2d grid of blobs resulting from dividing an image 
        :type blob_col_index: int
        :param blob_row_index: The row (y) index of the blob in the 2d grid of blobs resulting from dividing an image 
        :type blob_row_index: int
        :param map_bbox_image_fn: A function that maps a bbox in (image_grid_x, image_grid_y, bbox) to the space reconustructed image
        :type map_bbox_image_fn: A function definition (image_grid_x, image_grid_y, bbox:list) -> bbox:list
        :param map_normalize_bbox_tile_fn: A function that maps a bbox in reconstructed image space to the tile space normalized to (0, 1)
        :type map_normalize_bbox_tile_fn: A function definition (bbox:list) -> bbox:list                  
        :return: A list of CoralObject objects
        :rtype: list
        """
        object_list = []
        yolo_result_list = yolo_result.get_all_objects()
        yolo_result:ObjectType
        for yolo_result in yolo_result_list:
            # extract findings from one result
            bbox_in_blob = yolo_result.bbox
            bbox_in_image = (bbox_in_blob[0] + corner[0], bbox_in_blob[1] + corner[1], bbox_in_blob[2] + corner[0], bbox_in_blob[3] + corner[1])
            bbox_in_reconstructed_image = map_bbox_image_fn(image_col_index, image_row_index, bbox_in_image)
            centre = (bbox_in_reconstructed_image[0] + yolo_result.size[0] // 2, bbox_in_reconstructed_image[1] + yolo_result.size[1] // 2,)
            bbox_in_tile = bbox_in_tile_normalized = None
            # convert locations in the recontructed image space into tile space, by considering the location of the frames
            # bbox_in_tile = locate_tile_model.map_bbox(bbox_in_reconstructed_image)
            # bbox_in_tile_normalized = locate_tile_model.normalize_bbox(bbox_in_tile)
            bbox_in_tile, bbox_in_tile_normalized = map_normalize_bbox_tile_fn(bbox_in_reconstructed_image)  # revised
            centre_normalized = ((bbox_in_tile_normalized[0] + bbox_in_tile_normalized[2]) / 2, (bbox_in_tile_normalized[1] + bbox_in_tile_normalized[3]) / 2,)
            size_normalized = bbox_in_tile_normalized[2] - bbox_in_tile_normalized[0], bbox_in_tile_normalized[3] - bbox_in_tile_normalized[1] 
            # compute the coral class fron the class_name output in yolo_result
            coral_class = None
            for coral_class_in_map in classes_map:
                if yolo_result.cls_name in classes_map[coral_class_in_map]:
                    coral_class = coral_class_in_map
                    break            
            # populate the contour with normalized points
            contour_points = []
            for point in yolo_result.points:
                point_in_image = (point[0] + corner[0], point[1] + corner[1], 0, 0)
                point_in_reconstructed_image = map_bbox_image_fn(image_col_index, image_row_index, point_in_image)
                point_in_tile, point_in_tile_normalized = map_normalize_bbox_tile_fn(point_in_reconstructed_image)  # revised    
                contour_points.append([point_in_tile_normalized[0], point_in_tile_normalized[1],])
            # find area defined by the contour
            contour_area = None
            if len(contour_points) > 0:
                shape_geometries = self._get_shape_geometries(contour_points)
                contour_area = shape_geometries['contour_area']
                
            confidence = yolo_result.conf
            # create the object from the extracted data
            coral_object = CoralObject(
                blob_row_index = blob_row_index,
                blob_col_index = blob_col_index,
                image_row_index = image_row_index,
                image_col_index = image_col_index,
                cls_id = yolo_result.cls_id,
                yolo_class = yolo_result.cls_name,      # class at the yolo layer
                coral_class = coral_class,              # class at the coral layer
                present_class = None,                   # class at the presentation layer
                bbox = bbox_in_reconstructed_image,
                centre = centre,
                size = yolo_result.size,
                bbox_in_blob = bbox_in_blob,
                bbox_in_image = bbox_in_image,
                bbox_in_tile = bbox_in_tile,
                bbox_normalized = bbox_in_tile_normalized,
                centre_normalized = centre_normalized,
                size_normalized = size_normalized,
                points_normalized = contour_points,
                contour_area_normalized = contour_area,
                confidence = confidence
            )
            object_list.append(coral_object)
        return object_list
    
    @staticmethod
    def _get_shape_geometries(points):
        ctr = np.array(points).reshape((-1, 1, 2)).astype(np.float32)
        contour_area = cv2.contourArea(ctr)
        # moments = cv2.moments(ctr)
        # bounding rectangles
        rect = cv2.minAreaRect(ctr)
        bbox = np.float32(cv2.boxPoints(rect))  # list of 2-lists
        # compute length long and short side
        length_1 = math.dist(bbox[0], bbox[1])
        length_2 = math.dist(bbox[1], bbox[2])
        if length_1 > length_2:
            long, short = length_1, length_2
        else:
            long, short = length_2, length_1
        return {
            'contour_area': contour_area,
            'long_side': long,
            'short_side': short,
        }           
    
    @staticmethod
    def _invalidate_duplicate_objects(raw_object_list_of_blobs:dict, image_col_index:int, image_row_index:int, image_blob_grid_size:tuple) -> int:
        """ a generic function for invalidating objects associated with every blobs in an image that are found to be duplicates.

        :param raw_object_list_of_blobs: a 2d grid of object lists, each of which stores objects found from the corresponding blob in the 2d grid of image blobs of an image
        :type raw_object_list_of_blobs: dict
        :param image_col_index:the column (x) index of the image, used for data logging
        :type image_col_index: int
        :param image_row_index: the row index (y) of the image, used for data logger
        :type image_row_index: int        
        :param image_blob_grid_size: the dimension of the 2d grid of object list, which equals to the 2d grid of image blobs
        :type image_blob_grid_size: tuple
        :param max_displacement: the threshold distance beyond which two objects can be considered as duplicates
        :type max_displacement: float
        :return: the total number of objects marked as invalidated by this function
        :rtype: int
        """
        total_duplicates_removed = 0
        overlap_sets_list = []
        # iterate through each row and then each grid locations along a row
        for blob_row_index in range(image_blob_grid_size[1]):
            for blob_col_index in range(image_blob_grid_size[0]):
                if blob_col_index < image_blob_grid_size[0] - 1:
                    # resolve diplicate between (blob_col_index, blob_row_index) and (blob_col_index + 1, blob_row_index)
                    object_list_index_1 = (image_col_index, image_row_index, blob_col_index, blob_row_index)
                    object_list_index_2 = (image_col_index, image_row_index, blob_col_index + 1, blob_row_index)
                    CoralObjectListHelper.extract_overlap_objects_same_class(raw_object_list_of_blobs[object_list_index_1], 
                                                                                                       raw_object_list_of_blobs[object_list_index_2], overlap_sets_list)
                    # logger.info(f'Number of duplicate removed between blobs {object_list_index_1} and {object_list_index_2} (max disp: {max_displacement}): {num_duplicates_removed}')
                if blob_row_index >= image_blob_grid_size[1] - 1:
                    continue
                # resolve diplications between (blob_col_index, blob_row_index) and (blob_col_index, blob_row_index + 1)
                object_list_index_1 = (image_col_index, image_row_index, blob_col_index, blob_row_index)
                object_list_index_2 = (image_col_index, image_row_index, blob_col_index, blob_row_index + 1)
                CoralObjectListHelper.extract_overlap_objects_same_class(raw_object_list_of_blobs[object_list_index_1], 
                                                                                                   raw_object_list_of_blobs[object_list_index_2], overlap_sets_list)
                # logger.info(f'Number of duplicate removed between blobs {object_list_index_1} and {object_list_index_2} (max disp: {max_displacement}): {num_duplicates_removed}')
                if blob_col_index < image_blob_grid_size[0] - 1:
                    # resolve diplicate between (blob_col_index, blob_row_index) and (blob_col_index + 1, blob_row_index + 1)
                    object_list_index_1 = (image_col_index, image_row_index, blob_col_index, blob_row_index)
                    object_list_index_2 = (image_col_index, image_row_index, blob_col_index + 1, blob_row_index + 1)
                    CoralObjectListHelper.extract_overlap_objects_same_class(raw_object_list_of_blobs[object_list_index_1], 
                                                                                                       raw_object_list_of_blobs[object_list_index_2], overlap_sets_list)
                    # logger.info(f'Number of duplicate removed between blobs {object_list_index_1} and {object_list_index_2} (max disp: {max_displacement}): {num_duplicates_removed}')
        # go through the overlap sets one at a time
        for overlap_set in overlap_sets_list:
            # find the one with the largest size and mark all other invalidate
            largest_object:CoralObject = max(overlap_set, key=lambda x: x.size[0] * x.size[1])
            for obj in overlap_set:
                if obj == largest_object:
                    obj.invalidated = False
                else:
                    obj.invalidated = True
                    total_duplicates_removed += 1 
                    
        # logger.info(f'Total number of duplicates removed from overlapped regions between blobs: {total_duplicates_removed}')  
        return total_duplicates_removed     
    

class CoralObjectListHelper():
    """ CoralObjectListHelper provides generic functions for processing lists of coral objects

    """

    @staticmethod            
    def extract_overlap_objects_same_class(object_list_1:list, object_list_2:list, overlap_sets_list:list, min_overlap_ratio:float=0.05, verbose=False) -> int:
        """ a generic function for invalidating objects from two lists if they are found to co-locate in the tile space, subject to a maximum distance, 
        using the greedy algorithm. The two objects may be of different class because one of them may be partial that results in a different class

        :param object_list_1: a list of CoralObject objects
        :type object_list_1: list
        :param object_list_2: another list of CoralObject objects
        :type object_list_2: list
        :param overlap_sets_list: a list of sets of overlapped objects of the same class
        :type overlap_sets_list: list
        :return: the number of object invalidated in this function
        :rtype: int
        """
        object_1:CoralObject
        object_2:CoralObject
        # compare the objects from two lists of objects
        for index_1, object_1 in enumerate(object_list_1):
            for index_2, object_2 in enumerate(object_list_2): 
                # if the two objects do not overlap, ignore them
                if not CompareTools.overlap_bbox(object_1.bbox, object_2.bbox):
                    continue
                # if CompareTools.jaccard_bbox(object_1.bbox, object_2.bbox) < min_overlap_ratio:
                #     continue
                # if the coral class of any of the two objects are undefined, skip this pair
                if object_1.coral_class == ClassHierarchyCoral.UNDEFINED.value or object_2.coral_class == ClassHierarchyCoral.UNDEFINED.value:
                    continue
                # if the two objects are of different classes, ignore them
                if object_1.coral_class != object_2.coral_class:
                    continue                
                # search for objects in the overlap sets list
                object_set_list = [s for s in overlap_sets_list if object_1 in s or object_2 in s]
                if not object_set_list:
                    overlap_sets_list.append(set((object_1, object_2,)))
                else:
                    merged_set = set((object_1, object_2,))
                    for s in object_set_list:
                        overlap_sets_list.remove(s)
                        merged_set = merged_set.union(s)
                    # add the merged set to the overlap sets list
                    overlap_sets_list.append(merged_set)

    @staticmethod 
    def annotate_image_with_objects(object_list:list, output_image:np.ndarray, print_name=True, include_invalidated=False) -> np.ndarray:
        """ draw objects from a list at their locations on the given numpy image

        :param object_list: a list that contains CoralObject objects to be drawn
        :type object_list: list
        :param output_image: The numpy image as the canvas for drawing
        :type output_image: np.ndarray
        :param print_name: to include classname in the drawing, defaults to True
        :type print_name: bool, optional
        :param include_invalidated: include the invalidated objects, defaults to False
        :type include_invalidated: bool, optional
        :return: the numpy image annotated with locations of the objects
        :rtype: np.ndarray
        """
        palette = YoloResult._get_palette()
        coral_object:CoralObject
        for coral_object in object_list:
            if (not include_invalidated) and coral_object.invalidated:
                continue
            color = palette[int(coral_object.cls_id)]
            cv2.rectangle(output_image, (int(coral_object.bbox_in_image[0]), int(coral_object.bbox_in_image[1])), (int(coral_object.bbox_in_image[2]), int(coral_object.bbox_in_image[3])), color, 3)          
            if print_name:
                text_to_draw = f'{coral_object.yolo_class}/{coral_object.coral_class} ({coral_object.confidence:.1f})'
                cv2.putText(output_image, text_to_draw,
                        (int(coral_object.bbox_in_image[0]) + random.randint(0, 20), int(coral_object.bbox_in_image[1]) - 10 + random.randint(0, 20)),
                        cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 0), 1)
        return output_image        

    @staticmethod 
    def annotate_image_with_blob_bbox(blob_meta_dict:dict, output_image:np.ndarray) -> np.ndarray:    
        for blob_metadata_key in blob_meta_dict.keys():
            blob_metadata = blob_meta_dict[blob_metadata_key]
            start_x, start_y, end_x, end_y = blob_metadata['cod_blob_bbox']
            cv2.rectangle(output_image, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (0, 0, 255), 1)
            cv2.putText(output_image, f'{blob_metadata_key}',
                        (int(start_x + 15), int(start_y + 15)), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 1)           
        return output_image
        
    @staticmethod
    def get_index_permutations(sequence:list) -> list:
        """ generate a list of all permutations from the given sequence of objects

        :param sequence: a sequence of objects for the generation of permutations
        :type sequence: list
        :return: a list all permutations from the given sequence of objects
        :rtype: list
        """
        # returns if the sequence is empty and there are no permutations
        if len(sequence) == 0:
            return []
        elif len(sequence) == 1:
            return [sequence]
        output_list = [] # empty list that will store current permutation
        # iterate the input sequence and calculate the permutation
        for i in range(len(sequence)):
            m = sequence[i]
            remLst = sequence[:i] + sequence[i+1:]
            for p in CoralObjectListHelper.get_index_permutations(remLst):
                output_list.append([m] + p)
        return output_list
    
    @staticmethod
    def index_permutations(sequence:list):
        """ a generator that yield permutations from the given sequence of objects

        :param sequence: a sequence of objects for the generation of permutations
        :type sequence: list
        :return: one permutation
        :rtype: Generator[Any, Any, Any]
        """
        if len(sequence) <= 1:
            yield sequence
            return
        for perm in CoralObjectListHelper.index_permutations(sequence[1:]):
            for i in range(len(sequence)):
                yield perm[:i] + sequence[0:1] + perm[i:]

class CoralObjectDetectModelHelper():
    """ CoralObjectDetectModelHelper provides helper functions for caching CoralObjectDetectModel to the file system and retrieve the object

    """
    @classmethod            
    def to_yaml_file(cls, cod_model:CoralObjectDetectModel, object_file:str) -> None:
        """ Save an object of CoralObjectDetectModelHelper to a yaml file

        :param cod_model: The CoralObjectDetectModel object
        :type cod_model: CoralObjectDetectModel
        :param object_file: The target file path of the yaml file, defaults to None, which returns the yaml as a string
        :type object_file: str, optional
        """
        logger.info(f'{type(cls).__name__}: Save CoralObjectDetectModel to {object_file}')
        cod_model._save_object_list_of_images(object_file)
        
    @staticmethod
    def from_yaml_file(object_file:str) -> CoralObjectDetectModel:
        """ Load an object of CoralObjectDetectModelHelper from a yaml file

        :param object_file: The source file path of the yaml file
        :type object_file: str
        :return: An object of CoralObjectDetectModel loaded from the yaml file 
        :rtype: CoralObjectDetectModel
        """
        return CoralObjectDetectModel.from_yaml_file(object_file)
