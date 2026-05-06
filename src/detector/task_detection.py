# Copyright 2024 - Andrew Kwok Fai LUI, 
# Robotics and Autonomous Systems Group, REF, RI
# and the Queensland University of Technology

__author__ = 'Andrew Lui'
__copyright__ = 'Copyright 2024'
__license__ = 'GPL'
__version__ = '1.0'
__email__ = 'ak.lui@qut.edu.au'
__status__ = 'Development'

import os, math, yaml, contextlib, glob, time, shutil, traceback
from enum import Enum
from collections import defaultdict, OrderedDict
import datetime
from time import strftime, localtime
import numpy as np
import pandas as pd
import cv2

from cgras_datatools.lock_tools import synchronized
from detector.models import logger, ModelsConfigNames, DetectorFailed, DetectorAborted, DetectorCancelled, DetectorExceptionCodes
from detector.models.detect import CoralObjectDetectModel, CoralObjectDetectModelHelper, YoloObjectDetector, CoralObject, ClassHierarchyPresentation

from detector.models.reconstruct import ImageReconstructModel, ImageReconstructModelHelper

from detector.models.locate_tile import LocateTileModel, LocateTileModelHelper
from detector.html.lightbox import LightboxHelper
from detector.model import DETECT_DAO, APP_FILE_MANAGER, CONFIG, SystemConfigNames     


class ProgressStages(Enum):
    UNKNOWN = -1
    INIT = 0
    RECO = 1
    LOCTILE = 2
    OBJECT_DETECT = 3
    COLLECT_STAT = 4
    COMPLETED = 5

class ProgressModel():
    def __init__(self):
        self.progress = OrderedDict()
        self.start_time = OrderedDict()
        self.end_time = OrderedDict()
        for stage in ProgressStages:
            self.progress[stage] = [0, 0]  # (currnet step, num_steps)
            self.start_time[stage] = None
            self.end_time[stage]  = None
        self.current_stage = ProgressStages.UNKNOWN

    @synchronized
    def start_stage(self, stage:ProgressStages):
        self.current_stage = stage
        self.progress[stage][1] = 1
        self.start_time[stage] = time.time()

    @synchronized
    def update_stage_progress(self, stage:ProgressStages, current_step:int, num_steps:int):
        self.progress[stage][0] = current_step
        self.progress[stage][1] = num_steps

    @synchronized
    def end_stage(self, stage:ProgressStages):
        self.current_stage = stage        
        self.progress[stage][0] = self.progress[stage][1]
        self.end_time[stage] = time.time()

    @synchronized
    def get_total_time(self):
        total_time = 0
        for stage in ProgressStages:
            if self.start_time[stage] is not None and self.end_time[stage] is not None:
                total_time += self.end_time[stage] - self.start_time[stage]
        return total_time
    
    def get_current_stage(self):
        return self.current_stage
    
    def get_progress_at_stage(self, stage):
        return self.progress[stage]


class DetectionTaskModel():
    WHOLE_RECO_HTML_FILENAME = 'whole_reco_image.html'
    ANNOTATED_WHOLE_RECO_HTML_FILENAME = 'annotated_whole_reco_image.html'
    ROTATED_ANNOTATED_ORIGINAL_SCALE_HTML_FILENAME = 'rotated_annotated_original_image.html'
    FEATURE_MATCH_HTML_FILENAME = 'feature_match_images.html'
    ANNOTATED_BLOBS_HTML_FILENAME = 'annotated_blobs_at_{}_{}.html'
    ANNOTATED_BLOBS_INDEX_HTML_FILENAME = 'annotated_blobs_index.html'
    ANNOTATED_IMAGES_HTML_FILENAME = 'annotated_images.html'
    LANDING_HTML_FILENAME = 'index.html'
    
    def __init__(self, tile_sample_id:str, params:dict=None):
        # progress tracking
        self.progress_model = ProgressModel()
        self.progress_model.start_stage(ProgressStages.INIT)
        # model variables
        self.image_map_as_list:list = None
        self.image_grid_dim:tuple = None
        self.reco_model:ImageReconstructModel = None
        self.loctile_model:LocateTileModel = None
        self.cod_model:CoralObjectDetectModel = None
        self.detection_stat = {} 
        self.to_cancel = False
        self.start_time = time.time()
        # extract information about this tile sample
        self.tile_sample_id = tile_sample_id
        self.tile_sample_dict = DETECT_DAO.get_tile_sample(self.tile_sample_id)
        if self.tile_sample_dict is None:
            logger.warning(f'{type(self).__name__}: The tile_sample_id ({self.tile_sample_id}) not found')
            raise AssertionError(f'Invalid parameter (tile_sample_id)')   
             
        self.tile_id, self.batch_id, self.batch_time = self.tile_sample_dict['tile_id'], self.tile_sample_dict['batch_id'], self.tile_sample_dict['batch_time']
        self.species, self.settle_time, self.season = self.tile_sample_dict['species'], self.tile_sample_dict['settle_time'], self.tile_sample_dict['season']
        
        self.medadata = self.tile_sample_dict.get('metadata', None)
        if self.medadata is not None:
            self.tile_size, self.frame_size = self.medadata.get('tile_size', None), self.medadata.get('frame_size', None)
        else:
            self.tile_size = self.frame_size = None
        # tile_size and frame_size are essential parameters, should not use any default value
        if self.tile_size is None or self.frame_size is None:
            raise DetectorFailed(DetectorExceptionCodes.INPUT_DATA_INVALID, f'Tile sample does not contain tile size or frame size', e = AssertionError('Tile sample error'))
            
        # evaluate the number of days since settlement
        try:
            self.settle_date_dt, self.capture_date_dt = pd.to_datetime(self.settle_time, utc=True), pd.to_datetime(self.batch_time, utc=True)
            self.days_since_settle = (self.capture_date_dt - self.settle_date_dt).days
        except Exception as e:
            raise DetectorFailed(DetectorExceptionCodes.INPUT_DATA_INVALID, f'Invalid date/time format in batch time or settle time', e = e)
        
        # resolve the active yolo models for this tile (may be empty — stitching-only mode)
        self.yolo_model_dict_list = DETECT_DAO.query_yolo_model(self.species, self.days_since_settle)
        if not self.yolo_model_dict_list:
            logger.warning(f'{type(self).__name__}: No active yolo model for species ({self.species}) days_since_settle ({self.days_since_settle}) — running in stitching/tile-location-only mode')
            self.yolo_model_dict_list = [] 
        # just pick the first one if more than one yolo model is suitable
        # self.yolo_model_dict = self.yolo_model_list[0]

        # build parameters for the detection process
        if params is not None:
            self.params = params
        else:
            self.params = CONFIG.to_params([SystemConfigNames, ModelsConfigNames])
        # load the params with other config variables
        self.working_scale = self.params.get(ModelsConfigNames.RECO_WORKING_SCALE.value, 0.1) 
        self.logdata_folder = self.params[ModelsConfigNames.LOGDATA_FOLDER.value] = APP_FILE_MANAGER.get_detector_subfolder(APP_FILE_MANAGER.DATA_SUBFOLDER, self.season, self.tile_sample_id)
        # self.params[ModelsConfigNames.YOLO_MODEL_FILE.value] = self.yolo_model_dict['model_file_path']
        # self.params[ModelsConfigNames.COD_BLOB_SIZE.value] = (self.yolo_model_dict['input_image_width'], self.yolo_model_dict['input_image_height'], )
        # self.params[ModelsConfigNames.OBJECT_CLASSES_MAP.value] = self.yolo_model_dict['classes_map']
        self.debug_images_at_original_scale = self.params.get(ModelsConfigNames.RECO_DEGUG_IMAGE_ORIGINAL_SCALE.value, False)
        self.params[ModelsConfigNames.TILE_SIZE_IN_MM.value] = self.tile_size
        self.params[ModelsConfigNames.FRAME_SIZE_IN_MM.value] = self.frame_size        
        # add other tile info to the params for metadata yaml file output
        self.params['tile_sample_id'] = self.tile_sample_id
        self.params['tile_id'] = self.tile_id
        self.params['batch_id'] = self.batch_id
        self.params['batch_time'] = self.batch_time
        self.params['season'] = self.season
        self.params['settle_date'] = self.settle_time
        self.params['species'] = self.species
        self.params['coral_age_in_days'] = self.days_since_settle
        
        # write the params to the log folder
        task_params_metadata_filename = self.params.get(ModelsConfigNames.TASK_PARAMS_FILENAME.value, '_params.yaml')
        try:
            param_yaml_file = os.path.join(self.logdata_folder, task_params_metadata_filename)
            with open(param_yaml_file, 'w') as outfile:
                yaml.dump(self.params, outfile, Dumper=yaml.SafeDumper)
        except Exception as e:
            logger.warning(f'{type(self).__name__}: unable to save detect model parameter to the logdata folder {param_yaml_file}')
            raise DetectorAborted(DetectorExceptionCodes.OS_ERROR, f'Failed to save detect model parameter to the logdata folder {param_yaml_file}', e=e)
        # copy the script files for the generated html files to the logdata folder of a tile sample
        scripts_folder = os.path.join(self.logdata_folder, 'scripts')
        APP_FILE_MANAGER.copy_scripts_folder(scripts_folder)
        # signal the end of the INIT stage
        self.progress_model.end_stage(ProgressStages.INIT)

    def get_tile_sample_id(self) -> str:
        return self.tile_sample_id
    
    def get_start_time(self) -> float:
        return self.start_time
    
    def get_time_lapsed(self) -> float:
        return time.time() - self.start_time    
    
    def get_start_time_iso8601(self) -> str:
        # return str(datetime.datetime.fromtimestamp(self.start_time))
        return strftime('%Y-%m-%d %H:%M:%S', localtime(self.start_time))
    
    def get_params(self) -> dict:
        return self.params

    @staticmethod
    def _build_image_map_as_list(tile_sample_id:str) -> list:
        captured_image_list = DETECT_DAO.query_source_images_of_tile_sample(tile_sample_id)
        capture_image_grid = defaultdict(lambda: None)
        # collect the image file path into a 2d grid first, before building a 2d list
        grid_size_x, grid_size_y = 0, 0
        for captured_image in captured_image_list:
            capture_image_grid[(captured_image['capture_x'], captured_image['capture_y'])] = captured_image['file_path']
            grid_size_x = max(grid_size_x, captured_image['capture_x'])
            grid_size_y = max(grid_size_y, captured_image['capture_y'])
        # build the 2d list
        image_map_as_list = []
        for y in range(grid_size_y + 1):
            row_images = []
            for x in range(grid_size_x + 1):
                if (x, y) not in capture_image_grid:
                    logger.warning(f'DetectionTaskModel._build_image_map_as_list: One or more images are missing in the 2d grid of images')
                    raise DetectorFailed(DetectorExceptionCodes.INPUT_DATA_INVALID, f'Missing image in the capture grid')
                row_images.append(capture_image_grid[(x, y)])
            image_map_as_list.append(row_images)
        return image_map_as_list, (grid_size_x, grid_size_y,)
    
    def execute_task_reco(self):
        self.progress_model.start_stage(ProgressStages.RECO)
        # build image_map_as_list from the captured images
        try:
            self.image_map_as_list, self.image_grid_dim = self._build_image_map_as_list(self.tile_sample_id)
        except Exception as e:
            raise DetectorFailed(DetectorExceptionCodes.INPUT_DATA_INVALID, f'Image file in the capture grid not found', e = e)
        # load the cached ImageReconstructModel if exists, or build a new model from captured images
        reco_model_file = os.path.join(self.logdata_folder, self.params.get(ModelsConfigNames.RECO_MODEL_FILENAME.value, 'reco_model.yaml'))
        try:
            logger.info(f'{type(self).__name__}: Attempting to load cached ImageReconstructModel')
            self.reco_model:ImageReconstructModel = ImageReconstructModelHelper.from_yaml_file(reco_model_file)
        except Exception as e:
            logger.info(f'{type(self).__name__}: No valid cached file. Building the ImageReconstructModel from capture images')
        
        if self.reco_model is None:
            self.reco_model = ImageReconstructModel(self.image_map_as_list, **self.params) 
            if self.to_cancel:
                self.progress_model.end_stage(ProgressStages.RECO)  
                raise DetectorCancelled(DetectorExceptionCodes.CANCELLED_BY_SYSTEM, 'Received an cancel command from the system')
            self.reco_model.build()
            ImageReconstructModelHelper.to_yaml(self.reco_model, reco_model_file)
        self.progress_model.end_stage(ProgressStages.RECO)
        
    def execute_task_loctile(self):
        self.progress_model.start_stage(ProgressStages.LOCTILE)
        # load the cached LocateTileModel if exists, or build a new model from captured images and the ImageReconstructModel
        loctile_model_file = os.path.join(self.logdata_folder, self.params.get(ModelsConfigNames.LOCTILE_MODEL_FILENAME.value, 'loctile_model.yaml'))
        try:
            logger.info(f'{type(self).__name__}: Attempting to load cached LocateTileModelHelper')
            self.loctile_model:LocateTileModel = LocateTileModelHelper.from_yaml_file(loctile_model_file)
        except:
            logger.info(f'{type(self).__name__}: No valid cached file. Building the loctile_model_file from capture images')
            
        if self.loctile_model is None:
            self.loctile_model = LocateTileModel(self.image_map_as_list, map_location_fn=self.reco_model.map_locations, image_size_in_px=self.reco_model.get_whole_reco_image_size(), **self.params)
            if self.to_cancel:
                self.progress_model.end_stage(ProgressStages.LOCTILE)
                raise DetectorCancelled(DetectorExceptionCodes.CANCELLED_BY_SYSTEM, 'Received an cancel command from the system')
            self.loctile_model.build()
            LocateTileModelHelper.to_yaml(self.loctile_model, loctile_model_file)
        if self.params.get(ModelsConfigNames.RECO_TAB_GRID_IMAGE.value, True):
            self._draw_tab_grid_images()
        self.progress_model.end_stage(ProgressStages.LOCTILE)

    def _draw_tab_grid_images(self):
        if not self.logdata_folder:
            return
        n_cols = self.tile_sample_dict.get('tab_ncols', 20)
        n_rows = self.tile_sample_dict.get('tab_nrows', 20)
        if n_cols is None or n_cols <= 0:
            n_cols = 20
        if n_rows is None or n_rows <= 0:
            n_rows = 20
        tile_origin = self.loctile_model.get_tile_origin_in_image_space()
        tile_size = self.loctile_model.get_tile_size_in_image_space()
        # working-scale tab grid image
        rotated_image_file = os.path.join(self.logdata_folder, LocateTileModel.ROTATED_WHOLE_RECO_IMAGE_FILENAME)
        if os.path.isfile(rotated_image_file):
            image = cv2.imread(rotated_image_file)
            if image is not None:
                output_file = os.path.join(self.logdata_folder, 'rotated_whole_reco_image_tab_grid.jpg')
                CoralObjectDetectModel.draw_tab_grid_on_image(image, self.working_scale, tile_origin, tile_size, n_cols, n_rows, output_file)
        # full-scale tab grid image — scale line width and font proportionally to 1/working_scale
        full_scale_reco_file = os.path.join(self.logdata_folder, LocateTileModel.FILENAME_WHOLE_RECO_FULL_SCALE_IMAGE)
        if os.path.isfile(full_scale_reco_file):
            image = cv2.imread(full_scale_reco_file)
            if image is not None:
                M = self.loctile_model.affine_transform_matrix
                rotated = cv2.warpAffine(image, M, (int(image.shape[1] * 1.1), int(image.shape[0] * 1.1)))
                output_file = os.path.join(self.logdata_folder, 'rotated_whole_reco_original_image_tab_grid.jpg')
                scale_factor = max(1.0, 1.0 / self.working_scale)
                CoralObjectDetectModel.draw_tab_grid_on_image(rotated, 1.0, tile_origin, tile_size, n_cols, n_rows, output_file,
                                                              line_width=max(1, round(scale_factor)),
                                                              font_size=0.6 * scale_factor)

    def execute_task_object_detection(self):
        self.progress_model.start_stage(ProgressStages.OBJECT_DETECT)
        self.progress_model.update_stage_progress(ProgressStages.OBJECT_DETECT, 0, self.image_grid_dim[0] * self.image_grid_dim[1])
        # load the cached CoralObjectDetectionModel if exists, or build a new model from captured images, the ImageReconstructModel, and the yolo model
        cod_model_file = os.path.join(self.logdata_folder, self.params.get(ModelsConfigNames.COD_MODEL_FILENAME.value, 'coral_object_detect_model.yaml'))
        try:
            logger.info(f'{type(self).__name__}: Attempting to load cached CoralObjectDetectModel')
            self.cod_model = CoralObjectDetectModelHelper.from_yaml_file(cod_model_file)
        except:
            logger.info(f'{type(self).__name__}: No valid cached file. Building the CoralObjectDetectModel from capture images, reco model, loctile model, and yolo model')
            
        if self.cod_model is None:
            if not self.yolo_model_dict_list:
                logger.info(f'{type(self).__name__}: No active yolo models — skipping object detection')
                self.progress_model.end_stage(ProgressStages.OBJECT_DETECT)
                return
            yolo_detector_model_list = []
            if self.to_cancel:
                self.progress_model.end_stage(ProgressStages.OBJECT_DETECT)
                raise DetectorCancelled(DetectorExceptionCodes.CANCELLED_BY_SYSTEM, 'Received an cancel command from the system')
            for yolo_model_dict in self.yolo_model_dict_list:
                yolo_model_file =  yolo_model_dict['model_file_path']
                blob_size = (yolo_model_dict['input_image_width'], yolo_model_dict['input_image_height'], )
                classes_map = yolo_model_dict['classes_map']
                predict_params = yolo_model_dict['predict_params']
                keep_object_filter = yolo_model_dict['keep_object_filter']
                try:
                    logger.info(f'{type(self).__name__}: Attempting to load the yolo_model_file at {yolo_model_file}')
                    yolo_detector_model:YoloObjectDetector = YoloObjectDetector(yolo_model_file, blob_size, classes_map, predict_params, keep_object_filter)
                    yolo_detector_model_list.append(yolo_detector_model)
                except Exception as e:
                    logger.info(f'{type(self).__name__}: Failed to load the yolo model file: {e}')
                    raise DetectorAborted(DetectorExceptionCodes.YOLO_MODEL_FILE_ERROR, f'Failed to load the yolo model file ({yolo_model_file})', e = e)
            # raise exception if the current job is cancelled
            if self.to_cancel:
                self.progress_model.end_stage(ProgressStages.OBJECT_DETECT)
                raise DetectorCancelled(DetectorExceptionCodes.CANCELLED_BY_SYSTEM, 'Received an cancel command from the system')
            # build the cod model
            logger.info(f'DetectionTaskModel build COD for images of the tile sample ({self.tile_sample_id}) using yolo model file: {yolo_model_file}')
            logger.info(f'object class map: {classes_map}')
            logger.info(f'yolo predict params: {predict_params}')
            self.cod_model = CoralObjectDetectModel(self.image_map_as_list, yolo_detector_model_list, self.reco_model.map_bbox, self.loctile_model.map_and_normalize_bbox, self.loctile_model.get_tile_size_in_image_space(),
                                                    self._execute_task_object_detection_cb, **self.params)
            self.cod_model.build()
            try:
                CoralObjectDetectModelHelper.to_yaml_file(self.cod_model, cod_model_file)
            except Exception as e:
                raise DetectorAborted(DetectorExceptionCodes.OS_ERROR, f'Failed to write cod model to yaml file {cod_model_file}', e=e)
            
        # annotate an image of the reconstructed space with the objects for validation
        if self.logdata_folder:
            rotated_reco_image_file = os.path.join(self.logdata_folder, LocateTileModel.ROTATED_WHOLE_RECO_IMAGE_FILENAME)
            if os.path.isfile(rotated_reco_image_file):
                rotated_image = cv2.imread(rotated_reco_image_file)
                if rotated_image is not None:
                    image_file_name = self.cod_model.ANNOTATED_WHOLE_RECO_IMAGE_FILENAME
                    target_image_file = os.path.join(self.logdata_folder, image_file_name)
                    self.cod_model.annotate_whole_reco_image_with_objects(rotated_image, self.working_scale, self.loctile_model.get_tile_origin_in_image_space(), 
                                                                        self.loctile_model.get_tile_size_in_image_space(), target_image_file)
        # annotate full scale if the flag debug_images_at_original_scale is True
        if self.logdata_folder and self.debug_images_at_original_scale:
            rotated_reco_image_file = os.path.join(self.logdata_folder, LocateTileModel.ROTATED_WHOLE_RECO_FULL_SCALE_IMAGE_FILENAME)
            if os.path.isfile(rotated_reco_image_file):
                rotated_image = cv2.imread(rotated_reco_image_file)
                if rotated_image is not None:
                    image_file_name = self.cod_model.ANNOTATED_WHOLE_RECO_ORIGINAL_SCALE_IMAGE_FILENAME
                    target_image_file = os.path.join(self.logdata_folder, image_file_name)
                    self.cod_model.annotate_whole_reco_image_with_objects(rotated_image, 1.0, self.loctile_model.get_tile_origin_in_image_space(), 
                                                                        self.loctile_model.get_tile_size_in_image_space(), target_image_file,
                                                                        line_width=4, font_size=5.0, draw_coral_class=True)
        # the end of the object detection task
        self.progress_model.end_stage(ProgressStages.OBJECT_DETECT)        

    def _execute_task_object_detection_cb(self, progress_tuple:tuple):
        if progress_tuple is not None:
            self.progress_model.update_stage_progress(ProgressStages.OBJECT_DETECT, *progress_tuple)   
         
    def execute_task_record(self):
        self.progress_model.start_stage(ProgressStages.COLLECT_STAT)
        # extract statistics of the tile
        self.detection_stat['tile_pixel_x'], self.detection_stat['tile_pixel_y'] = self.loctile_model.get_tile_size_in_image_space()
        if self.to_cancel:
            self.progress_model.end_stage(ProgressStages.COLLECT_STAT)
            raise DetectorCancelled(DetectorExceptionCodes.CANCELLED_BY_SYSTEM, 'Received an cancel command from the system')
        try:
            if self.cod_model is not None:
                # detection ran — save objects and real counts
                num_objects = self.cod_model.get_num_objects()
                num_invalid_objects = self.cod_model.get_num_invalidated_objects()
                logger.info(f'{type(self).__name__}: Saving objects to database (valid/all): {num_objects - num_invalid_objects} {num_objects}')
                DETECT_DAO.delete_detected_objects_of_tile_sample(self.tile_sample_id)
                stat = DETECT_DAO.add_detected_object_from_coral_object_list(self.tile_sample_id, self.cod_model.get_object_list(), exclude_outside_of_tile=True)
                stat.update({
                    'duplicates_removed': self.cod_model.get_num_invalidated_objects(),
                    'total_object_count': self.cod_model.get_num_objects(),
                })
            else:
                # stitching-only mode — record zero counts
                logger.info(f'{type(self).__name__}: No detection model was active; recording zero counts')
                DETECT_DAO.delete_detected_objects_of_tile_sample(self.tile_sample_id)
                stat = {'coral_alive_count': 0, 'coral_dead_count': 0, 'other_count': 0,
                        'duplicates_removed': 0, 'total_object_count': 0}
            logger.info(f'{type(self).__name__}: Statistics {stat}')
            self.detection_stat.update(stat)
            self._update_detection_stat(self.detection_stat)
        except Exception:
            traceback.print_exc()
            raise DetectorAborted(DetectorExceptionCodes.DB_ERROR, 'Failed to write detection results to the database')
        # generate html files
        try:
            self.generate_html_files()
        except Exception as e:
            raise DetectorAborted(DetectorExceptionCodes.OS_ERROR, f'Unable to write html files: {e}', e=e)
        self.progress_model.end_stage(ProgressStages.COLLECT_STAT)
        self.progress_model.end_stage(ProgressStages.COMPLETED)

    def cancel_task(self):
        self.to_cancel = True
        if self.reco_model is not None:
            logger.warning(f'Task Detection: attempt to cancel building of reco_model')
            self.reco_model.cancel_build()        
        if self.cod_model is not None:
            logger.warning(f'Task Detection: attempt to cancel building of cod_model')
            self.cod_model.cancel_build()
    
    def generate_html_files(self):
        link_dict_list = []
        try:
            # generate the html file for viewing the whole reconstructed image
            whole_reco_image_filename = self.reco_model.FILENAME_WHOLE_RECO_IMAGE
            html_output_file = os.path.join(self.logdata_folder, DetectionTaskModel.WHOLE_RECO_HTML_FILENAME)
            LightboxHelper.generate_single_image_lightbox(html_output_file, whole_reco_image_filename)
            link_dict_list.append({'href': whole_reco_image_filename, 'text': 'Reconstructed image of the tile sample'})
        except Exception as e:
            logger.warning(f'Failed to generate HTML file for showing the whole reconstructed image')
            traceback.print_exc()
            raise
        
        if self.cod_model is not None:
            try:
                # generate the html file for viewing the annotated whole reconstructed image
                annotated_whole_reco_filename = self.cod_model.ANNOTATED_WHOLE_RECO_IMAGE_FILENAME
                html_output_file = os.path.join(self.logdata_folder, DetectionTaskModel.ANNOTATED_WHOLE_RECO_HTML_FILENAME)
                LightboxHelper.generate_single_image_lightbox(html_output_file, annotated_whole_reco_filename)
                link_dict_list.append({'href': annotated_whole_reco_filename, 'text': 'Annotated whole tile sample'})
            except Exception as e:
                logger.warning(f'Failed to generate HTML file for showing the annotated whole tile sample')
                traceback.print_exc()
                raise

            try:
                # generate the html file for viewing the annotated whole reconstructed image of original scale
                annotated_original_scale_filename = self.cod_model.ANNOTATED_WHOLE_RECO_ORIGINAL_SCALE_IMAGE_FILENAME
                html_output_file = os.path.join(self.logdata_folder, DetectionTaskModel.ROTATED_ANNOTATED_ORIGINAL_SCALE_HTML_FILENAME)
                LightboxHelper.generate_single_image_lightbox(html_output_file, annotated_original_scale_filename)
                link_dict_list.append({'href': annotated_original_scale_filename, 'text': 'Annotated whole tile sample original scale'})
            except Exception as e:
                logger.warning(f'Failed to generate HTML file for showing the annotated whole tile sample at original scale')
                traceback.print_exc()
                raise

        try:
            # generate the html file for viewing the feature matching images
            feature_matching_image_dict_list = self.reco_model.get_feature_match_image_dict_list()
            html_output_file = os.path.join(self.logdata_folder, DetectionTaskModel.FEATURE_MATCH_HTML_FILENAME)
            title = 'Feature Matching Output in the Reconstruction of Tile Image'
            LightboxHelper.generate_multi_images_lightbox(html_output_file, feature_matching_image_dict_list, title)
            link_dict_list.append({'href': DetectionTaskModel.FEATURE_MATCH_HTML_FILENAME, 'text': 'Feature matching for image reconstruction'})
        except:
            logger.warning(f'Failed to generate HTML file for showing feature matching between images')
            traceback.print_exc()
            raise

        if self.cod_model is not None:
            try:
                # generate the html file for viewing the annotated blobs
                annotated_blob_dict_list_as_dict = self.cod_model.get_annotated_blob_filename_dict_lists_as_dict()
                for image_index in annotated_blob_dict_list_as_dict.keys():
                    col_index, row_index = image_index
                    html_output_file = os.path.join(self.logdata_folder, DetectionTaskModel.ANNOTATED_BLOBS_HTML_FILENAME.format(col_index, row_index))
                    title = f'Image Blobs Annotated with Detected Objects in Image {image_index}'
                    LightboxHelper.generate_multi_images_lightbox(html_output_file, annotated_blob_dict_list_as_dict[image_index], title)
                image_index_list = sorted(annotated_blob_dict_list_as_dict.keys())
                annotated_blob_index_links_list = []
                for image_index in image_index_list:
                    col_index, row_index = image_index
                    annotated_blob_index_links_list.append({'href': DetectionTaskModel.ANNOTATED_BLOBS_HTML_FILENAME.format(col_index, row_index), 'text': f'Image Blobs Annotated with Detected Objects in Image {image_index}'})
                html_output_file = os.path.join(self.logdata_folder, DetectionTaskModel.ANNOTATED_BLOBS_INDEX_HTML_FILENAME)
                title = f'Image Blobs Annotated with Detected Objects in Tile Sample {self.tile_sample_id}'
                LightboxHelper.generate_landing_page(html_output_file, annotated_blob_index_links_list, title)
                link_dict_list.append({'href': DetectionTaskModel.ANNOTATED_BLOBS_INDEX_HTML_FILENAME, 'text': 'Image Blobs Annotated with Detected Objects'})
                # generate the html file for viewing the annotated images
                annotated_image_dict_list = self.cod_model.get_annotated_image_filename_dict_list()
                html_output_file = os.path.join(self.logdata_folder, DetectionTaskModel.ANNOTATED_IMAGES_HTML_FILENAME)
                title = 'Capture Images Annotated with Detected Objects'
                LightboxHelper.generate_multi_images_lightbox(html_output_file, annotated_image_dict_list, title)
                link_dict_list.append({'href': DetectionTaskModel.ANNOTATED_IMAGES_HTML_FILENAME, 'text': 'Capture Images Annotated with Detected Objects'})
            except:
                logger.warning(f'Failed to generate HTML file for showing annotated images')
                traceback.print_exc()
                raise
        # generate the landing page
        try:
            html_output_file = os.path.join(self.logdata_folder, DetectionTaskModel.LANDING_HTML_FILENAME)
            title = f'Imagery records for the image reconstruction modelling for the tile sample {self.tile_sample_id}'
            LightboxHelper.generate_landing_page(html_output_file, link_dict_list, title)
        except:
            logger.warning(f'Failed to generate the landing page HTML file')  
            traceback.print_exc()  
            raise

    def get_time_since_start(self):
        return time.time() - self.start_time

    def get_progress(self) -> ProgressModel:
        return self.progress_model

    def _update_detection_stat(self, detection_stat:dict):
        # convert the dict into yaml format
        # coral_alive_count, coral_dead_count, other_count, duplicates_removed
        yaml_data = yaml.dump(detection_stat, Dumper=yaml.SafeDumper)
        DETECT_DAO.update_tile_sample_detect_stat(self.tile_sample_id, detection_stat['tile_pixel_x'], detection_stat['tile_pixel_y'], 
                            detection_stat['coral_alive_count'], detection_stat['coral_dead_count'], detection_stat['other_count'], 
                            detection_stat['duplicates_removed'], yaml_data)

    # NOTE: not used in execute_task_record     
    # def _process_detected_objects(self, cod_model:CoralObjectDetectModel) -> dict:
    #     stat = {
    #         'coral_alive_count': 0,
    #         'coral_dead_count': 0,
    #         'other_count': 0,
    #         'duplicates_removed': cod_model.get_num_invalidated_objects(),
    #         'total_object_count': cod_model.get_num_objects(),
    #     }
    #     coral_object_list:list = cod_model.get_object_list()
    #     coral_object:CoralObject
    #     for coral_object in coral_object_list:
    #         if coral_object.invalidated:
    #             continue
    #         centre_x, centre_y = coral_object.centre_normalized[0], coral_object.centre_normalized[1]
    #         corner_x1, corner_y1 = coral_object.bbox_normalized[0], coral_object.bbox_normalized[1]
    #         size_x, size_y = coral_object.bbox_normalized[2] - corner_x1, coral_object.bbox_normalized[3] - corner_y1
    #         if coral_object.present_class == ClassHierarchyPresentation.ALIVE_CORAL.value:
    #             stat['coral_alive_count'] += 1
    #         elif coral_object.present_class == ClassHierarchyPresentation.DEAD_CORAL.value:
    #             stat['coral_dead_count'] += 1
    #         elif coral_object.present_class == ClassHierarchyPresentation.OTHER.value:
    #             stat['other_count'] += 1
    #         DETECT_DAO.add_detected_object_from_coral_object(self.tile_sample_id, coral_object)
    #     return stat

    @staticmethod
    def delete_cache_files(tile_sample_id:str, delete_reco=False, delete_object_list=False, delete_object_detection_model=False):
        with contextlib.suppress(FileNotFoundError, Exception):
            logdata_folder = DetectionTaskModel.get_cache_folder(tile_sample_id)        
            if delete_reco:
                os.remove(os.path.join(logdata_folder, CONFIG.get(ModelsConfigNames.RECO_MODEL_FILENAME.value, 'reco_model.yaml')))
                os.remove(os.path.join(logdata_folder, CONFIG.get(ModelsConfigNames.LOCTILE_MODEL_FILENAME.value, 'loctile_model.yaml')))
                
            if delete_object_list:
                delete_object_detection_model = True
                for file in glob.glob(os.path.join(logdata_folder, 'object_list_*.yaml')):
                    logger.warning(f'delete_cache_files: {file}')
                    os.remove(file)
            
            if delete_object_detection_model:            
                os.remove(os.path.join(logdata_folder, CONFIG.get(ModelsConfigNames.COD_MODEL_FILENAME.value, 'coral_object_detect_model.yaml')))
                
    @staticmethod
    def delete_cache_folder(tile_sample_id:str):
        with contextlib.suppress(FileNotFoundError, Exception):
            logdata_folder = DetectionTaskModel.get_cache_folder(tile_sample_id)
            shutil.rmtree(logdata_folder, ignore_errors=True)        

    @staticmethod
    def get_cache_folder(tile_sample_id:str):
        try:
            tile_sample_dict = DETECT_DAO.get_tile_sample(tile_sample_id)
            logdata_folder = APP_FILE_MANAGER.get_detector_subfolder(APP_FILE_MANAGER.DATA_SUBFOLDER, tile_sample_dict['season'], tile_sample_id)
            return logdata_folder
        except:
            return None
    
    @staticmethod
    def get_partial_cache_folder(tile_sample_id:str):
        try:
            tile_sample_dict = DETECT_DAO.get_tile_sample(tile_sample_id)
            partial_logdata_folder = os.path.join(tile_sample_dict['season'], tile_sample_id)
            return partial_logdata_folder    
        except:
            return None

# ---------------------------------------
# test functions

def get_basic_detection_params() -> dict:
    params = {
        'reco_model_filename': 'reco_model.yaml',
        'loctile_model_filename': 'loctile_model.yaml',
        'reco_debug_images_at_original_scale': False,
        'reco_debug_feature_matching_images': True,
        'reco_feature_detector': 'sift',
        'reco_feature_matching_confidence_threshold': 1.0,
        'reco_image_matching_min_confidence': 1.5,
        'reco_working_scale': 0.1,
        'cod_model_filename': 'coral_object_detect_model.yaml', 
        'cod_debug_blob_images': True,
        'cod_blob_overlap_pix': 128,
        'cod_use_cached_object_detection': True,
        'cod_duplicate_max_displacement_images': 64,
        'cod_duplicate_max_displacement_blobs': 32, 
        # to be updated in the DetectionTaskModel using the database
        'logdata_folder': None, 
        'yolo_model_file': None,  
        'cod_blob_size': None,     
        'coral_classes': None,                                                             
    }  
    return params


if __name__ == '__main__':
    # tile_sample_id = '2023Dec-P00003-CG1-202311201200'
    # tile_sample_id = '2023Dec-P10001-CG1-202402161404'
    tile_sample_id = '2023Dec-P20000-CG1-202311231200'
    dt_model = DetectionTaskModel(tile_sample_id, get_basic_detection_params())
    dt_model.execute_task_reco()
    dt_model.execute_task_loctile()
    dt_model.execute_task_object_detection()
    dt_model.execute_task_record()    
    
    # DetectionTaskModel.delete_cache_files(tile_sample_id, True, True)
