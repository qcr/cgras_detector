#!/usr/bin/env python3
# 
# # Copyright 2024 - Andrew Kwok Fai LUI, 
# Robotics and Autonomous Systems Group, REF, RI
# and the Queensland University of Technology

__author__ = 'Andrew Lui'
__copyright__ = 'Copyright 2024'
__license__ = 'GPL'
__version__ = '1.0'
__email__ = 'ak.lui@qut.edu.au'
__status__ = 'Development'

import yaml, os
from enum import Enum
from collections.abc import MutableMapping
# ros params
import rospy

class ModelsConfigNames(Enum):
    """ Maps the config names as a sting to a constant
    """
    TASK_PARAMS_FILENAME = 'task_params_filename'
    RECO_MODEL_FILENAME = 'reco_model_filename'
    LOCTILE_MODEL_FILENAME = 'loctile_model_filename'
    RECO_DEGUG_IMAGE_ORIGINAL_SCALE = 'reco_debug_images_at_original_scale'
    RECO_MANUAL_VALIDATION_ORIGINAL_SCALE = 'reco_manual_validation_original_scale'
    RECO_TAB_GRID_IMAGE = 'reco_tab_grid_image'
    RECO_DEBUG_FEATURE_MATCH_IMAGES = 'reco_debug_feature_matching_images'
    OUTPUT_JPEG_QUALITY = 'output_jpeg_quality'
    RECO_FEATURE_DETECTORS = 'reco_feature_detectors'
    RECO_FEATURE_MATCHING_CONFIDENCE_THRESHOLD = 'reco_feature_matching_confidence_threshold'
    RECO_IMAGE_MATCHING_MIN_CONFIDENCE = 'reco_image_matching_min_confidence'
    RECO_IMAGE2D_MATCHING_MIN_CONFIDENCE = 'reco_image2d_matching_min_confidence'
    RECO_WORKING_SCALE = 'reco_working_scale'
    RECO_ERROR_CORRECTION = 'reco_error_correction'
    RECO_ASPECT_RATIO_ROI_ERROR_REL = 'reco_aspect_ratio_roi_error_rel'
    RECO_MISPLACED_ROI_ERROR_REL = 'reco_misplaced_roi_erro_rel'
    
    TILE_SIZE_IN_MM = 'tile_size_in_mm'     # the size of the tile in mm (width, height)
    FRAME_SIZE_IN_MM = 'frame_size_in_mm'   # the size of the frame in mm (width, height)
    
    
    LOCTILE_WORKING_SCALE = 'loctile_working_scale'
    LOCTILE_PIXEL_CLASSIFIER_FILENAME = 'loctile_pixel_classifier_filename'
    LOCTILE_TEMPLATE_SIZE = 'loctile_template_size'
    LOCTILE_TEMPLATE_CORNER_SIZE = 'loctile_template_corner_size'
    LOCTILE_MATCHING_SCORE_MIN = 'loctile_matching_score_min'
    LOCTILE_DEBUG_IMAGES = 'loctile_debug_images'
    LOCTILE_ROTATE_ANGLE_MAX = 'loctile_rotate_angle_max'
    LOCTILE_ASPECT_RATIO_DIFF_MAX_REL = 'loctile_aspect_ratio_diff_max_rel'
    LOCTILE_ASPECT_RATIO_DIFF_MAX_ABS = 'loctile_aspect_ratio_diff_max_abs'
    
    COD_MODEL_FILENAME = 'cod_model_filename'
    COD_DEBUG_BLOB_IMAGES = 'cod_debug_blob_images'
    COD_BLOB_OVERLAP_PIX = 'cod_blob_overlap_pix'
    COD_USE_CACHED_OBJECT_DETECTION = 'cod_use_cached_object_detection'
    COD_CORAL_CHILD_MIN_OVERLAP_RATIO = 'cod_coral_child_min_overlap_ratio'
    COD_MERGE_MULTI_MODELS = 'cod_merge_mutli_models'
    COD_MASK_POLYP_KEYPART = 'cod_mask_polyp_keypart'
    
    # the following are parameters generated dynamically during task execution
    LOGDATA_FOLDER = 'logdata_folder'
    COD_BLOB_SIZE = 'cod_blob_size'
    OBJECT_CLASSES_CORAL = 'coral_classes'
    OBJECT_CLASSES_DEAD_CORAL = 'dead_coral_classes'
    OBJECT_CLASSES_MAP = 'classes_map'
    YOLO_MODEL_FILE = 'yolo_model_file'
    
