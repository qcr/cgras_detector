# Copyright 2024 - Andrew Kwok Fai LUI, 
# Robotics and Autonomous Systems Group, REF, RI
# and the Queensland University of Technology

__author__ = 'Andrew Lui'
__copyright__ = 'Copyright 2024'
__license__ = 'GPL'
__version__ = '1.0'
__email__ = 'ak.lui@qut.edu.au'
__status__ = 'Development'

import os, yaml, numbers, logging, traceback
from enum import Enum
from collections import defaultdict
from datetime import datetime
import cv2
import numpy as np
from statistics import median
# from the stitching module
from stitching.images import Images
from stitching.feature_detector import FeatureDetector
from stitching.feature_matcher import FeatureMatcher
from stitching.subsetter import Subsetter
from stitching.camera_estimator import CameraEstimator
from stitching.camera_adjuster import CameraAdjuster
from stitching.camera_wave_corrector import WaveCorrector
from stitching.warper import Warper
from stitching.seam_finder import SeamFinder
from stitching.blender import Blender

from detector.models.imaging_tools import ImageMap, CameraTransformTools, test_get_cgras_sample_images_as_list
from detector.models import ModelsConfigNames, get_logger, DetectorFailed, DetectorExceptionCodes, DetectorAborted, DetectorCancelled, RecoErrorCorrection

"""
About the coral detection and visualization system.
The reconstruction component is currently designed so that if no reliable parameters for image reconstruction are found (such as inadequate matching features), the tile sample is marked rejected. This design is based on the principle that accuracy is absolutely important.  it is better to report an error and give up. Clearly, sometimes if only 1 out of 24 images is problematic, then the whole tile sample would be given up.
I suggest that we change the design principle to 'extract as much information as possible, even if it means some locations of objects will not be accurate'.
Some changes will be made to the reconstruction components, including attempt to use different feature extraction schemes (brisk/sift) instead of just one.  Use the homography matrix of the nearby row
"""

class ImageReconstructModel():
    """ ImageReconstructModel is the wrapper class of the other image reconstruction classes in this module, that provides functions for mapping locations from the space of an individual input image to the 
        space of the reconstructed image. 
    """
    FILENAME_WHOLE_RECO_IMAGE = 'whole_reco_image.jpg'
    FILENAME_WHOLE_RECO_FULL_SCALE_IMAGE = 'whole_reco_image_original_scale.jpg'
    def __init__(self, images_2d_list:list, **kwargs):
        """ The constructor, which accepts a list of lists of images in row-major manner. The top-level list contains lists 
        of images in one row. The images are part of a larger image, and the order of the images in the data structure is consistent
        of that the images are foudnd in the larger image

        :param images_2d_list: A list of lists of images, each of which can be image paths (str typed) or image pixels (np.ndarray), arranged in a 2D grid
        :type images_2d_list: list
        """
        # control parameter
        self.to_cancel = False
        # model parameter
        self.reco_2d_model = None
        # obtain the logger
        self.logger = kwargs.get('logger', get_logger())
        # ignore the constructor if the object is loaded from yaml file and simply return
        if images_2d_list is None:
            return
        self.images_2d_list = images_2d_list
        self.params = kwargs
        # input parameters
        self.working_scale = kwargs.get(ModelsConfigNames.RECO_WORKING_SCALE.value, 0.1)   # the scale to reduce image size for which the transforms are calculated to speed up execution 
        if self.working_scale < 0.01 or self.working_scale > 1:
            self.logger.error(f'System configuration parameter working scale is beyond valid range between 0.01 and 1.00: {self.working_scale}')
            raise AssertionError(f'System configuration parameter working scale is beyond valid range between 0.01 and 1.00: {self.working_scale}')
        self.scaling_factor = 1 / self.working_scale                                 # the scaling factor to restore locations at the original scale
        self.params['scaling_factor'] = self.scaling_factor
        # load the error correction model
        self.apply_error_correction = kwargs.get(ModelsConfigNames.RECO_ERROR_CORRECTION.value, False)
        # self.logger.warning(f'Apply Error Correction: {self.apply_error_correction}')
        if self.apply_error_correction:
            self.error_correction_model = RecoErrorCorrection()

    def build(self):
        # build the model
        self.logger.info(f'ImageReconstructModel working_scale: {self.working_scale} (scaling: {self.scaling_factor})')
        self.reco_2d_model = ImageReconstruct2DModel(self.images_2d_list, **self.params)  
        if self.to_cancel:  # stop processing if abort signal is recieved
            raise DetectorCancelled(DetectorExceptionCodes.CANCELLED_BY_SYSTEM, 'Received an cancel command from the system')     
        
        self.reco_2d_model.build()  
        # retrieve major parameters from the model
        self.ncols, self.nrows = self.reco_2d_model.get_image_map_size()               # the number of rows and columns in the 2d grid of images
        # self.original_image_size = self.reco_2d_model.get_original_image_size()        # the size of the original images (assuming all images have the same size)
        self.working_image_size = self.reco_2d_model.get_working_image_size()          # the size of the working image
        # retrieve the confidence matrix used to evaluate the feature matching between images in 1D rows and between rows
        self.confidence_matrix_list = self.reco_2d_model.get_confidence_matrix_list()                # a list containing confidence matrix of stitching images in a row
        self.confidence_matrix_between_rows = self.reco_2d_model.get_confidence_matrix_between_rows()  # the confidence matrix of stitching reconstructed images between rows
        # retrieve the camera transform from the model
        self.camera_transforms_row_list = self.reco_2d_model.get_camera_transforms_row_list()              # a list that contains lists of camera transforms for each image along a row
        self.camera_transforms_between_rows = self.reco_2d_model.get_camera_transforms_between_rows()      # a list that contains camera tranforms of the row reconstructed images
        self.whole_reco_image_size = self.reco_2d_model.get_whole_reco_image_size()                        # the size of the whole reconstructed image at working scale
        # validate the camera_transforms_between_rows
        if len(self.camera_transforms_row_list) < self.nrows:
            raise DetectorFailed(DetectorExceptionCodes.RECO_MATCH_FAILED, f'Cannot obtain camera transform for every rows: lack sufficient features or feature detector not suitable')

        if len(self.camera_transforms_between_rows) < self.nrows:
            raise DetectorFailed(DetectorExceptionCodes.RECO_MATCH_FAILED, f'Cannot obtain camera transforms between rows: lack sufficient between row features or feature detector not suitable')
        # extract the image size of the row reconstruction images from the between rows camera transform
        self.row_reco_image_sizes = []
        for row_index in range(self.nrows):
            camera_transform = self.camera_transforms_between_rows[row_index]
            image_size = (camera_transform.ppy * 2, camera_transform.ppx * 2)    # note: the transform was computed based on clockwise rotated image and x,y are swapped
            self.row_reco_image_sizes.append(image_size)
        # extract the origin offsets of the row reco images and the whole reco image
        self.row_reco_image_origin_offsets_list = self.reco_2d_model.get_row_reco_image_origin_offsets_list()
        self.whole_reco_image_origin_offset = self.reco_2d_model.get_whole_reco_image_origin_offset()
        # retrieve the sample feature matching image file dict 
        self.feature_match_image_dict_list = self.reco_2d_model.get_feature_match_image_dict_list()
        # discard the model to save memory
        self.reco_2d_model = None
        # initialize frequently used objects when this object is queried
        self._prepare_objects()
    
    def _prepare_objects(self):
        # create warpers for location mapping
        self.warper = []
        for row_index in range(self.nrows):
            self.warper.append(cv2.PyRotationWarper('spherical', self.get_row_median_focal(row_index)))
        self.warper_between_rows = cv2.PyRotationWarper('spherical', self.get_between_rows_median_focal())  

    def cancel_build(self):
        """ abort the current reconstruction process """
        self.to_cancel = True
        self.logger.warning(f'ImageReconstructModel: received ABORT signal')
        if hasattr(self, 'reco_2d_model') and self.reco_2d_model is not None:
            self.reco_2d_model.cancel_build()

    def map_bbox(self, col_index:int, row_index:int, bbox, use_working_scale=False):
        """ returns the mapped coordinates in the reconstructed image of the coordinates of a bounding box of one of the original images 

        :param col_index: The column index of the original image
        :type col_index: int
        :param row_index: The row index of the original image
        :type row_index: int
        :param bbox: The bounding box (x1, y1, x2, y2) specified as a 4-tuple or 4-list
        :type bbox: A 4-list or 4-tuple      
        """   
        points = [(bbox[0], bbox[1],), (bbox[2], bbox[3],)]
        results = self.map_locations(col_index, row_index, points, use_working_scale)
        return [*results[0], *results[1]]
    
    def map_locations(self, col_index:int, row_index:int, points, use_working_scale=False):
        """ returns the coordinate in the reconstructed image of one or more points in one of the original images

        :param col_index: The column index of the original image
        :type col_index: int
        :param row_index: The row index of the original image
        :type row_index: int
        :param points: The parameter can be a point (a tuple of two floats indicating the (x, y) position) or a list of points
        :type points: A list of tuples or a tuple
        """
        single_point = False
        if type(points) in (tuple, list) and len(points) == 2 and isinstance(points[0], numbers.Number):
            single_point = True
            points = [points]
        # iterate through each point in an original image and compute their location in the whole reconstructed image
        results = []
        camera_transform:cv2.detail.CameraParams = self.camera_transforms_row_list[row_index][col_index]
        row_reco_image_origin_offset = self.row_reco_image_origin_offsets_list[row_index]
        K = camera_transform.K().astype(np.float32)
        for point in points:
            # the point is scaled to that was used to compute the camera transform
            if not use_working_scale:
                self.logger.debug(f'map location {point} of image {col_index, row_index} at original scale (multiply location by {self.scaling_factor})')
                sx, sy = point[0] * self.working_scale, point[1] * self.working_scale
            else:
                self.logger.debug(f'map location {point} at working scale {self.working_scale}')
            # compute step-by-step for the mapping
            # 1. map the point to (ax2, ay2) which is the location in the row constructed image for row_index using the row reco image wrapper and the output (ax, ay) offseted to find (ax2, ay2).                
            # 2. map (ax2, ay2) to (bx, by) which is the location in the counter-clockwise rotated row reconstructed image
            # 3. map (bx, by) to (cx2, cy2) which is the location in the whole constructed image using the whole image warper and the output (cx, cy) offseted to find (cx2, cy2).
            # 4. map (cx2, cy2) to (dx, dy) which is the location in the clockwise rotated whole reconstructed image  
            if self.ncols == 1:
                ax, ay = sx, sy
            else:
                ax, ay = self.warper[row_index].warpPoint((sx, sy,), K, camera_transform.R)
            self.logger.debug(f'ax, ay = {ax, ay}')
            ax2, ay2 = ax - row_reco_image_origin_offset[0], ay - row_reco_image_origin_offset[1]
            self.logger.debug(f'offseted ax, ay = {ax2, ay2} based on {row_reco_image_origin_offset}')
            reco_image_size = self.row_reco_image_sizes[row_index]
            bx, by = ay2, reco_image_size[0] - ax2
            self.logger.debug(f'rotated ax, ay = {bx, by}')
            camera_transform_between_row = self.camera_transforms_between_rows[row_index]
            if self.nrows == 1:
                cx, cy = bx, by
            else:
                K2 = camera_transform_between_row.K().astype(np.float32)
                cx, cy = self.warper_between_rows.warpPoint((bx, by,), K2, camera_transform_between_row.R)
            self.logger.debug(f'cx, cy = {cx, cy}')
            cx2, cy2 = cx - self.whole_reco_image_origin_offset[0], cy - self.whole_reco_image_origin_offset[1]
            self.logger.debug(f'offseted cx, cy = {cx2, cy2} based on {self.whole_reco_image_origin_offset}')                                
            dx, dy = self.whole_reco_image_size[0] - cy2, cx2
            self.logger.debug(f'rotated cx, cy = {dx, dy}')  
            # 5. adjust the reference frame of the location based on the origin_point if available 
            # if hasattr(self, 'tile_origin') and self.tile_origin is not None:
            #     dx, dy = (dx - self.origin_point[0], dy - self.origin_point[1],)
            # 6. the result is rescaled to the original scale of the input
            if not use_working_scale:
                dx, dy = (dx * self.scaling_factor, dy * self.scaling_factor,)
            mapped_point = (dx, dy)
            if self.apply_error_correction:
                error = self.error_correction_model.predict_error_point(mapped_point)
                mapped_point = (dx + error[0], dy + error[1])
                self.logger.warning(f'Location {dx, dy} Error {error} Corrected {mapped_point}')
            if single_point:
                return mapped_point
            results.append(mapped_point)
        return results
    
    def get_row_camera_transform(self, col_index:int, row_index:int, print=True) -> cv2.detail.CameraParams:
        """ Returns the camera transform parameters (CameraParams) of the given image at col_index and row_index

        :param col_index: The column index of the image
        :type col_index: int
        :param row_index: The row index of the image
        :type row_index: int
        :param print: Display the parameters to the screen, defaults to True
        :type print: bool, optional
        :return: The camera transform parameters
        :rtype: cv2.detail.CameraParams
        """
        camera_transform = self.camera_transforms_row_list[row_index][col_index]
        if print:
            self.logger.info(f'Camera matrices for image index {col_index, row_index}:')
            self.print_camera_transform(camera_transform)
        return camera_transform
    
    def get_between_row_camera_transform(self, row_index:int, print=True) -> cv2.detail.CameraParams:
        """ Returns the camera transform parameters (CameraParams) of the row reconstructed image at row_index

        :param row_index: The row index of the reconstructed image
        :type row_index: int
        :param print: Display the parameters to the screen, defaults to True
        :type print: bool, optional
        :return: The camera transform parameters
        :rtype: cv2.detail.CameraParams
        """
        camera_transform = self.camera_transforms_between_rows[row_index]
        if print:
            self.logger.info(f'Camera matrices for row index {row_index}:')
            self.print_camera_transform(camera_transform)
        return camera_transform
    
    def get_row_median_focal(self, row_index:int) -> float:
        """ Returns the median of the parameter focals of the images along the row_index

        :param row_index: The row index of the image
        :type row_index: int
        :return: The median of the parameter focals of the images along the row_index
        :rtype: float
        """
        focals = []
        for col_index in range(self.ncols):
            if len(self.camera_transforms_row_list[row_index]) < self.ncols:
                raise DetectorFailed(DetectorExceptionCodes.RECO_FAILED, f'Unable to obtain camera transforms for every image in the row {row_index}')
            focals.append(self.camera_transforms_row_list[row_index][col_index].focal)
        return median(focals)
    
    def get_between_rows_median_focal(self) -> float:
        """ Returns the median of the parameter focals of the row reconstructed images 

        :return: The median of the parameter focals of the row reconstructed images 
        :rtype: float
        """
        focals = []
        for row_index in range(self.nrows):
            focals.append(self.camera_transforms_between_rows[row_index].focal)
        return median(focals) 
    
    def get_image_map_size(self) -> tuple:
        """ Returns the grid size of the input 2d images

        :return: The grid size (xdim, ydim) of the input 2d images
        :rtype: tuple
        """
        return (self.ncols, self.nrows,)
    
    def get_whole_reco_image_size(self) -> tuple:
        """ Returns the size of the whole reconstructed image

        :return: The size (xdim, ydim) of the whole reconstructed image
        :rtype: tuple
        """
        return (self.whole_reco_image_size[0] * self.scaling_factor, self.whole_reco_image_size[1] * self.scaling_factor,)

    def get_feature_match_image_dict_list(self) -> list:
        """ Returns the list of file names of images showing feature matching for debug purpose

        :return: the list of image file names
        """
        return self.feature_match_image_dict_list

    def print_info(self):
        """ Displays the information of this ImageReconstructModel
        """
        self.logger.info(f'Image Grid Col x Row: {self.ncols, self.nrows}')
        self.logger.info(f'Whole reco image size (scale: {self.working_scale}): {self.whole_reco_image_size}')
        self.logger.info(f'Whole reco image origin offset (scale: {self.working_scale}): {self.whole_reco_image_origin_offset}')
        self.logger.info(f'Rows Camera Matrices')
        for row_index in range(self.nrows):
            for col_index in range(self.ncols):
                self.logger.info(f'Index {col_index, row_index}')
                ImageReconstructModelHelper.print_camera_transform(self.camera_transforms_row_list[row_index][col_index])
        self.logger.info(f'Between Rows Camera Matrices')
        for row_index in range(self.nrows):
            self.logger.info(f'Index {row_index}')
            self.logger.info(f'Reco image size: {self.row_reco_image_sizes[row_index]}')
            self.logger.info(f'Reco image origin offset: {self.row_reco_image_origin_offsets_list[row_index]}')
            ImageReconstructModelHelper.print_camera_transform(self.camera_transforms_between_rows[row_index])           
                

class ImageReconstructModelHelper():
    """ ImageReconstructModelHelper provides helper functions for caching ImageReconstructModel to the file system and retrieve the object

    """
    @staticmethod
    def print_camera_transform(camera_transform:cv2.detail.CameraParams, header='', logger=None):
        """ Displays the camera transform parameters to the screen

        :param camera_transform: The camera parameters to be displayed
        :type camera_transform: cv2.detail.CameraParams
        """
        logger = get_logger() if logger is None else logger
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        if header:
            logger.info(f'{header}')
        logger.info(f'K: {camera_transform.K().reshape(1, -1)}')
        logger.info(f'R: {camera_transform.R.reshape(1, -1)}')
        logger.info(f"T: {camera_transform.t.tolist()}") 
        logger.info(f'Aspect: {camera_transform.aspect} Focal: {camera_transform.focal} PPX and PPY: {camera_transform.ppx, camera_transform.ppy}')
        np.set_printoptions()    

    @staticmethod            
    def to_yaml(reco_model:ImageReconstructModel, object_file:str = None) -> str:
        """ Save an object of ImageReconstructModel to a yaml file

        :param reco_model: The ImageReconstructModel object
        :type reco_model: ImageReconstructModel
        :param object_file: The target file path of the yaml file, defaults to None, which returns the yaml as a string
        :type object_file: str, optional
        :return: The yaml file as a string
        :rtype: str
        """
        object_dict = ImageReconstructModelHelper._to_dict(reco_model)
        if object_file is None:
            return yaml.dump(object_dict, Dumper=yaml.Dumper)
        else:
            try:
                with open(object_file, 'w') as outfile:
                    yamlstr = yaml.dump(object_dict, outfile, Dumper=yaml.Dumper)
            except Exception as e:
                get_logger().warning(f'Failed to save ImageReconstructModel object to {outfile}')
                raise DetectorAborted(DetectorExceptionCodes.OS_ERROR, f'Failed to write ImageReconstructModel to {object_file}', e=e)
            return yamlstr
        
    @staticmethod
    def from_yaml_file(object_file:str) -> ImageReconstructModel:
        """ Load an object of ImageReconstructModel from a yaml file

        :param object_file: The source file path of the yaml file
        :type object_file: str
        :return: An object of ImageReconstructModel loaded from the yaml file 
        :rtype: ImageReconstructModel
        """
        with open(object_file, 'r') as infile:
            data = yaml.load(infile, Loader=yaml.Loader)
        return ImageReconstructModelHelper._create_reco_model(data)
    
    @staticmethod
    def from_yaml(yaml_str:str) -> ImageReconstructModel:
        """ Load an object of ImageReconstructModel from a yaml string

        :param yaml_str: The yaml string
        :type yaml_str: str
        :return: An object of ImageReconstructModel loaded from the yaml string 
        :rtype: ImageReconstructModel
        """
        data = yaml.load(yaml_str, Loader=yaml.Loader)
        return ImageReconstructModelHelper._create_reco_model(data)
    
    @staticmethod
    def _to_dict(reco_model:ImageReconstructModel) -> dict:
        """ An internal function for building a dict from an object of ImageReconstructModel. The dict is then used for conversion into a yaml file

        :param reco_model: An object of ImageReconstructModel
        :type reco_model: ImageReconstructModel
        :return: A dict data structure suitable for conversion into a yaml file
        :rtype: dict
        """
        data = {
            'ncols': reco_model.ncols,
            'nrows': reco_model.nrows,
            'working_scale': reco_model.working_scale,
            'scaling_factor': reco_model.scaling_factor,
        }
        # create the branch for row_camera_matrices
        row_camera_matrices_list = []
        # iterate the image grid to extract the camera transforms (within a row) of every source image and add them to the list
        for row_index in range(reco_model.nrows):
            for col_index in range(reco_model.ncols):
                row_camera_transform = reco_model.camera_transforms_row_list[row_index][col_index]
                camera_dict = CameraTransformTools.camera_to_dict(row_camera_transform)
                camera_dict['col_index'] = col_index
                camera_dict['row_index'] = row_index
                row_camera_matrices_list.append(camera_dict)
        data['row_camera_matrices'] = row_camera_matrices_list
        # create the branch for between_rows_camera_matrices
        between_rows_camera_matrices_list = []
        # iterate the rows to extract the camera transforms (between row) of every row reconstructed image and add them to the list
        for row_index in range(reco_model.nrows):
            camera_dict = CameraTransformTools.camera_to_dict(reco_model.camera_transforms_between_rows[row_index])    
            camera_dict['row_index'] = row_index
            reco_image_size = reco_model.row_reco_image_sizes[row_index]
            camera_dict['row_reco_image_size'] = [*reco_image_size]
            row_reco_image_origin_offset = reco_model.row_reco_image_origin_offsets_list[row_index]
            camera_dict['row_reco_image_origin_offset'] = [*row_reco_image_origin_offset]
            between_rows_camera_matrices_list.append(camera_dict)
        data['between_rows_camera_matrices'] = between_rows_camera_matrices_list 
        # add the confidence matrix values for reference (NOTE: not to be needed for loading of reconstruction model)
        data['confidence_matrix_list'] = []
        for confidence_matrix in reco_model.confidence_matrix_list:
            data['confidence_matrix_list'].append(confidence_matrix.tolist())
        data['confidence_matrix_between_rows'] = reco_model.confidence_matrix_between_rows.tolist()   # not to be needed for loading of reconstruction model
        # add the size and the offset (the output location of camera transform is offseted) of the whole reconstructed image
        data['whole_reco_image_size'] = [*reco_model.whole_reco_image_size]  
        data['whole_reco_image_origin_offset'] = [*reco_model.whole_reco_image_origin_offset]  
        # add the feature matching image dict list
        data['feature_match_image_dict'] = reco_model.feature_match_image_dict_list
        # add the error correction flag
        data['apply_error_correction'] = reco_model.apply_error_correction
        return data
    
    @staticmethod
    def _create_reco_model(data:dict) -> ImageReconstructModel:
        """ Internal function for conversion of the dict structure from yaml file to an object of ImageReconstructModel

        :param data: The data extracted from a yaml file of ImageReconstructModel
        :type data: dict
        :return: An object of ImageReconstructModel
        :rtype: ImageReconstructModel
        """
        reco_model = ImageReconstructModel(None)
        # extract the basic parameters
        reco_model.ncols = data['ncols']
        reco_model.nrows = data['nrows']
        reco_model.working_scale = data['working_scale']
        reco_model.scaling_factor = data['scaling_factor']     
        # extract the row_camera_matrices
        reco_model.camera_transforms_row_list = []
        for row_index in range(reco_model.nrows):
             reco_model.camera_transforms_row_list.append([None] * reco_model.ncols)
        row_camera_matrices_list = data['row_camera_matrices']
        for row_camera_matrices_dict in row_camera_matrices_list:
            row_index, col_index = row_camera_matrices_dict['row_index'], row_camera_matrices_dict['col_index']
            camera_transform = CameraTransformTools.dict_to_camera(row_camera_matrices_dict)
            reco_model.camera_transforms_row_list[row_index][col_index] = camera_transform
        # extract the between_rows_camera_matrices
        reco_model.row_reco_image_sizes = [None] * reco_model.nrows
        reco_model.camera_transforms_between_rows = [None] * reco_model.nrows
        between_rows_camera_matrices_list = data['between_rows_camera_matrices']
        reco_model.row_reco_image_origin_offsets_list = [None] * reco_model.nrows
        for between_rows_camera_matrices_dict in between_rows_camera_matrices_list:
            row_index = between_rows_camera_matrices_dict['row_index']
            camera_transform = CameraTransformTools.dict_to_camera(between_rows_camera_matrices_dict)
            reco_model.camera_transforms_between_rows[row_index] = camera_transform
            row_reco_image_size = between_rows_camera_matrices_dict['row_reco_image_size']
            reco_model.row_reco_image_sizes[row_index] = row_reco_image_size
            row_reco_image_origin_offset = between_rows_camera_matrices_dict['row_reco_image_origin_offset']
            reco_model.row_reco_image_origin_offsets_list[row_index] = row_reco_image_origin_offset
        # add the size and the offset (the output location of camera transform is offseted) of the whole reconstructed image
        reco_model.whole_reco_image_origin_offset = data['whole_reco_image_origin_offset']  
        reco_model.whole_reco_image_size = data['whole_reco_image_size']  
        # add the feature matching image dict list
        reco_model.feature_match_image_dict_list = data['feature_match_image_dict']
        # add the error correction flag
        reco_model.apply_error_correction = data['apply_error_correction']
        if reco_model.apply_error_correction:
            reco_model.error_correction_model = RecoErrorCorrection()
        # initialize frequently used objects when the reco_model is queried
        reco_model._prepare_objects() 
        return reco_model  


class ImageReconstruct2DModel():
    """ ImageReconstruct2DModel models the parameters for the reconstruction of a 2D grid of images. It uses the class ImageReconstruct1DModel to handle the reconustruction of 1D sequence of images, 
        including every row of images in the 2d grid and the row reconstructed images (a merged image of each row)
    """
    def __init__(self, images_2d_list:list, **kwargs):
        """ The constructor, which accepts a list of lists of images in row-major manner. The top-level list contains lists 
        of images in one row. The images are part of a larger image, and the order of the images in the data structure is consistent
        of that the images are foudnd in the larger image

        :param images_as_2dlist: A list of lists of images, each of which can be image paths (str typed) or image pixels (np.ndarray), arranged in a 2D grid
        :type images_as_2dlist: list
        :param working_scale: The scale of the original images at which the reconstruction operations are working on to speed up in exchange of accuracy
        :type working_scale: float        
        """
        # control parameter
        self.to_cancel = False 
        # input parameters
        self.params = kwargs
        self.images_2d_list = images_2d_list
        self.logger = kwargs.get('logger', get_logger())
        self.logdata_folder = kwargs.get(ModelsConfigNames.LOGDATA_FOLDER.value, None)
        self.debug_images_at_original_scale = kwargs.get(ModelsConfigNames.RECO_DEGUG_IMAGE_ORIGINAL_SCALE.value, False)
        self.debug_feature_matching_images = kwargs.get(ModelsConfigNames.RECO_DEBUG_FEATURE_MATCH_IMAGES.value, False)
        self.working_scale = kwargs.get(ModelsConfigNames.RECO_WORKING_SCALE.value, 0.1)
        # model variables     
        self.scaling_factor = 1 / self.working_scale #  the working scale and its inverse
        self.image_map = ImageMap(images_2d_list, working_scale=self.working_scale)   # the image map model that validates the 2d image list and loads the images from the file system if needed
        # list of data structure for recording the results of operations involved in image reconstruction
        self.reco_row_model_list = []                   # a list containing 1d reconstruction models for each row
        self.confidence_matrix_list = []                # a list containing confidence matrix of stitching images in a row
        self.confidence_matrix_between_rows = None       # the confidence matrix of stitching reconstructed images between rows
        self.camera_transforms_row_list = []            # a list containing lists of camera transform of images of a row
        self.camera_transforms_between_rows = None      # a list of camera tansforms between row reconstructed images
        self.row_reco_image_origin_offsets_list = []    # a list containing the origin offset of the row reconstructed images for position transformation
        self.whole_reco_image_origin_offset = None      # the origin offset of the whole reconstructed images for position transformation 
        
        # list of images displaying matching features
        self.feature_match_image_dict_list = []

        # iterate through the rows and construct transformation for each images in a rowrows
        self.ncols, self.nrows = self.image_map.get_image_map_size()
        self.num_steps = (self.ncols + 1) * self.nrows 

    def build(self):
        self.logger.info(f'{type(self).__name__} Step 1: building 1d row reconstruction models for {self.ncols} x {self.nrows} images')
        # preparation: create folder for log and cache files
        try:
            os.makedirs(self.logdata_folder, exist_ok=True)
        except:
            raise DetectorAborted(DetectorExceptionCodes.OS_ERROR, f'Failed to create folder for log and cache files: {self.logdata_folder}')

        # step 1: construct a 1d reconstruction model for each row of images
        failed_rows = []    # the row index of the rows that failed to build ImageReconstruct1DModel
        succeeded_rows = []
        model_build_error:Exception = None
        for row_index in range(self.nrows):
            if self.to_cancel:  # stop processing if abort signal is recieved
                raise DetectorCancelled(DetectorExceptionCodes.CANCELLED_BY_SYSTEM, 'Received an cancel command from the system')
            # if an exception occurs in the constructor of ImageReconstruct1DModel, mark the row as fail and continue with the next row
            try:
                images_in_row = self.image_map.get_row_images_at_working_scale(y = row_index)
                reco_row_model = ImageReconstruct1DModel(images_in_row, row_index=row_index, **self.params)
                succeeded_rows.append(row_index)
            except Exception as e:
                self.reco_row_model_list.append(None)       
                self.confidence_matrix_list.append(None)     
                self.camera_transforms_row_list.append(None)
                self.row_reco_image_origin_offsets_list.append(None)
                failed_rows.append(row_index)
                model_build_error = e
                continue
            
            # extract the relevant model parameters
            self.reco_row_model_list.append(reco_row_model)       
            self.confidence_matrix_list.append(reco_row_model.get_confidence_matrix())     
            self.camera_transforms_row_list.append(reco_row_model.get_camera_transforms_row())
            self.row_reco_image_origin_offsets_list.append(reco_row_model.get_reco_image_origin_offset())
            # save the feature matching image for debug purpose if the flag is on
            if self.debug_feature_matching_images:
                debug_images = reco_row_model.get_debug_images_feature_matching()
                for image_index_1, image_index_2, image in debug_images:
                    image_file_name = f'feature_match_row_images_{image_index_1}_{row_index}_{image_index_2}_{row_index}.jpg'
                    image_dict = {'title': f'Feature matching between column {image_index_1} and {image_index_2} on row {row_index}', 'src': image_file_name}
                    self.feature_match_image_dict_list.append(image_dict)
                    image_file = os.path.join(self.logdata_folder, image_file_name)
                    if not cv2.imwrite(image_file, image):
                        raise DetectorAborted(DetectorExceptionCodes.OS_ERROR, f'Failed to write feature matching output to {image_file}')
        # step 1A: for each row in that reconsturction was failed, use the nearest row
        if len(succeeded_rows) == 0:
            # if no row was successfully reconstructed, throw the error
            raise model_build_error
        for failed_row in failed_rows:
            nearest_row = None
            for succeded_row in succeeded_rows:
                if nearest_row is None or (abs(succeded_row - failed_row) < abs(nearest_row - failed_row)):
                    nearest_row = succeded_row
            if nearest_row is not None:
                self.logger.warning(f'{type(self).__name__} Step 1A: Failed to reconstruct row {failed_row}, use the model for row {nearest_row} as the fallback')
                self.reco_row_model_list[failed_row] = self.reco_row_model_list[nearest_row] 
                self.confidence_matrix_list[failed_row] = self.confidence_matrix_list[nearest_row]
                self.camera_transforms_row_list[failed_row] = self.camera_transforms_row_list[nearest_row]
                self.row_reco_image_origin_offsets_list[failed_row] = self.row_reco_image_origin_offsets_list[nearest_row]

        # step 2: generate the reconstructed images row by row at the current scale
        self.logger.info(f'{type(self).__name__} Step 2: generating {self.nrows} reconstructed row images')
        row_recoimages_list = []
        output_file = None
        for row_index in range(self.nrows):
            if self.to_cancel:  # stop processing if abort signal is recieved
                raise DetectorCancelled(DetectorExceptionCodes.CANCELLED_BY_SYSTEM, 'Received an cancel command from the system')
            images_in_row = self.image_map.get_row_images_at_working_scale(y = row_index)
            output_file = os.path.join(self.logdata_folder, f'row_reco_image_{row_index}.jpg') if self.logdata_folder is not None else None
            row_recoimage, normalized_warped_roi_corners, warped_roi_sizes = self._generate_1d_recoimage_with_scaling(images_in_row, self.camera_transforms_row_list[row_index], 
                                                                                                             images_list_scaling_factor=1.0, output_file=output_file)
            row_recoimages_list.append(row_recoimage)
            if output_file is not None:  # save the images only if output_folder is provided
                self.logger.info(f'{type(self).__name__}: Writing 1d row reconstructed image (size: {row_recoimage.shape[:2][::-1]}) to file {output_file}')
                if not cv2.imwrite(output_file, row_recoimage):
                    raise DetectorAborted(DetectorExceptionCodes.OS_ERROR, f'Failed to write row reconstructed image to {output_file}')

        # step 3: if there is only one row, skip the rest
        if len(row_recoimages_list) == 1:
            whole_reco_image = row_recoimages_list[0]
            self.camera_transforms_between_rows = self.reco_row_model_list[0].get_camera_transforms_row()
            self.whole_reco_image_origin_offset = self.reco_row_model_list[0].get_reco_image_origin_offset()
            self.confidence_matrix_between_rows = np.asarray([[1.0]], dtype=np.float32)
        else:
            # step 4: prepare between rows image stitching by rotating each row reco images counterclockwise 
            self.logger.info(f'{type(self).__name__} Step 4: Rotate counter-clockwise the {self.nrows} reconstructed row images')
            row_recoimages_rotated_list = self._rotate_cw90_images_list(row_recoimages_list, self.logdata_folder)
            
            # step 4: build the between rows reconstruction model
            self.logger.info(f'{type(self).__name__} Step 5: Build the top-level 1d reconstruction model from the {self.nrows} rotated images')
            reco_whole_model = ImageReconstruct1DModel(row_recoimages_rotated_list, row_index=None, **self.params)
            # save the debug matching images if the flag is on
            if self.debug_feature_matching_images:        
                debug_images = reco_whole_model.get_debug_images_feature_matching()
                for image_index_1, image_index_2, image in debug_images:
                    if self.to_cancel:  # stop processing if abort signal is recieved
                        raise DetectorCancelled(DetectorExceptionCodes.CANCELLED_BY_SYSTEM, 'Received an cancel command from the system')
                    image_file_name = f'feature_match_between_rows_images_{image_index_1}_{image_index_2}.jpg'
                    image_dict = {'title': f'Feature matching between rows {image_index_1} and {image_index_2}', 'src': image_file_name}
                    self.feature_match_image_dict_list.append(image_dict)
                    image_file = os.path.join(self.logdata_folder, image_file_name)
                    if not cv2.imwrite(image_file, cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)):
                        raise DetectorAborted(DetectorExceptionCodes.OS_ERROR, f'Failed to write feature matching results between row reconstructed images to {image_file}')
            # using the computed transforms to generate the reconstructed image for the input 2d grid of images
            self.confidence_matrix_between_rows = reco_whole_model.get_confidence_matrix() 
            self.camera_transforms_between_rows = reco_whole_model.get_camera_transforms_row() 
            self.whole_reco_image_origin_offset = reco_whole_model.get_reco_image_origin_offset()
            if self.to_cancel:  # stop processing if abort signal is recieved
                raise DetectorCancelled(DetectorExceptionCodes.CANCELLED_BY_SYSTEM, 'Received an cancel command from the system')
            whole_reco_image, normalized_warped_roi_corners, warped_roi_sizes = self._generate_1d_recoimage_with_scaling(row_recoimages_rotated_list, self.camera_transforms_between_rows, 
                                                                                                                images_list_scaling_factor=1.0)
            # write the whole reconstructed image (at working scale) to the logdata folder
            whole_reco_image = cv2.rotate(whole_reco_image, cv2.ROTATE_90_CLOCKWISE)
        if self.to_cancel:  # stop processing if abort signal is recieved
            raise DetectorCancelled(DetectorExceptionCodes.CANCELLED_BY_SYSTEM, 'Received an cancel command from the system')
        # save the whole reco images only if output_folder is provided
        if self.logdata_folder is not None:  
            output_file = os.path.join(self.logdata_folder, ImageReconstructModel.FILENAME_WHOLE_RECO_IMAGE)
            self.logger.info(f'{type(self).__name__}: Writing whole reconstructed image (size: {whole_reco_image.shape[:2][::-1]}) to file {output_file}')
            if not cv2.imwrite(output_file, whole_reco_image):
                raise DetectorAborted(DetectorExceptionCodes.OS_ERROR, f'Failed to write whole reconstructed image to {output_file}')
        self.whole_reco_image_size = whole_reco_image.shape[:2][::-1]
        if self.to_cancel:  # stop processing if abort signal is recieved
            raise DetectorCancelled(DetectorExceptionCodes.CANCELLED_BY_SYSTEM, 'Received an cancel command from the system')
        # step 5: generate the debug images in original scale
        if self.logdata_folder and self.debug_images_at_original_scale:
            images_list_scaling_factor = 1 / self.working_scale
            self._generate_whole_recoimage_with_scaling(self.images_2d_list, self.camera_transforms_row_list, self.camera_transforms_between_rows,
                                                                                          images_list_scaling_factor, logdata_folder=self.logdata_folder)
    def cancel_build(self):
        """ abort the current reconstruction process """
        self.logger.warning(f'ImageReconstruct2DModel: received ABORT')
        self.to_cancel = True    
            
    def get_image_map_size(self) -> tuple:
        """ 

        :return: The size (xdim, ydim) of the input 2d image grid
        :rtype: tuple
        """
        return self.image_map.get_image_map_size()
    
    def get_working_image_size(self) -> tuple:
        """ 

        :return: The size (xdim, ydim) of the images at the working scale
        :rtype: tuple
        """
        return self.image_map.get_scaled_image_size()
    
    def get_whole_reco_image_size(self) -> tuple:
        """ Returns the size of the whole reconstructed image

        :return: The size (xdim, ydim) of the whole reconstructed image
        :rtype: tuple
        """
        return self.whole_reco_image_size  

    def get_confidence_matrix_list(self) -> list:
        """ Returns the list of confidence matrix for reconstructing 1d row image in this ImageReconstruct2DModel. Each element of this list is 2d confidence matrix

        :return: The list of confidence matrix for reconstructing 1d row image
        :rtype: list
        """
        return self.confidence_matrix_list
    
    def get_confidence_matrix_between_rows(self) -> np.ndarray:
        """ Returns the confidence matrix for reconstructing the whole image from 1d row images

        :return: The confidence matrix for reconstructing the whole image from 1d row images
        :rtype: np.ndarray
        """
        return self.confidence_matrix_between_rows   
    
    def get_camera_transforms_row_list(self) -> list:
        """ Returns the 2d list of camera transform for this ImageReconstruct2DModel. Each element of this list is a list of camera transform of a row of images

        :return: The 2d list of camera transform for this ImageReconstruct2DModel
        :rtype: list
        """
        return self.camera_transforms_row_list
    
    def get_camera_transforms_between_rows(self) -> list:
        """ Returns the list of camera transforms for merging the row reconstructed images

        :return: The list of camera transforms for merging the row reconstructed images
        :rtype: list
        """
        return self.camera_transforms_between_rows   
    
    def get_row_reco_image_origin_offsets_list(self) -> list:
        """ Returns the origin offsets of all row reconstructed images for location transformation as a list of tuples

        :return: the origin offsets of all row reconstructed images 
        :rtype: list
        """
        return self.row_reco_image_origin_offsets_list
    
    def get_whole_reco_image_origin_offset(self) -> tuple:
        """ Returns the origin offset of the whole reconstructed image

        :return: the origin offset of the whole reconstructed image
        :rtype: tuple
        """
        return self.whole_reco_image_origin_offset    
    
    def get_feature_match_image_dict_list(self) -> list:
        """ Returns the list of file names of images showing feature matching for debug purpose

        :return: the list of image file names
        """
        return self.feature_match_image_dict_list
    
    @staticmethod
    def _load_images(images_list:list) -> list:
        """ Internal function to load the image from file if any element in the list is a str representing an image file path

        :param images_list: a list of either a numpy array image or a string representing an image file path
        :type images_list: list
        :raises AssertionError: An element in the list which is neither a file path nor a numpy image
        :return: a list of numpy images
        :rtype: list
        """
        loaded_images_list = []
        for image in images_list:
            if type(image) == str:
                image = cv2.imread(image)
            if type(image) == np.ndarray:
                loaded_images_list.append(image)
            else:
                raise AssertionError(f'_load_images: Parameter images_list contains an object which is neither a file path nor a numpy image')
        return loaded_images_list      

    @staticmethod
    def _rotate_cw90_images_list(images_list:list, debug_folder=None) -> list:
        """ Interal function to rotate the images in a list 90 degrees counter-clockwise

        :param images_list: a list of numpy images to be rotated
        :type images_list: list
        :param debug_folder: The folder to which the rotated images are written for logging, defaults to None
        :type debug_folder: _type_, optional
        :return: the same list with the images rotated
        :rtype: list
        """
        # rotate the 1d row reconstructed images counter-clockwise
        images_rotated_list = []
        for row_index, row_recoimage in enumerate(images_list):
            image_rotated = cv2.rotate(row_recoimage, cv2.ROTATE_90_COUNTERCLOCKWISE)
            images_rotated_list.append(image_rotated)
            if debug_folder:
                output_file = os.path.join(debug_folder, f'row_reco_image_{row_index}_rotated.jpg')
                if not cv2.imwrite(output_file, image_rotated):
                    raise DetectorAborted(DetectorExceptionCodes.OS_ERROR, f'Failed to write rotated reconstructed image to {output_file}')
        return images_rotated_list

    def _generate_whole_recoimage_with_scaling(self, images_2d_list:list, camera_transforms_row_list:list, camera_transforms_between_rows:list, images_list_scaling_factor:float=1.0, logdata_folder:str=None) -> np.ndarray:
        """ Generates and returns the whole reconstructed images from the source images in the given image_2d_list and the camera transforms already computed

        :param images_2d_list: The source images to be reonconstructed as a whole image, arranged in a 2D list (list of list of images)
        :type images_2d_list: list
        :param camera_transforms_row_list: A list of lists of camera transforms within the rows 
        :type camera_transforms_row_list: list
        :param camera_transforms_between_rows: A list of camera transform for the row reconstructed images
        :type camera_transforms_between_rows: list
        :param images_list_scaling_factor: The scaling factor of the source images with reference to the working scale, defaults to 1.0
        :type images_list_scaling_factor: float, optional
        :param logdata_folder: The target folder where the images are written to, defaults to None
        :type logdata_folder: str, optional
        :return: the whole reconstructed image
        :rtype: np.ndarray
        """
        row_recoimages_list = []
        # generate the merged image for each row of images
        nrows = len(images_2d_list)
        for row_index in range(nrows):
            images_list, camera_tranforms_row = images_2d_list[row_index], camera_transforms_row_list[row_index]
            images_list = self._load_images(images_list)
            
            row_reco_image_original_scale, normalized_warped_roi_corners, warped_roi_sizes = ImageReconstruct1DModel.generate_reco_image(images_list, camera_tranforms_row, 
                                                                                                                                images_list_scaling_factor)
            if logdata_folder is not None:
                output_file = os.path.join(logdata_folder, f'row_reco_image_{row_index}_original_scale.jpg')
                self.logger.info(f'{type(self).__name__}: Writing full-scale row reconstructed image to file {output_file}')
                if not cv2.imwrite(output_file, row_reco_image_original_scale):
                   raise DetectorAborted(DetectorExceptionCodes.OS_ERROR, f'Failed to write whole reconstructed image at original scale to {output_file}')
                
            row_reco_image_original_scale = cv2.rotate(row_reco_image_original_scale, cv2.ROTATE_90_COUNTERCLOCKWISE)
            row_recoimages_list.append(row_reco_image_original_scale)
            
        # generate the whole merged image from the rotated row reconstructed images
        whole_reco_image_original_scale, normalized_warped_roi_corners, warped_roi_sizes = ImageReconstruct1DModel.generate_reco_image(row_recoimages_list, camera_transforms_between_rows, 
                                                                                                                            images_list_scaling_factor)
        whole_reco_image_original_scale = cv2.rotate(whole_reco_image_original_scale, cv2.ROTATE_90_CLOCKWISE)
        if logdata_folder is not None:
            output_file = os.path.join(logdata_folder, ImageReconstructModel.FILENAME_WHOLE_RECO_FULL_SCALE_IMAGE)
            self.logger.info(f'{type(self).__name__}: Writing whole full-scale reconstructed image to file {output_file}')
            if not cv2.imwrite(output_file, whole_reco_image_original_scale):
                raise DetectorAborted(DetectorExceptionCodes.OS_ERROR, f'Failed to write whole reconstructed image at original scale to {output_file}')
            
        return whole_reco_image_original_scale
        
    def _generate_1d_recoimage_with_scaling(self, images_list:list, camera_tranforms_row:list, images_list_scaling_factor:float=1.0, output_file:str=None) :
        """ Generates and returns a row reconstructed image from a list of images of a row and the corresponding camera transforms 

        :param images_list: The source images to be merged into a row reconstructed image as a list
        :type images_list: list
        :param camera_tranforms_row: A list of camera transforms within the row
        :type camera_tranforms_row: list
        :param images_list_scaling_factor: The scaling factor of the source images with reference to the working scale, defaults to 1.0
        :type images_list_scaling_factor: float, optional
        :param output_file: the target file where the image is written to, defaults to None, defaults to None
        :type output_file: str, optional
        :return: a tuple of (the row reconstructed image, the list of corners of the transformed source images, the list of sizes of the transformed source images)  
        :rtype: tuple
        """
        row_reco_image, normalized_warped_roi_corners, warped_roi_sizes = ImageReconstruct1DModel.generate_reco_image(images_list, camera_tranforms_row, images_list_scaling_factor)
        if output_file is not None:  # save the images only if output_folder is provided
            self.logger.info(f'{type(self).__name__}: Writing 1d reconstructed image (size: {row_reco_image.shape[:2][::-1]}) to file {output_file}')
            if not cv2.imwrite(output_file, row_reco_image):
                raise DetectorAborted(DetectorExceptionCodes.OS_ERROR, f'Failed to write row reconstructed image at scale to {output_file}')
        return row_reco_image, normalized_warped_roi_corners, warped_roi_sizes 
 
            
class ImageReconstruct1DModel():
    """ ImageReconstruct1DModel computes critical parameters based on feature extraction and matching to enable stitching a horizontal list of images into one image
        The class returns a list of camera matricses (K, R, and T) for each image in the list that enables them to be reconstructed into one
    """
    def __init__(self, images_1d_list:list, row_index, **kwargs):
        """ internal class that models the reconstruction of a 1D horizontal sequence of images

        :param images_1d_list: A list of images in the order from left to right
        :type images_1d_list: list
        """
        assert images_1d_list is not None and len(images_1d_list) > 0, 'Parameter images_as_list is empty'
        assert type(images_1d_list[0]) == np.ndarray, 'Parameter images_as_list should contain only numpy images'


        
        # input parameters
        self.logger = kwargs.get('logger', get_logger())
        logdata_folder = kwargs.get(ModelsConfigNames.LOGDATA_FOLDER.value, None)
        debug_feature_matching_images = kwargs.get(ModelsConfigNames.RECO_DEBUG_FEATURE_MATCH_IMAGES.value, False)
        # input parameters and model variables  
        self.num_images_in_list = len(images_1d_list)      
        feature_matching_confidence_threshold = kwargs.get(ModelsConfigNames.RECO_FEATURE_MATCHING_CONFIDENCE_THRESHOLD.value, 1.0)  # for matching features between two images
        if row_index is None:
            image_matching_min_confidence = kwargs.get(ModelsConfigNames.RECO_IMAGE2D_MATCHING_MIN_CONFIDENCE.value, 1.0) 
        else:
            image_matching_min_confidence = kwargs.get(ModelsConfigNames.RECO_IMAGE_MATCHING_MIN_CONFIDENCE.value, 1.0) 
        # input parameters: reconstruction error checking
        aspect_ratio_roi_error_rel = kwargs.get(ModelsConfigNames.RECO_ASPECT_RATIO_ROI_ERROR_REL.value, 0.1)
        misplaced_roi_error_rel = kwargs.get(ModelsConfigNames.RECO_MISPLACED_ROI_ERROR_REL.value, 0.1)
        # get image size
        self.images_as_list = images_1d_list
        # step 1: extract the keyword parameters
        self.image_sizes, self.image_nchannels = ImageReconstruct1DModel.extract_image_sizes_nchannels(images_1d_list)
        # self.image_size = self.image_sizes[0]
        self.logger.info(f'{type(self).__name__}: Number of images: {len(images_1d_list)}')
        # test if there is a single image
        # reco_image_origin_offset = (0, 0)
        # camera_transforms_row
        if len(images_1d_list) == 1:
            self.reco_image_origin_offset = (0, 0,)
            data = {
                'R': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                't': [0, 0, 0],
                'aspect': 0.0,
                'focal': 1.0,
                'ppx': self.image_sizes[0][0] / 2,
                'ppy': self.image_sizes[0][1] / 2,
            }
            self.camera_transforms_row = [CameraTransformTools.dict_to_camera(data)]
            self.confidence_matrix = np.asarray([[1.0]], dtype=np.float32)
            self.debug_images_feature_matching = []
            return

        # attempt to build model with different combinations of feature extractors
        model_build_success:bool = False
        model_build_error:Exception = None
        # the desperate flag
        tried_desperate = False
        # the list of hyper-parameters for search
        default_feature_detectors = ['akaze', 'brisk', 'sift']
        try_feature_detectors = kwargs.get(ModelsConfigNames.RECO_FEATURE_DETECTORS.value, default_feature_detectors)
        self.logger.info(f'try_feature_detectors: {try_feature_detectors}')
        if type(try_feature_detectors) == str:
            try_feature_detectors = [try_feature_detectors]
        try_matcher_types = ['affine']
        param_search_list = self._generate_parameter_search(try_feature_detectors, try_matcher_types, image_matching_min_confidence)  
        # attempt to reconstruct using each of the parameter sets
        while True:
            if len(param_search_list) == 0:
                if not tried_desperate:
                    self.logger.info(f'{type(self).__name__}: All parameter combinations tried but failed, go to desperate mode and use 0.5 of min threshold in the confidence matrix')
                    param_search_list = self._generate_parameter_search(try_feature_detectors, try_matcher_types, image_matching_min_confidence * 0.5)  # tried half of the threshold  
                    tried_desperate = True
                else:
                    break
            search_param = param_search_list.pop(0)
            feature_detector, matcher_type, matrix_min_conf = search_param
            self.logger.info(f'{type(self).__name__}: Attempt reconstruction of 1d image using parameters {feature_detector} {matcher_type} conf_matrix_min {matrix_min_conf}')
            try: 
                # step 2: extract features
                self.features_finder = FeatureDetector(detector=feature_detector)  # orb, sift, brisk, akaze
                self.features = [self.features_finder.detect_features(image) for image in images_1d_list]
                # step 3: match features between all pairs of images
                self.features_matcher = FeatureMatcher(matcher_type=matcher_type)  # 'affine' or 'homography', only affine works for tile reconstruction
                self.matches_raw = self.features_matcher.match_features(self.features)
                # NOTE: a hack into the stitching module to overcome the bug in working out the sequence by assuming the ordering the image list is true
                self.matches = []
                matchinfo:cv2.detail.MatchesInfo
                for matchinfo in self.matches_raw:
                    if matchinfo.dst_img_idx == matchinfo.src_img_idx + 1:
                        self.matches.append(matchinfo)
                    else:
                        # fix the next matching image for each image
                        replace_matchinfo = cv2.detail.MatchesInfo()
                        replace_matchinfo.src_img_idx = matchinfo.src_img_idx
                        replace_matchinfo.dst_img_idx = matchinfo.dst_img_idx
                        replace_matchinfo.confidence = replace_matchinfo.num_inliers = 0
                        replace_matchinfo.H = matchinfo.H
                        replace_matchinfo.inliers_mask = matchinfo.inliers_mask
                        self.matches.append(replace_matchinfo)
                # evaluate the confidence matrix
                self.confidence_matrix = self.features_matcher.get_confidence_matrix(self.matches)
                self.logger.info(f'The confidence matrix:\n{self.confidence_matrix}')
                min_value = None
                for image_index in range(0, len(images_1d_list) - 1):
                    if min_value is None or self.confidence_matrix[image_index, image_index + 1] < min_value:
                        min_value = self.confidence_matrix[image_index, image_index + 1]
                if min_value < matrix_min_conf:
                    raise DetectorFailed(DetectorExceptionCodes.RECO_MATCH_FAILED, f'Cannot merge adjacent images: possibly error in image capturing')
                # step 4: save the images annotated with matching results to a class variables 
                self.debug_images_feature_matching = None
                if debug_feature_matching_images:
                    self.debug_images_feature_matching = self.features_matcher.draw_matches_matrix(images_1d_list, self.features, self.matches, conf_thresh=feature_matching_confidence_threshold, 
                                                        inliers=True, matchColor=(0, 255, 0))
                    # for image_index_1, image_index_2, image in self.debug_images_feature_matching:
                    #     image_file = os.path.join(logdata_folder, f'feature_match_row_images_{image_index_1}_{row_index}_{image_index_2}_{row_index}.jpg')
                    #     cv2.imwrite(image_file, image)
                try:
                    # step 5: split the features and matches group by the images
                    subsetter = Subsetter(confidence_threshold=feature_matching_confidence_threshold)
                    indices = subsetter.get_indices_to_keep(self.features, self.matches)
                    
                    subset_images_1d_list = subsetter.subset_list(images_1d_list, indices)
                    self.features = subsetter.subset_list(self.features, indices)
                    self.matches = subsetter.subset_matches(self.matches, indices)
                    # step 6: estimate the camera intrinsics and rotation matrics based on the matched features and their locations
                    camera_estimator = CameraEstimator()
                    camera_adjuster = CameraAdjuster(adjuster='ray')  # use 'ray' ('reproj", 'affine', 'no' have been tested and all failed)
                    wave_corrector = WaveCorrector()        
                    camera_transforms_row = camera_estimator.estimate(self.features, self.matches)
                    # self._print_camera_transforms_row(camera_transforms_row)
                    camera_transforms_row = camera_adjuster.adjust(self.features, self.matches, camera_transforms_row)
                    self.camera_transforms_row = wave_corrector.correct(camera_transforms_row)

                    # step 7: compute roi for each image and normalize the locations 
                    self.roi_corners, self.normalized_roi_corners, self.roi_sizes = ImageReconstruct1DModel._compute_warp_rois(self.image_sizes, self.camera_transforms_row, scale = 1)

                    # step 8: check if the warp_rois were computed for all images in the row
                    self.logger.info(f'ROI Corners: {self.roi_corners}')
                    self.logger.info(f'Normalized ROI Corners: {self.normalized_roi_corners}')
                    self.logger.info(f'ROI Sizes: {self.roi_sizes}')
                    # error checking 1: warp rois cannot be found for one or more images
                    if len(self.roi_corners) != self.num_images_in_list:
                        raise DetectorFailed(DetectorExceptionCodes.RECO_FAILED, f'Cannot find warp rois for all images in the row') 
                    # error checking 2: roi size is not reasonable for one or more images
                    if not self._is_aspect_ratio_roi_sizes_reasonable(subset_images_1d_list, self.roi_sizes, tol_rel=aspect_ratio_roi_error_rel):
                        raise DetectorFailed(DetectorExceptionCodes.RECO_FAILED, f'The aspect ratio of one or more rois is different from the original image (max rel error: {aspect_ratio_roi_error_rel})') 
                    # error checking 3:
                    if not self._is_roi_corners_reasonable(self.roi_corners, self.roi_sizes):
                        raise DetectorFailed(DetectorExceptionCodes.RECO_FAILED, f'_is_roi_corners_reasonable: roi corners not regularly placed') 
                    self.reco_image_origin_offset = self._find_reco_image_origin_offset(self.roi_corners)
                    self.logger.info(f'ImageID reco_iamge_origin_offset: {self.reco_image_origin_offset}')
                    self.roi_corners = self._normalize_corners(self.roi_corners, self.reco_image_origin_offset)
                    # step 9: estimate roi of the reconstructed image
                    self.estimated_reco_image_size = self._estimate_model_outer_roi_size(self.roi_corners, self.roi_sizes)
                    self.logger.info(f'Estimated reconstructed image size: {self.estimated_reco_image_size} and\ncorners: {self.roi_corners}')
                    model_build_success = True
                    break
                except Exception as e:
                    self.logger.info(e)
                    raise DetectorFailed(DetectorExceptionCodes.RECO_FAILED, f'Cannot combine images into a grid: wayward homography due to inadequate quality unique features')
            except DetectorFailed as e:
                model_build_error = e            
            except Exception as e:
                self.logger.error(traceback.format_exc())
                model_build_error = e
        # raise the Exception if the model build was not successful 
        if not model_build_success and model_build_error is not None:
            raise model_build_error
        
    def _generate_parameter_search(self, try_feature_detectors, try_matcher_types, conf_matrix_min_confidence):
        # setup the list of parameters to search 
        try_min_confidence = [conf_matrix_min_confidence]
        param_search_list = []
        for fd in try_feature_detectors:
            for mt in try_matcher_types:
                for mc in try_min_confidence:
                    param_search_list.append((fd, mt, mc,))
        return param_search_list

    def _is_aspect_ratio_roi_sizes_reasonable(self, images_1d_list:list, roi_sizes:list, tol_rel:float=0.1) -> bool:
        if len(images_1d_list) != len(roi_sizes):
            self.logger.info(f'_is_roi_sizes_reasonable: input parameters are of different lengths')
            return False
        for image, roi in zip(images_1d_list, roi_sizes):
            image_size = image.shape[:2][::-1]
            if abs(image_size[0] - roi[0]) / image_size[0] > tol_rel or abs(image_size[1] - roi[1]) / image_size[1] > tol_rel:
                return False
        return True
    
    def _is_roi_corners_reasonable(self, roi_corners:list, roi_sizes:list, corner_tol_rel:float=0.1) -> bool:
        # evaluate roi corners
        x_diff_list = []
        y_diff_list = []
        # evaluate the median
        for index in range(1, len(roi_corners)):
            x_diff_list.append(roi_corners[index][0] - roi_corners[index - 1][0])
            y_diff_list.append(roi_corners[index][1] - roi_corners[index - 1][1])
        x_median, y_median = median(x_diff_list), median(y_diff_list)
        # evaluate the difference
        for index in range(1, len(roi_corners)):
            x_diff = abs(roi_corners[index][0] - roi_corners[index - 1][0] - x_median) / roi_sizes[index][0]
            y_diff = abs(roi_corners[index][1] - roi_corners[index - 1][1] - y_median) / roi_sizes[index][1]
            if x_diff > corner_tol_rel or y_diff > corner_tol_rel:
                return False
        return True

    @classmethod
    def _print_camera_transforms_row(cls, camera_transforms_row, logger):
        for index, camera_transform in enumerate(camera_transforms_row):
            ImageReconstructModelHelper.print_camera_transform(camera_transform, f'Camera transform matrices for image index {index}:', logger)

    def get_camera_transforms_row(self) -> list:
        """ Returns the camera transforms of this 1D sequence of images

        :return: A list of camera transforms within the sequence
        :rtype: list
        """
        return self.camera_transforms_row

    def get_camera_transform(self, index:int) -> cv2.detail.CameraParams:
        """ Returns the camera transform object of the image at a given index (NOT USED)

        :param index: The index of the image 
        :type index: int
        :param print: To display the camera transforms to the screen, defaults to False
        :type print: bool, optional
        :return: The camera transform parameters
        :rtype: cv2.detail.CameraParams
        """
        assert index is not None and index >= 0 and index < len(self.camera_transforms_row), 'Parameter index is invalid'
        camera_transform = self.camera_transforms_row[index]
        return camera_transform

    def get_confidence_matrix(self, print=False) -> np.ndarray:
        """ Returns the confidence matrix between all pairs of input source images

        :param print: to display the results to the screen, defaults to False
        :type print: bool, optional
        :return: the confidence matrix
        :rtype: np.ndarray
        """
        if print:
            self.logger.info(f'Confidence matrix:\n{self.confidence_matrix}')
        return self.confidence_matrix

    def get_roi_corners_sizes(self) -> list:
        """ Returns the roi corners, normalized roi corners, and roi sizes of the image list (NOT USED)

        :return: A 3-tuple
        :rtype: 3-tuple of list
        """
        return self.roi_corners, self.normalized_roi_corners, self.roi_sizes

    # def get_reco_image_size(self, print=True):
    #     if print:
    #         self.logger.info(f'Reconstructed Image Size: {self.estimated_reco_image_size}')
    #     return self.estimated_reco_image_size
    
    # def get_roi_corners(self, print=True):
    #     if print:
    #         self.logger.info(f'ROI Corners of Original Images after Reconstruction: {self.roi_corners}')
    #     return self.roi_corners
    
    def get_reco_image_origin_offset(self, print=True):
        if print:
            self.logger.info(f'Origin Offset of Reconstructed Image: {self.reco_image_origin_offset}')
        return self.reco_image_origin_offset       

    def get_debug_images_feature_matching(self):
        return self.debug_images_feature_matching

    @staticmethod
    def _compute_warp_roi(image_size:list, camera_transform:cv2.detail.CameraParams, focal:float=None, scale:float=1.0):
        """ Internal function to compute the ROI of an image given the size and the camera transform parameters

        :param image_size: the size (xdim, ydim) of the image of which the ROI of its transformed version is computed
        :type image_size: tuple or list
        :param camera_transform: the camera transform parameters corresponding to the image
        :type camera_transform: cv2.detail.CameraParams
        :param focal: the focal parameter if given will override that in the camera transform, defaults to None
        :type focal: float, optional
        :param scale: the scale of the image compared to the source input images, defaults to 1.0
        :type scale: float, optional
        :return: a 4-tuple indicating the top-left location and size (x, y, size_x, size_y)
        :rtype: tuple
        """
        # create warper object
        if focal is None:
            focal = camera_transform.focal
        warper = cv2.PyRotationWarper("spherical", focal * scale)
        K = Warper.get_K(camera_transform, scale)
        roi = warper.warpRoi(image_size, K, camera_transform.R)
        return roi
    
    @staticmethod
    def _compute_warp_rois(image_sizes:list, camera_tranforms_row:list, scale:float=1.0):
        """ Internal function to compute the ROI of images given the sizes and the camera transform parameters as lists

        :param image_sizes: a list of sizes of the images of which the ROI of their transformed versions are computed
        :type image_sizes: list
        :param camera_tranforms_row: a list of the camera transform parameters corresponding to the images
        :type camera_tranforms_row: list
        :param scale: the resize scale of the images compared to the source input images, defaults to 1.0
        :type scale: float, optional
        :param print: to display the calculation results to the screen, defaults to False
        :type print: bool, optional
        :return: a tuple of (a list of 2-tuple top-left corners, a list of sizes of the ROIs)
        :rtype: tuple
        """
        roi_corners = []
        roi_sizes = []
        focals = [cam.focal for cam in camera_tranforms_row]
        median_focals = median(focals)
        for index, (image_size, camera_transform) in enumerate(zip(image_sizes, camera_tranforms_row)):
            roi = ImageReconstruct1DModel._compute_warp_roi(image_size, camera_transform, median_focals, scale)            
            roi_corners.append(roi[0:2])
            roi_sizes.append(roi[2:4])
        # compute normalized roi_corners
        reco_image_origin_offset = ImageReconstruct1DModel._find_reco_image_origin_offset(roi_corners)
        normalized_roi_corners = ImageReconstruct1DModel._normalize_corners(roi_corners, reco_image_origin_offset)
        return roi_corners, normalized_roi_corners, roi_sizes
    
    @staticmethod
    def _find_reco_image_origin_offset(roi_corners:list):
        """ From a list of ROI corners (2-tuple top-left), computes and returns the location for setting the origin (0, 0)

        """
        tl_min_x, tl_min_y = roi_corners[0][0], roi_corners[0][1]
        for corner in roi_corners:
            tl_min_x, tl_min_y = min(corner[0], tl_min_x), min(corner[1], tl_min_y)
        return (tl_min_x, tl_min_y,)        
    
    @staticmethod
    def _normalize_corners(roi_corners:list, origin_offset:tuple):
        """ Returns a list of nornalized corner locations from the original corners and the origin offsets 

        :param roi_corners: a list of top-left corners of images
        :type roi_corners: list
        :param origin_offset: the location for setting the origin
        :type origin_offset: tuple
        """
        new_roi_corners = []
        for corner in roi_corners:
            new_roi_corners.append((corner[0] - origin_offset[0], corner[1] - origin_offset[1]))
        return new_roi_corners

    @staticmethod
    def scale_corners(roi_corners:list, scaling_factor:float):
        """ Returns a list of new corner locations if the images are scaled by the given scaling factor

        :param roi_corners: a list of top-left corners of images
        :type roi_corners: list
        :param scaling_factor: the image resize factor 
        :type scaling_factor: float
        """        
        new_roi_corners = []
        for corner in roi_corners:
            scaled_corner = (int(corner[0] * scaling_factor), int(corner[1] * scaling_factor))
            new_roi_corners.append(scaled_corner)
        return new_roi_corners

    @staticmethod
    def _estimate_model_outer_roi_size(roi_corners:list, roi_sizes:list) -> tuple:
        """ Returns the size of the bounding box that encampasses the images given their corners and sizes as lists

        :param roi_corners: The top-left corners of the images
        :type roi_corners: list
        :param roi_sizes: The size of the images
        :type roi_sizes: list
        :return: the size of the outer bounding ROI 
        :rtype: tuple
        """
        br_max_x, br_max_y = -1, -1  # initial value of size -1, -1
        for corner, size in zip(roi_corners, roi_sizes):
            br_max_x, br_max_y = max(corner[0] + size[0], br_max_x), max(corner[1] + size[1], br_max_y)
        return (br_max_x, br_max_y)

    @staticmethod
    def extract_image_sizes_nchannels(images_as_list:list):
        """ Returns a tuple of two lists. The first list is the sizes of the image in the input images list and the second list is the number of channels of the input images

        :param images_as_list: A list of source images
        :type images_as_list: list
        :return: a tuple of two lists of which the first list is the sizes of the image in the input images list and the second list is the number of channels of the input images
        :rtype: tuple
        """
        image_sizes = []
        image_nchannels = []
        for image in images_as_list:
            image_sizes.append(image.shape[:2][::-1])
            image_nchannels.append(image.shape[2])
        return image_sizes, image_nchannels
       
    @staticmethod
    def generate_reco_image(images_as_list:list, camera_tranforms_row:list, scaling_factor_of_images:float=1.0):
        """ Generates and returns the reconstructed image of the input sequence of images of this object. It also returns the ROI corners and sizes of the transformed versions of the input iamges

        :param images_as_list: A list of source images
        :type images_as_list: list
        :param camera_tranforms_row: a list of the camera transform parameters corresponding to the images
        :type camera_tranforms_row: list
        :param scaling_factor_of_images: the resize scale of the images compared to the source input images, defaults to 1.0
        :type scaling_factor_of_images: float, optional
        """
        if len(images_as_list) == 1:
            the_image = images_as_list[0]
            normalized_warped_roi_corners = [(0, 0,)]
            warped_roi_sizes = the_image[:2][::-1]
            return the_image, normalized_warped_roi_corners, warped_roi_sizes
        image_sizes, image_nchannels = ImageReconstruct1DModel.extract_image_sizes_nchannels(images_as_list)
        # create warper
        warper = Warper()
        warper.set_scale(camera_tranforms_row)
        # warp every image in the list at the scaling factor and the pre-computed camera transforms
        warped_images_list = list(warper.warp_images(images_as_list, camera_tranforms_row, scaling_factor_of_images))
        warped_roi_corners, normalized_roi_corners, warped_roi_sizes = ImageReconstruct1DModel._compute_warp_rois(image_sizes, camera_tranforms_row, scaling_factor_of_images)
        # logger = get_logger()
        # logger.info(f'warped image size from calculation: {warped_roi_sizes}')
        # logger.info(f'warped corners from calculation: {warped_roi_corners}')
        reco_image_origin_offset = ImageReconstruct1DModel._find_reco_image_origin_offset(warped_roi_corners)
        # logger.info(f'reco image origin offset: {reco_image_origin_offset}')
        normalized_warped_roi_corners = ImageReconstruct1DModel._normalize_corners(warped_roi_corners, reco_image_origin_offset)
        # logger.info(f'normalized warped corners from calculation: {warped_roi_corners}')
        # going through the warped images list to estimate the buffer size for the reconstructed image
        br_max_x, br_max_y = None, None
        for image, corner, image_size in zip(warped_images_list, normalized_warped_roi_corners, warped_roi_sizes):
            # compute the top left corner according to the scaling factor
            br_x, br_y = image_size[0] + corner[0], image_size[1] + corner[1]
            br_max_x = br_x if br_max_x is None else max(br_max_x, br_x)
            br_max_y = br_y if br_max_y is None else max(br_max_y, br_y)
        # record the size of the reconstructed image
        adj_roi_reco_size = (br_max_x, br_max_y)
        image_buffer_shape = (adj_roi_reco_size[1], adj_roi_reco_size[0], image_nchannels[0])
        image_buffer = None 
        index = 0
        # going through the warped images again and pasting them to the reconstructed image buffer
        for image, warped_corner in zip(warped_images_list, normalized_warped_roi_corners):
            image_size = image.shape[:2][::-1]
            if image_buffer is None:       
                image_buffer = np.zeros(image_buffer_shape, dtype=np.uint8)
                image_buffer[warped_corner[1]:warped_corner[1]+image_size[1], warped_corner[0]:warped_corner[0]+image_size[0], :] = image
            else:
                image_buffer_roi = image_buffer[warped_corner[1]:warped_corner[1]+image_size[1], warped_corner[0]:warped_corner[0]+image_size[0], :]
                image_buffer[warped_corner[1]:warped_corner[1]+image_size[1], warped_corner[0]:warped_corner[0]+image_size[0], :] = np.where((image_buffer_roi != [0, 0, 0]), image_buffer_roi, image)            
            # cv2.imwrite(f'/home/qcr/cgras_data/detector/row_step_{index}.jpg', image_buffer)
            index += 1
        return image_buffer, normalized_warped_roi_corners, warped_roi_sizes    
