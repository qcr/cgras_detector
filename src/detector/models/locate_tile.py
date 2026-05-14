# Copyright 2024 - Andrew Kwok Fai LUI, 
# Robotics and Autonomous Systems Group, REF, RI
# and the Queensland University of Technology

__author__ = 'Andrew Lui'
__copyright__ = 'Copyright 2024'
__license__ = 'GPL'
__version__ = '1.0'
__email__ = 'ak.lui@qut.edu.au'
__status__ = 'Development'

import os, math, yaml, numbers, random
from enum import Enum
import numpy as np
import cv2
import pickle

from detector.models.frame_detector import FrameDetector

from detector.models.imaging_tools import ImageMap, test_get_cgras_sample_images_as_list
from detector.models.reconstruct import ImageReconstructModel, ImageReconstructModelHelper

from detector.models import logger
from detector.models.models_config import ModelsConfigNames
from detector.models.detector_error import DetectorFailed, DetectorAborted, DetectorExceptionCodes

class WhichCorner(Enum):
    TOP_LEFT = 0
    TOP_RIGHT = 1
    BOTTOM_RIGHT = 2
    BOTTOM_LEFT = 3

class LocateTileModel():
    """ LocateTileModel uses computer vision means to detect the 4 corners of tile frames so to enable transformation from reconstructed image space to the tile space
    """
    # constants
    FILENAME_WHOLE_RECO_IMAGE = 'whole_reco_image.jpg'
    FILENAME_WHOLE_RECO_FULL_SCALE_IMAGE = 'whole_reco_image_original_scale.jpg'
    LOCTILE_WHOLE_RECO_IMAGE_FILENAME = 'loctile_whole_reco_image.jpg'
    ROTATED_WHOLE_RECO_IMAGE_FILENAME = 'rotated_whole_reco_image.jpg'
    ROTATED_WHOLE_RECO_FULL_SCALE_IMAGE_FILENAME = 'rotated_whole_reco_image_original_scale.jpg'

    
    def __init__(self, images_2d_list:list, map_location_fn, image_size_in_px:tuple, **kwargs):
        """ The constructor

        :param images_2d_list: The 2D grid of input source images as a list of lists  
        :type images_2d_list: list
        :param map_location_fn: A function that maps (image_x, image_y, x, y) to the whole reconstructed space (x, y)
        :type map_location_fn: function object
        :param image_size_in_px: The size of the whole reconstructed image
        :type image_size_in_px: 2-tuple        
        """

        # ignore the constructor if the object is loaded from yaml file
        if images_2d_list is None:
            return
        self.images_2d_list = images_2d_list
        self.params = kwargs
        self.map_location_fn = map_location_fn
        self.image_size_in_px = image_size_in_px
        # placement_grid
        self.placement_grid_dim = (len(self.images_2d_list[0]), len(self.images_2d_list))
        
        # optional input parameter (for fallback if a corner cannot be found) NOTE: fallback is not implemented
        # self.whole_tile_image_size = kwargs.get(ModelsConfigNames.WHOLE_TILE_IAMGE_SIZE.value, None)
        # self.tile_holder_width = kwargs.get(ModelsConfigNames.TILE_HOLDER_WIDTH.value, None)
        # if self.whole_tile_image_size:
        #     logger.info(f'LocateTileModel default whole tile image size: {self.whole_tile_image_size}')
        
        # the info coming from the tile information
        self.tile_size_in_mm = kwargs.get(ModelsConfigNames.TILE_SIZE_IN_MM.value, None)
        self.frame_size_in_mm = kwargs.get(ModelsConfigNames.FRAME_SIZE_IN_MM.value, None)
        # the frame size and the tile size in pixels are to be detected and computed 
        self.detected_frame_size_in_px = None
        self.detected_tile_size_in_px = None
        # other input parameters
        self.logdata_folder = kwargs.get('logdata_folder', None)
        self.write_debug_images = kwargs.get(ModelsConfigNames.LOCTILE_DEBUG_IMAGES.value, False)
        self.write_debug_images_original_scale = kwargs.get(ModelsConfigNames.RECO_DEGUG_IMAGE_ORIGINAL_SCALE.value, False)
        self.write_manual_validation_original_scale = kwargs.get(ModelsConfigNames.RECO_MANUAL_VALIDATION_ORIGINAL_SCALE.value, False)
        self.jpeg_quality = kwargs.get(ModelsConfigNames.OUTPUT_JPEG_QUALITY.value, 95)
        # self.blue_ratio_min = kwargs.get(ModelsConfigNames.LOCTILE_BLUE_RATIO_MIN.value, 0.35)
        # self.red_ratio_max = kwargs.get(ModelsConfigNames.LOCTILE_RED_RATIO_MAX.value, 0.15)
        self.working_scale = kwargs.get(ModelsConfigNames.LOCTILE_WORKING_SCALE.value, 0.1)
        # pixel classifier model filename
        self.pixel_classifier_model_filename = kwargs.get(ModelsConfigNames.LOCTILE_PIXEL_CLASSIFIER_FILENAME.value, 'train_set_1.model')
        # other input parameters: template matching of corner
        self.template_corner_size = kwargs.get(ModelsConfigNames.LOCTILE_TEMPLATE_CORNER_SIZE.value, 60)  # default 60 pixels
        self.template_size = kwargs.get(ModelsConfigNames.LOCTILE_TEMPLATE_SIZE.value, 120)  # default 120 pixels
        self.template_type = kwargs.get(ModelsConfigNames.LOCTILE_TEMPLATE_TYPE.value, 'edge')
        self.matching_score_min = kwargs.get(ModelsConfigNames.LOCTILE_MATCHING_SCORE_MIN.value, 0.5)
        self.rotate_angle_max = kwargs.get(ModelsConfigNames.LOCTILE_ROTATE_ANGLE_MAX.value, 3.0)
        # important model variables
        self.corners_in_reco_space = {}
        self.tile_offset_in_px = self.tile_size_in_px = None

    def build(self):
        logger.info(f'LocateTileModel: starting corner detection with parameters:')
        logger.info(f'  pixel_classifier_model: {self.pixel_classifier_model_filename}')
        logger.info(f'  working_scale: {self.working_scale}')
        logger.info(f'  template_size: {self.template_size} px')
        logger.info(f'  template_corner_size: {self.template_corner_size} px')
        logger.info(f'  matching_score_min: {self.matching_score_min}')
        logger.info(f'  rotate_angle_max: {self.rotate_angle_max} deg')
        logger.info(f'  debug_images: {self.write_debug_images}')
        single_row = len(self.images_2d_list) == 1
        single_col = len(self.images_2d_list[0]) == 1
        # search for the 4 corners 
        # assume that the corners are in the topleft, topright, bottomleft and bottomright images
        image_filepath = self.images_2d_list[0][0]
        image = cv2.imread(image_filepath)
        height, width, _ = image.shape
        if single_col or single_row:
            self.corners_in_reco_space[WhichCorner.TOP_LEFT] = self._locate_corner(image[:height//2, :width//2], WhichCorner.TOP_LEFT)
        else:
            self.corners_in_reco_space[WhichCorner.TOP_LEFT] = self._locate_corner(image, WhichCorner.TOP_LEFT)
        image_filepath = self.images_2d_list[0][-1]
        image = cv2.imread(image_filepath)
        if single_col or single_row:
            self.corners_in_reco_space[WhichCorner.TOP_RIGHT] = self._locate_corner(image[:height//2, width//2:], WhichCorner.TOP_RIGHT)
            self.corners_in_reco_space[WhichCorner.TOP_RIGHT] = (self.corners_in_reco_space[WhichCorner.TOP_RIGHT][0] + width // 2, self.corners_in_reco_space[WhichCorner.TOP_RIGHT][1])
        else:
            self.corners_in_reco_space[WhichCorner.TOP_RIGHT] = self._locate_corner(image, WhichCorner.TOP_RIGHT)
        image_filepath = self.images_2d_list[-1][0]
        image = cv2.imread(image_filepath)
        if single_row or single_col:
            self.corners_in_reco_space[WhichCorner.BOTTOM_LEFT] = self._locate_corner(image[height//2:, :width//2], WhichCorner.BOTTOM_LEFT)
            self.corners_in_reco_space[WhichCorner.BOTTOM_LEFT] = (self.corners_in_reco_space[WhichCorner.BOTTOM_LEFT][0], self.corners_in_reco_space[WhichCorner.BOTTOM_LEFT][1] + height // 2)
        else:
            self.corners_in_reco_space[WhichCorner.BOTTOM_LEFT] = self._locate_corner(image, WhichCorner.BOTTOM_LEFT)
        image_filepath = self.images_2d_list[-1][-1]
        image = cv2.imread(image_filepath)
        if single_row or single_col:
            self.corners_in_reco_space[WhichCorner.BOTTOM_RIGHT] = self._locate_corner(image[height//2:, width//2:], WhichCorner.BOTTOM_RIGHT)
            self.corners_in_reco_space[WhichCorner.BOTTOM_RIGHT] = (self.corners_in_reco_space[WhichCorner.BOTTOM_RIGHT][0] + width // 2, self.corners_in_reco_space[WhichCorner.BOTTOM_RIGHT][1] + height // 2)
        else:
            self.corners_in_reco_space[WhichCorner.BOTTOM_RIGHT] = self._locate_corner(image, WhichCorner.BOTTOM_RIGHT)
        # report error
        if self.corners_in_reco_space[WhichCorner.TOP_LEFT] is None or self.corners_in_reco_space[WhichCorner.TOP_RIGHT] is None or self.corners_in_reco_space[WhichCorner.BOTTOM_LEFT] is None or self.corners_in_reco_space[WhichCorner.BOTTOM_RIGHT] is None:
            raise DetectorFailed(DetectorExceptionCodes.LOC_FRAME_MISSING, f'Not all four corners are found')
        # map these corners (locations in image space) to the reconstructed image space
        self.corners_in_reco_space[WhichCorner.TOP_LEFT] = self.map_location_fn(0, 0, self.corners_in_reco_space[WhichCorner.TOP_LEFT])
        self.corners_in_reco_space[WhichCorner.TOP_RIGHT] = self.map_location_fn(self.placement_grid_dim[0]-1, 0, self.corners_in_reco_space[WhichCorner.TOP_RIGHT])
        self.corners_in_reco_space[WhichCorner.BOTTOM_LEFT] = self.map_location_fn(0, self.placement_grid_dim[1]-1, self.corners_in_reco_space[WhichCorner.BOTTOM_LEFT])
        self.corners_in_reco_space[WhichCorner.BOTTOM_RIGHT] = self.map_location_fn(self.placement_grid_dim[0]-1, self.placement_grid_dim[1]-1, self.corners_in_reco_space[WhichCorner.BOTTOM_RIGHT])
        logger.info(f'LocateTileModel tile corners: {self.corners_in_reco_space}')

        # write annotated whole image if found in the logfile folder
        self._write_annotate_whole_image()

        # estimate the whole tile image size if not found in the input parameters
        detected_frame_size_x = int(math.dist(self.corners_in_reco_space[WhichCorner.TOP_LEFT], self.corners_in_reco_space[WhichCorner.TOP_RIGHT]))
        detected_frame_size_y = int(math.dist(self.corners_in_reco_space[WhichCorner.TOP_LEFT], self.corners_in_reco_space[WhichCorner.BOTTOM_LEFT]))
        self.detected_frame_size_in_px = (detected_frame_size_x, detected_frame_size_y,)
        logger.info(f'LocateTileModel estimated detected_frame_size: {self.detected_frame_size_in_px}')
        
        # compute affine transformation (this is obsolete, only rotation is assumed)
        # self.affine_transform_matrix = cv2.getAffineTransform(input_pts, output_pts)
        # compute the origin (this is obsolete because the transformed/rotated corners do not give good estimation of the original)
        # self.origin_offset = (self.detected_frame_size_in_px[0] // 2 + self.corners_in_reco_space[WhichCorner.TOP_LEFT][0], self.detected_frame_size_in_px[1] // 2 + self.corners_in_reco_space[WhichCorner.TOP_LEFT][1])
        # use the reco image as the reference to determin the origin (assumed only rotation occured)
        self.image_origin_offset = (self.image_size_in_px[0] // 2, self.image_size_in_px[1] // 2)
        
        # the affirm transform matrix is defined in the space of the tile holder
        # the reconstructed image size is used as the reference of the origin, so if the affine transform matrix is used to operate in the frame/tile space, offset the result is required
        self.affine_transform_matrix, rotate_angle = self._compute_affine_transform_only_rotation(self.corners_in_reco_space, self.image_origin_offset)
        logger.info(f'AffineTransform matrix: {self.affine_transform_matrix} rotation origin offset {self.image_origin_offset} rotation angle {rotate_angle} (degrees)')
        if abs(rotate_angle) > self.rotate_angle_max:
            raise DetectorFailed(DetectorExceptionCodes.LOC_FAILED, f'Angle of rotation ({rotate_angle:.1f} degrees) outside of the valid range (+/- {self.rotate_angle_max} degrees)')

        frame_offset = self._apply_affine_transform(self.corners_in_reco_space[WhichCorner.TOP_LEFT], adjust_offset=False) 
        frame_topright_corrected = self._apply_affine_transform(self.corners_in_reco_space[WhichCorner.TOP_RIGHT], adjust_offset=False) 
        frame_bottomleft_corrected = self._apply_affine_transform(self.corners_in_reco_space[WhichCorner.BOTTOM_LEFT], adjust_offset=False) 
        frame_bottomright_corrected = self._apply_affine_transform(self.corners_in_reco_space[WhichCorner.BOTTOM_RIGHT], adjust_offset=False) 
        # update the detected frame_size in px
        # detected_frame_size_in_px = (frame_topright_corrected[0] - frame_offset[0], frame_bottomleft_corrected[1] - frame_offset[1],)
        self.detected_frame_size_in_px = (frame_topright_corrected[0] - frame_offset[0], frame_bottomleft_corrected[1] - frame_offset[1],)        
        # test if the ratio of the length and width is not similar to the given frame dimension
        detected_frame_size_ratio = self.detected_frame_size_in_px[0] / self.detected_frame_size_in_px[1]
        frame_size_in_mm_ratio = self.frame_size_in_mm[0] / self.frame_size_in_mm[1]  
        rel_tol = self.params.get(ModelsConfigNames.LOCTILE_ASPECT_RATIO_DIFF_MAX_REL.value, 0.1)
        abs_tol = self.params.get(ModelsConfigNames.LOCTILE_ASPECT_RATIO_DIFF_MAX_ABS.value, 0.1)
        if not math.isclose(detected_frame_size_ratio, frame_size_in_mm_ratio, rel_tol=rel_tol, abs_tol=abs_tol):
            raise DetectorFailed(DetectorExceptionCodes.LOC_FAILED, f'The aspect ratio of the detected frame ({self.detected_frame_size_in_px} {detected_frame_size_ratio:.2f}) is deviated from that of the frame size ({frame_size_in_mm_ratio})')

        # compute the detected tile size in pixels from the tile size and frame size
        # approx_pixel_per_mm_x = detected_frame_size_x / self.frame_size_in_mm[0]
        # approx_pixel_per_mm_y = detected_frame_size_y / self.frame_size_in_mm[1]  
        approx_pixel_per_mm_x = self.detected_frame_size_in_px[0] / self.frame_size_in_mm[0]
        approx_pixel_per_mm_y = self.detected_frame_size_in_px[1] / self.frame_size_in_mm[1]                
        approx_pixel_per_mm = (approx_pixel_per_mm_x + approx_pixel_per_mm_y) / 2
        
        logger.info(f'LocateTile frame size in mm and detected in px: {self.frame_size_in_mm} {detected_frame_size_x, detected_frame_size_y}')
        logger.info(f'LocateTile approx_pixel_per_mm: {approx_pixel_per_mm_x:.2f} {approx_pixel_per_mm_y:.2f} (frame and tile size in mm: {self.frame_size_in_mm} {self.tile_size_in_mm})')
        logger.info(f'LocateTile frame_offset: {frame_offset}')
        logger.info(f'LocateTile frame_size: {self.detected_frame_size_in_px} (pixel per mm: {approx_pixel_per_mm_x:.2f} {approx_pixel_per_mm_y:.2f})')
        
        # compute tile offset, assuming that the frame holder width is the same on all sides
        holder_width_in_pixel = ((self.frame_size_in_mm[0] - self.tile_size_in_mm[0]) / 2) * approx_pixel_per_mm_x, ((self.frame_size_in_mm[1] - self.tile_size_in_mm[1]) / 2) * approx_pixel_per_mm_y
        # NOTE: for testing - override by the param 'test_only_holder_width_in_px' 
        if 'test_only_holder_width_in_px' in self.params:
            holder_width_in_pixel = self.params['test_only_holder_width_in_px']  # (x, y)
        # compute the tile offset using the frame offset and the holder width
        self.tile_offset_in_px = (frame_offset[0] + holder_width_in_pixel[0], frame_offset[1] + holder_width_in_pixel[1],)
        self.tile_size_in_px = (int(self.detected_frame_size_in_px[0] - 2 * holder_width_in_pixel[0]), int(self.detected_frame_size_in_px[1] - 2 * holder_width_in_pixel[1], ))
        logger.info(f'LocateTile tile_offset: {self.tile_offset_in_px}')
        logger.info(f'LocateTile tile_size in pixels: {self.tile_size_in_px} (holder_width in pixels: {holder_width_in_pixel})')   

        # write rotated whole image if found in the logfile folder
        self._write_rotated_whole_image() 
        # write rotated whole image original scale if found in the logfile folder
        self._write_rotated_whole_image_original_scale()
        
    def _locate_corner(self, original_image:np.ndarray, which_corner:WhichCorner, fine_search:bool=False):
        # image = self._apply_tile_filter(original_image)
        image = self._apply_tile_filter_classifier(original_image)
        
        # find approximate corner at working_scale
        down_width, down_height = int(image.shape[1] * self.working_scale), int(image.shape[0] * self.working_scale)
        image_downscaled = cv2.resize(image, (down_width, down_height,))
        # generate template
        if self.template_type == 'block':
            template, corner_offset = self._generate_template_block(which_corner, self.template_size)
        else:
            template, corner_offset = self._generate_template(which_corner, self.template_corner_size, self.template_size)

        res = cv2.matchTemplate(image_downscaled, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        
        # TEMPORARY: write the tile filter output for debugging
        debug_image_filepath = os.path.join(self.logdata_folder, f'classify_tile_{which_corner.name}.jpg')
        image_tile_classify = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(debug_image_filepath, image_tile_classify, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        # TEMPORARY: write the corner template output for debugging -
        debug_image_filepath = os.path.join(self.logdata_folder, f'template_{which_corner.name}.jpg')
        template = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(debug_image_filepath, template, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        # TEMPORARY: write the matchTemplate response for debugging
        debug_image_filepath = os.path.join(self.logdata_folder, f'match_res_{which_corner.name}.jpg')
        res_norm = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        cv2.imwrite(debug_image_filepath, res_norm, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        
        logger.info(f'self.write_debug_images: {self.write_debug_images}, self.logdata_folder: {self.logdata_folder}')
        logger.info(f'template max_val: {max_val} at location {max_loc} in the downscaled image for corner {which_corner.name}')
        if max_val < self.matching_score_min:
            return None
        # compute the corner at working scale
        top_left = max_loc
        # bottom_right = (max_loc[0] + self.template_size, max_loc[1] + self.template_size)
        # calciulate corner and scale up to the original resolution
        # corner = (int((top_left[0] + bottom_right[0]) / self.working_scale / 2), int((top_left[1] + bottom_right[1]) / self.working_scale / 2))
        corner = (int((top_left[0] + corner_offset[0]) / self.working_scale), int((top_left[1] + corner_offset[1]) / self.working_scale))

        if fine_search:
            # extract a local search region for fine tuning
            search_rectangle_size = int(2 / self.working_scale) 
            search_bbox = (corner[0] - search_rectangle_size,  corner[1] - search_rectangle_size, corner[0] + search_rectangle_size, corner[1] + search_rectangle_size)
            image_extracted = image[search_bbox[1]:search_bbox[3], search_bbox[0]:search_bbox[2]]
            res = cv2.matchTemplate(image_extracted, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            # if nothing is found, use the low resolution result as a fallback
            if max_val < self.matching_score_min:
                ...
            else:
                corner = (int(search_bbox[0] + max_loc[0] + self.template_corner_size / 2), int(search_bbox[1] + max_loc[1] + self.template_corner_size / 2))

        # write annotated image related to corner detection
        if self.write_debug_images and self.logdata_folder is not None:
            try:
                # write corner detect
                debug_image_filepath = os.path.join(self.logdata_folder, f'locate_corner_{which_corner.name}.jpg')
                image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                point_1 = (int(corner[0] - self.template_corner_size / self.working_scale), int(corner[1] - self.template_corner_size / self.working_scale))
                point_2 = (int(corner[0] + self.template_corner_size / self.working_scale), int(corner[1] + self.template_corner_size / self.working_scale))
                image_bgr = cv2.rectangle(image_bgr, point_1, point_2, (0, 0, 255,), 10)
                cv2.imwrite(debug_image_filepath, image_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                # write template
                # debug_image_filepath = os.path.join(self.logdata_folder, f'template_{which_corner.name}.jpg')
                # template = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
                # cv2.imwrite(debug_image_filepath, template, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            except:
                raise DetectorAborted(DetectorExceptionCodes.OS_ERROR, f'Failed to write feature matching output to {debug_image_filepath}')
        return corner
    
    @classmethod
    def _compute_affine_transform_only_rotation(cls, corners:dict, origin_offset:tuple):
        def _estimate_rotated_angle(P1, P2, P3):
            # the angle with P1 as the vertex, P1 - P2 and P1 - P3
            # P2 is the target position so that the tile is aligned
            # return the angle required to restore the tile back to aligned
            result = math.atan2(P3[1] - P1[1], P3[0] - P1[0]) - math.atan2(P2[1] - P1[1], P2[0] - P1[0])
            return result
        # consider only rotation
        angle_sample_1 = _estimate_rotated_angle(corners[WhichCorner.TOP_LEFT], (corners[WhichCorner.TOP_RIGHT][0], corners[WhichCorner.TOP_LEFT][1]), corners[WhichCorner.TOP_RIGHT])       
        angle_sample_2 = _estimate_rotated_angle(corners[WhichCorner.TOP_LEFT], (corners[WhichCorner.TOP_LEFT][0], corners[WhichCorner.BOTTOM_LEFT][1]), corners[WhichCorner.BOTTOM_LEFT])   
        rotate_angle = (angle_sample_1 + angle_sample_2) / 2
        rotate_angle = np.degrees(rotate_angle)
        logger.info(f'_estimate_affine_transform angle for correction: {angle_sample_1:3f} {angle_sample_2:3f} degrees {rotate_angle:2f} origin_offset {origin_offset}')
        affine_transform_matrix = cv2.getRotationMatrix2D(origin_offset, rotate_angle, 1.0)
        # test the affine transform matrix
        result:np.ndarray = np.matmul(affine_transform_matrix, np.float32([origin_offset[0], origin_offset[1], 1]).T).astype(np.int32)
        logger.info(f'_estimate_affine_transform test (origin {origin_offset}): {result}')
        return affine_transform_matrix, rotate_angle

    @classmethod
    def _compute_affine_transform(cls, corners:dict, origin_offset:tuple):    
        ...
        # compute the transformation matrix using cv2.getAffineTransform()
        # work out the affine transform if the whole tile size (in pixels) is given
        # input_pts = np.float32([self.corners[WhichCorner.TOP_LEFT], self.corners[WhichCorner.TOP_RIGHT], self.corners[WhichCorner.BOTTOM_RIGHT]])
        # output_pts = np.float32([self.corners[WhichCorner.TOP_LEFT], [self.corners[WhichCorner.TOP_LEFT][0] + self.whole_tile_image_size[0], self.corners[WhichCorner.TOP_LEFT][1]], \
        #                             [self.corners[WhichCorner.TOP_LEFT][0] + self.whole_tile_image_size[0], self.corners[WhichCorner.TOP_LEFT][1] + self.whole_tile_image_size[1]]])

        # input_pts = np.float32([self.corners[WhichCorner.TOP_LEFT], self.corners[WhichCorner.TOP_RIGHT], self.corners[WhichCorner.BOTTOM_LEFT]])
        # output_pts = np.float32([self.corners[WhichCorner.TOP_LEFT], [self.corners[WhichCorner.TOP_LEFT][0] + self.whole_tile_image_size[0], self.corners[WhichCorner.TOP_LEFT][1]], \
        #                             [self.corners[WhichCorner.TOP_LEFT][0], self.corners[WhichCorner.TOP_LEFT][1] + self.whole_tile_image_size[1]]])       
        # logger.info(f'input_pts: {input_pts}')
        # logger.info(f'output_pts: {output_pts}')

    def _apply_affine_transform(self, point_in_image_space:tuple, adjust_offset:bool=False):
        if self.affine_transform_matrix is not None:
            # point = (point[0] - self.origin_offset[0], point[1] - self.origin_offset[1])
            transformed:np.ndarray = np.matmul(self.affine_transform_matrix, np.float32([point_in_image_space[0], point_in_image_space[1], 1]).T).astype(np.int32)
            if adjust_offset:
                # the transformed location has to be adjusted by the offset distance because the affine transformation is computed using the origin of the image 
                offset:np.ndarray = np.matmul(self.affine_transform_matrix, np.float32([0, 0, 1]).T).astype(np.int32)
                # to obtain the transformed location in the tile holder space, the offset is needed
                result = tuple((transformed - offset).tolist())     
                return result
            else:
                return tuple(transformed.tolist())
        return point_in_image_space
        
    def _write_annotate_whole_image(self):
        if self.write_debug_images and self.logdata_folder is not None:
            reco_working_scale = self.params.get(ModelsConfigNames.RECO_WORKING_SCALE.value, None)
            # if the reco_working_scale is not known, the scale of the whole_reco_image.jpg is also no known and annotation cannot be done
            if reco_working_scale is None:
                return
            # attempt to load the whole_reco_image
            reco_whole_image_filepath = os.path.join(self.logdata_folder, LocateTileModel.FILENAME_WHOLE_RECO_IMAGE)
            reco_whole_image = cv2.imread(reco_whole_image_filepath)
            if reco_whole_image is None:
                return
            # specify the polygon using the corners
            pts = np.array([self.corners_in_reco_space[WhichCorner.TOP_LEFT],
                            self.corners_in_reco_space[WhichCorner.TOP_RIGHT],
                            self.corners_in_reco_space[WhichCorner.BOTTOM_RIGHT],
                            self.corners_in_reco_space[WhichCorner.BOTTOM_LEFT],], np.float32)
            pts = np.multiply(pts, reco_working_scale).astype(np.int32)
            annotated_reco_whole_image = cv2.polylines(reco_whole_image, [pts], True, [0, 0, 255], 2)
            if self.write_debug_images and self.logdata_folder is not None:
                try:
                    annotated_reco_whole_image_filepath = os.path.join(self.logdata_folder, LocateTileModel.LOCTILE_WHOLE_RECO_IMAGE_FILENAME)
                    cv2.imwrite(annotated_reco_whole_image_filepath, annotated_reco_whole_image, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                except:
                    raise DetectorAborted(DetectorExceptionCodes.OS_ERROR, f'Failed to write whole reco image with tile bounds output to {annotated_reco_whole_image_filepath}')            

    def _write_rotated_whole_image(self):
        if self.write_debug_images and self.logdata_folder is not None:
            reco_working_scale = self.params.get(ModelsConfigNames.RECO_WORKING_SCALE.value, None)
            # attempt to load the whole_reco_image
            reco_whole_image_filepath = os.path.join(self.logdata_folder, LocateTileModel.FILENAME_WHOLE_RECO_IMAGE)
            reco_whole_image = cv2.imread(reco_whole_image_filepath)
            if reco_whole_image is None:
                return    
            # compute the affine tranform matrix based on the detected corners
            affine_transform_matrix, rotate_angle = self._compute_affine_transform_only_rotation(self.corners_in_reco_space, (reco_whole_image.shape[0] // 2, reco_whole_image.shape[1] // 2,))
            rotated_whole_image = cv2.warpAffine(reco_whole_image, affine_transform_matrix, (int(reco_whole_image.shape[1] * 1.1), int(reco_whole_image.shape[0] * 1.1)))
            # draw frame bounds
            pts = np.array([self._apply_affine_transform(self.corners_in_reco_space[WhichCorner.TOP_LEFT], adjust_offset=False),
                            self._apply_affine_transform(self.corners_in_reco_space[WhichCorner.TOP_RIGHT], adjust_offset=False),
                            self._apply_affine_transform(self.corners_in_reco_space[WhichCorner.BOTTOM_RIGHT], adjust_offset=False),
                            self._apply_affine_transform(self.corners_in_reco_space[WhichCorner.BOTTOM_LEFT], adjust_offset=False),
            ], np.float32)
            pts = np.multiply(pts, reco_working_scale).astype(np.int32)
            annotated_rotated_whole_image = cv2.polylines(rotated_whole_image, [pts], True, [0, 0, 255], 2) 
            # draw tile bounds
            tile_point_1 = (int(self.tile_offset_in_px[0] * reco_working_scale), int(self.tile_offset_in_px[1] * reco_working_scale))
            tile_point_2 = (int(tile_point_1[0] + self.tile_size_in_px[0] * reco_working_scale), int(tile_point_1[1] + self.tile_size_in_px[1] * reco_working_scale))
            cv2.rectangle(annotated_rotated_whole_image, tile_point_1, tile_point_2, 
                          (32, 255, 32,), 1)
            try:
                rotated_reco_whole_image_filepath = os.path.join(self.logdata_folder, LocateTileModel.ROTATED_WHOLE_RECO_IMAGE_FILENAME)
                cv2.imwrite(rotated_reco_whole_image_filepath, annotated_rotated_whole_image, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            except:
                raise DetectorAborted(DetectorExceptionCodes.OS_ERROR, f'Failed to write rotated whole reco image with tile bounds output to {rotated_reco_whole_image_filepath}')
            return rotated_reco_whole_image_filepath
        return None
    
    def _write_rotated_whole_image_original_scale(self):
        if self.logdata_folder is not None and (self.write_debug_images_original_scale or self.write_manual_validation_original_scale):
            # attempt to load the whole_reco_image
            reco_whole_image_filepath = os.path.join(self.logdata_folder, LocateTileModel.FILENAME_WHOLE_RECO_FULL_SCALE_IMAGE)
            reco_whole_image = cv2.imread(reco_whole_image_filepath)
            if reco_whole_image is None:
                return
            # compute the affine tranform matrix based on the detected corners
            affine_transform_matrix, rotate_angle = self._compute_affine_transform_only_rotation(self.corners_in_reco_space, (reco_whole_image.shape[0] // 2, reco_whole_image.shape[1] // 2,))
            rotated_whole_image = cv2.warpAffine(reco_whole_image, affine_transform_matrix, (int(reco_whole_image.shape[1] * 1.1), int(reco_whole_image.shape[0] * 1.1)))
            rotated_reco_whole_image_filepath = os.path.join(self.logdata_folder, LocateTileModel.ROTATED_WHOLE_RECO_FULL_SCALE_IMAGE_FILENAME)
            cv2.imwrite(rotated_reco_whole_image_filepath, rotated_whole_image, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            return rotated_reco_whole_image_filepath
        return None    

    def _adjust_gamma(self, image, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype('uint8')
        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)

    @classmethod
    def _generate_template_block(cls, type: WhichCorner, template_size: int):
        """50/50 split template — matches the large blue corner blocks used in 2024 data."""
        half = template_size // 2
        blank_image = np.full((template_size, template_size), 0, dtype=np.uint8)
        if type == WhichCorner.TOP_LEFT:
            blank_image[half:, half:] = 255
        elif type == WhichCorner.TOP_RIGHT:
            # blank_image[:half, half:] = 255      
            blank_image[half:, :half] = 255
        elif type == WhichCorner.BOTTOM_RIGHT:
            blank_image[:half, :half] = 255
        elif type == WhichCorner.BOTTOM_LEFT:
            # blank_image[half:, :half] = 255
            blank_image[:half, half:] = 255
        corner_offset = (half, half)
        return blank_image, corner_offset

    @classmethod
    def _generate_template(cls, type:WhichCorner, template_corner_size:int, template_size:int):
        
        blank_image = np.full((template_size, template_size), 255, dtype=np.uint8)
        if type == WhichCorner.TOP_LEFT:
            blank_image[int(template_corner_size):, int(template_corner_size):] = 0
            corner_offset = (template_corner_size, template_corner_size,)
        elif type == WhichCorner.TOP_RIGHT:
            blank_image[int(template_corner_size):, :int(template_size - template_corner_size)] = 0
            corner_offset = (template_size - template_corner_size, template_corner_size,)
        elif type == WhichCorner.BOTTOM_RIGHT:
            blank_image[:int(template_size - template_corner_size), :int(template_size - template_corner_size)] = 0
            corner_offset = (template_size - template_corner_size, template_size - template_corner_size,)
        elif type == WhichCorner.BOTTOM_LEFT:
            blank_image[:int(template_size - template_corner_size), int(template_corner_size):] = 0
            corner_offset = (template_corner_size, template_size - template_corner_size,)

        return blank_image, corner_offset

    def _evaluate_image(self, image:np.ndarray):
        for c in range(3):
            image_ratio = np.divide(image[ :, :, c], image.sum(axis=2))
            print(image_ratio.shape)
            print(np.mean(image_ratio[:500, 4000:]))
            print(np.mean(image_ratio[500:, 4000:]))

    # def _apply_tile_filter(self, image:np.ndarray):
    #     pixel_sum_image = image.sum(axis=2)
    #     blue_ratio_met = np.divide(image[ :, :, 0], pixel_sum_image) >= self.blue_ratio_min
    #     red_ratio_met = np.divide(image[ :, :, 2], pixel_sum_image) <= self.red_ratio_max
    #     ratio_met = np.logical_and(blue_ratio_met, red_ratio_met)
    #     image = np.where(ratio_met, 255, 0).astype(np.uint8)
    #     return image
    
    def _apply_tile_filter_classifier(self, image:np.ndarray):
        frame_detector = FrameDetector(os.path.join(os.path.dirname(__file__), f'tile_filter/{self.pixel_classifier_model_filename}'))
        return frame_detector.classify_image(image)
    
    def map_and_normalize_bbox(self, bbox:tuple):
        """ converts a bounding box in the reconstructed image space to the tile space according to a detection of the frame or holder of the tile and then normalize the bbox to range [0, 1], 
        returns both bbox representations

        :param bbox: The bounding box (x1, y1, x2, y2) specified as a 4-tuple or 4-list
        :type bbox: A 4-list or 4-tuple      
        """   
        tile_bbox = self.map_bbox(bbox)
        return (tile_bbox, self.normalize_bbox(tile_bbox),)
    
    def map_bbox(self, bbox:tuple):
        """ converts a bounding box in the reconstructed image space to the tile space according to a detection of the frame or holder of the tile

        :param bbox: The bounding box (x1, y1, x2, y2) specified as a 4-tuple or 4-list
        :type bbox: A 4-list or 4-tuple      
        """   
        points = [(bbox[0], bbox[1],), (bbox[2], bbox[3],)]
        results = self.map_locations(points)
        return [*results[0], *results[1]]
     
    def map_locations(self, points:list) -> list:
        """ converts one or more points in the reconstructed image space to the tile space according to a detection of the frame or holder of the tile

        :param points: The parameter can be a point (a tuple of two floats indicating the (x, y) position) or a list of points
        :type points: A list of tuples or a tuple
        """        
        single_point = False
        if type(points) in (tuple, list) and len(points) == 2 and isinstance(points[0], numbers.Number):
            single_point = True
            points = [points]
        # iterate through each point in an original image and compute their location in the whole reconstructed image
        results = []
        for point in points:
            # apply affine transformation
            point_corrected_space = self._apply_affine_transform(point, adjust_offset=False) 
            point_in_tile_space = (point_corrected_space[0] - self.tile_offset_in_px[0], point_corrected_space[1] - self.tile_offset_in_px[1],)
            if single_point:
                return point_in_tile_space
            results.append(point_in_tile_space)
        return results

    def normalize_bbox(self, bbox:tuple):
        """ normalize a bounding box in the tile space in the range [0, 1] 

        :param bbox: The bounding box (x1, y1, x2, y2) specified as a 4-tuple or 4-list
        :type bbox: A 4-list or 4-tuple      
        """   
        points = [(bbox[0], bbox[1],), (bbox[2], bbox[3],)]
        results = self.normalize_locations(points)
        return [*results[0], *results[1]]
    
    def normalize_locations(self, mapped_points:list) -> list:
        """ normalize one or more mapped points in the tile space in the range [0, 1] 

        :param mapped_points: The parameter can be a point (a tuple of two floats indicating the (x, y) position) or a list of points in the tile space
        :type mapped_points: A list of tuples or a tuple of normalized locations
        """        
        single_point = False
        if type(mapped_points) in (tuple, list) and len(mapped_points) == 2 and isinstance(mapped_points[0], numbers.Number):
            single_point = True
            mapped_points = [mapped_points]
        # iterate through each point in the tile space and normalize by dividing by the size of the tile
        results = []
        for point in mapped_points:
            mapped_point = (point[0] / self.tile_size_in_px[0], point[1] / self.tile_size_in_px[1],)
            if single_point:
                return mapped_point
            results.append(mapped_point)
        return results
    
    def get_tile_size_in_image_space(self) -> tuple:
        """ returns the size (xdim, ydim) of the tile as a rectangle

        :return: _description_
        :rtype: tuple
        """
        return self.tile_size_in_px
    
    def get_tile_origin_in_image_space(self) -> tuple:
        """ returns the origin (x, y) of the tile in the image space

        :return: _description_
        :rtype: tuple
        """
        return self.tile_offset_in_px
    
    def get_corners_roi(self) -> list:
        """ returns the region of the tile frame specified by its corners from top-right clockwise 

        :return: a list of 4 tuples, each of which is a corner of the tile frame
        :rtype: list
        """
        roi_points = [
            self.corners_in_reco_space[WhichCorner.TOP_LEFT],
            self.corners_in_reco_space[WhichCorner.TOP_RIGHT],
            self.corners_in_reco_space[WhichCorner.BOTTOM_RIGHT],
            self.corners_in_reco_space[WhichCorner.BOTTOM_LEFT],
        ]
        return roi_points
    
    def print_info(self):
        """ prints the key parameters of the LocateTileModel object
        """
        logger.info(f'Tile Offset: {self.tile_offset_in_px}\nTile Size: {self.tile_size_in_px}')
        

class LocateTileModelHelper():
    """ LocateTileModelHelper provides helper functions for saving and loading an object of LocateFileModel to the file system

    """
    @staticmethod            
    def to_yaml(loctile_model:LocateTileModel, object_file:str = None) -> str:
        """ Save the model parameters of a LocateTileModel object as a yaml file to the path specified by the given object_file

        :param loctile_model: The object of LocateTileModel to be saved to a yaml file
        :type loctile_model: LocateTileModel
        :param object_file: The path where the yaml file is saved to, defaults to None
        :type object_file: str, optional
        :return: The string content of the yaml file
        :rtype: str
        """
        affine_transform_matrix = loctile_model.affine_transform_matrix.tolist()
        object_dict = {
            'tile_offset_in_px': loctile_model.tile_offset_in_px,
            'tile_size_in_px': loctile_model.tile_size_in_px,
            'affine_transform_matrix': affine_transform_matrix,
            'frame_size_in_mm': loctile_model.frame_size_in_mm,
            'tile_size_in_mm': loctile_model.tile_size_in_mm,
            'image_size_in_px': loctile_model.image_size_in_px,
            'image_origin_offset': loctile_model.image_origin_offset,
                       }
        try:
            rotation = np.degrees(math.acos(affine_transform_matrix[0][0]))
            object_dict['rotation_in_deg'] = float(rotation)
        except:
            ...
        if object_file is None:
            return yaml.dump(object_dict, Dumper=yaml.Dumper)
        else:
            with open(object_file, 'w') as outfile:
                yamlstr = yaml.dump(object_dict, outfile, Dumper=yaml.Dumper)
            return yamlstr
        
    @staticmethod
    def from_yaml_file(object_file:str) -> LocateTileModel:
        """ Create an LocateTileModel object from a yaml file

        :param object_file: the path to the yaml file 
        :type object_file: str
        :return: An ImageReconstructModel object created from the yaml file
        :rtype: ImageReconstructModel
        """
        with open(object_file, 'r') as infile:
            data = yaml.load(infile, Loader=yaml.Loader)
        return LocateTileModelHelper._create_model(data)
    
    @staticmethod
    def from_yaml(yaml_str:str) -> LocateTileModel:
        """ Create an LocateTileModel object from a yaml string

        :param yaml_str: the yaml string
        :type yaml_str: str
        :return: An ImageReconstructModel object created from the yaml string
        :rtype: ImageReconstructModel
        """        
        data = yaml.load(yaml_str, Loader=yaml.Loader)
        return LocateTileModelHelper._create_model(data)
    
    @staticmethod
    def _create_model(data:dict) -> LocateTileModel:
        """ internal function for creating a LocateTileModel 

        :param data: The parameters of a LocateTileModel as a dict, which comes from a yaml file or string
        :type data: dict
        :return: An ImageReconstructModel object
        :rtype: LocateTileModel
        """
        loctile_model = LocateTileModel(None, None, None)
        loctile_model.tile_offset_in_px = data['tile_offset_in_px']
        loctile_model.tile_size_in_px = data['tile_size_in_px']
        loctile_model.frame_size_in_mm = data['frame_size_in_mm']
        loctile_model.tile_size_in_mm = data['tile_size_in_mm']
        loctile_model.image_size_in_px = data['image_size_in_px']
        loctile_model.affine_transform_matrix = np.asarray(data['affine_transform_matrix'])
        loctile_model.image_origin_offset = data['image_origin_offset']
        return loctile_model
    
# ----------------------------------------------------------------------------------
# Test functions

def test_build_model(params, reco_model:ImageReconstructModel):
    image_map_as_list = test_get_cgras_sample_images_as_list()    
    logdata_folder = params['logdata_folder']
    loctile_model_file = os.path.join(logdata_folder, params['loctile_model_filename'])
    os.makedirs(logdata_folder, exist_ok=True)
    loctile_model = LocateTileModel(image_map_as_list, map_location_fn=reco_model.map_locations, image_size_in_px=reco_model.get_whole_reco_image_size(), **params) 
    logger.info('Saving LocateTileModel model to file')
    LocateTileModelHelper.to_yaml(loctile_model, loctile_model_file)
    return loctile_model

def test_load_model(params):
    logger.info('Loading LocateTileModel model from file')
    logdata_folder = params['logdata_folder']
    loctile_model_file = os.path.join(logdata_folder, params['loctile_model_filename'])
    loctile_model:LocateTileModel = LocateTileModelHelper.from_yaml_file(loctile_model_file)
    loctile_model.print_info()    
    return loctile_model

if __name__ == '__main__':
    logdata_folder = '/home/qcr/cgras_data/detector/data/2024-Nov/2024-Nov-P00001-CG1-202411151200/'
    params = {
        'logdata_folder': logdata_folder, 
        'reco_model_filename': 'reco_model.yaml',
        'loctile_model_filename': 'loctile_model.yaml',
    }
    # reco_model_file = '/home/qcr/cgras_data/detector/reconstruct/reco_model.yaml'
    # reco_model = test_load_reco_model(params)
    # loctile_model_file = '/home/qcr/cgras_data/detector/reconstruct/loctile_model.yaml'
    # loctile_model = test_build_model(params, reco_model)
    # loctile_model = test_load_model(params)
