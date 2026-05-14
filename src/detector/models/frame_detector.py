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
import numpy as np
import cv2

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

class FrameDetector():
    def __init__(self, model_file:str=None):
        # training input spec
        self.train_spec_file = None
        self.train_spec = None
        # intermediate data structures
        self.px_in_roi_frame = None
        self.px_in_roi_tile = None
        self.data_X = None
        self.data_y = None
        # classification model 
        if model_file is not None:
            model_file = self.load_classifier(model_file)
        else:
            self.classifier = None
        
    def train(self, train_spec_file:str):
        self.train_spec_file = train_spec_file
        try:
            with open(train_spec_file, 'r') as infile:
                self.train_spec = yaml.load(infile, Loader=yaml.Loader)
        except:
            raise OSError(f'Failed to load yaml file {self.train_spec_file}')
        # validate the train data spec
        if type(self.train_spec) is not dict or 'train_set' not in self.train_spec:
            raise AssertionError(f'The training data yaml file is malformed')
        # build training dataset
        for image_spec in self.train_spec['train_set']:
            print(image_spec)
            # load the image
            image_bgr = cv2.imread(image_spec['image_file'])
            image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
            # extract the pixel values of the frame class
            self.px_in_roi_frame = None
            for roi in image_spec['frame']:
                px_in_roi = self._extract_px_in_roi(image_hsv, *roi)
                self.px_in_roi_frame = px_in_roi if self.px_in_roi_frame is None else np.vstack((self.px_in_roi_frame, px_in_roi))
            # extract the pixel values of the tile class
            self.px_in_roi_tile = None
            for roi in image_spec['tile']:
                px_in_roi = self._extract_px_in_roi(image_hsv, *roi)
                self.px_in_roi_tile = px_in_roi if self.px_in_roi_tile is None else np.vstack((self.px_in_roi_tile, px_in_roi))     
        # build input parameter set
        self.data_X = np.vstack((self.px_in_roi_frame, self.px_in_roi_tile))
        self.data_y = np.hstack((np.full(self.px_in_roi_frame.shape[0], 1), np.full(self.px_in_roi_tile.shape[0], 0)))   # class 0 is tile and class 1 is frame
        
        # train classification model
        X_train, X_test, y_train, y_test = train_test_split(self.data_X, self.data_y, test_size=0.20)
        classifier = DecisionTreeClassifier()
        classifier.fit(X_train, y_train)
        # print the classifier
        text_representation = tree.export_text(classifier)
        print(text_representation)
        # evaluate classification model
        X_test = X_test.tolist() # added for the converted pure-predict
        y_pred = classifier.predict(X_test)
        # print examples
        test_pixel = np.array((200, 1, 0,))
        print(f'Test: {classifier.apply(test_pixel.reshape(1, -1))}')
        
        test_pixel = np.array((200, 200, 100,))
        print(f'Test: {classifier.apply(test_pixel.reshape(1, -1))}')
               
        # print classification model performance
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        # store the classifier in the object
        self.classifier = classifier
        # return the classifier
        return classifier
            
    def _extract_px_in_roi(self, image:np.ndarray, x, y, size_x, size_y):
        x2, y2 = x + size_x, y + size_y
        ch1 = image[y:y2, x:x2, 0].flatten()
        ch2 = image[y:y2, x:x2, 1].flatten()
        ch3 = image[y:y2, x:x2, 2].flatten()
        px_in_roi = np.stack((ch1, ch2, ch3), axis=1)        
        return px_in_roi
    
    def predict(self, X:np.ndarray) -> np.ndarray:
        return self.classifier.predict(X)
    
    def classify_image(self, image_bgr:np.ndarray) -> np.ndarray:
        # convert the image 2d numpy array of shape (height, width, 3) into (height * width, 3), each row is BGR value of a pixel
        image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        X = np.stack((
            image_hsv[:, :, 0].flatten(),
            image_hsv[:, :, 1].flatten(),
            image_hsv[:, :, 2].flatten(),
        ), axis = 1)
        # classified each pixel
        y = self.predict(X)
        # reshape the output which is of shape (height * width,) to (height, width)
        y = y.reshape((image_bgr.shape[0], image_bgr.shape[1]))
        # create the threshold greyscale image
        return np.where(y, 255, 0).astype(np.uint8)
    
    def save_classifier(self, model_file:str):
        with open(model_file, 'wb') as outfile:
            pickle.dump(self.classifier, outfile, protocol=0)

    def load_classifier(self, model_file:str):
        with open(model_file, 'rb') as infile:
            self.classifier = pickle.load(infile)

if __name__ == '__main__':
    train_spec_file = os.path.join(os.path.dirname(__file__), 'tile_filter/train_set_BLUE_20260514.yaml')
    classifier_model_file = os.path.join(os.path.dirname(__file__), 'tile_filter/train_set_BLUE_20260514_hsv.model')
    
    frame_detector = FrameDetector()
    frame_detector.train(train_spec_file)
    frame_detector.save_classifier(classifier_model_file)
    # load the model from a file
    frame_detector_2 = FrameDetector(classifier_model_file)
    # load test image
    # image_bgr = cv2.imread('/home/qcr/cgras_data/Source/2024/Chris_MIS5_T01_241031/CGRAS_Amag_241031_T01_00.jpg')
    # image_bgr = cv2.imread('/home/qcr/Workspace/training_images/blue_corners/resized/water_0_5.jpg')
    image_bgr = cv2.imread('/home/qcr/cgras_ws/src/cgras_detector/src/detector/models/tile_filter/samples/CGRAS_Amil_Room9_20241128_w0_T01_00_crop.jpg')
    image_threshold = frame_detector_2.classify_image(image_bgr)
    cv2.imwrite(os.path.join(os.path.dirname(__file__), 'tile_filter/output.jpg'), image_threshold)