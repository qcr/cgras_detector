# Copyright 2024 - Andrew Kwok Fai LUI, 
# Robotics and Autonomous Systems Group, REF, RI
# and the Queensland University of Technology

__author__ = 'Andrew Lui'
__copyright__ = 'Copyright 2024'
__license__ = 'GPL'
__version__ = '1.0'
__email__ = 'ak.lui@qut.edu.au'
__status__ = 'Development'

# general modules
import os, sys, threading, collections, time, shutil, traceback, fcntl
from enum import Enum
from datetime import datetime
# project modules
from cgras_datatools.logging_tools import logger

class ApplicationFileManager():
    # top level folders
    IMAGES_FOLDER = 'images'
    DATABASE_FOLDER = 'database'
    DETECTOR_FOLDER = 'detector'
    
    # platform subfolders
    SYSTEM_SUBFOLDER = 'system'
    SYSTEM_SCRIPTS_SUBFOLDER = 'system/scripts'
    SYSTEM_IMAGES_SUBFOLDER = 'system/images'
    DATA_SUBFOLDER = 'data'
    TEMP_SUBFOLDER = 'temp'
    
    def __init__(self, cgras_data_folder):
        self.log_lock = threading.Lock()
        self.user_home = os.path.expanduser('~') 
        self.cgras_data_folder = cgras_data_folder
        # crate subfolders
        self.images_folder = os.path.join(self.cgras_data_folder, ApplicationFileManager.IMAGES_FOLDER)
        os.makedirs(self.images_folder, exist_ok=True)
        self.database_folder = os.path.join(self.cgras_data_folder, ApplicationFileManager.DATABASE_FOLDER)
        os.makedirs(self.database_folder, exist_ok=True)        
        self.detector_folder = os.path.join(self.cgras_data_folder, ApplicationFileManager.DETECTOR_FOLDER)
        os.makedirs(self.detector_folder, exist_ok=True)
        # create the subfolders under the two platforms
        self._create_platform_subfolders(self.detector_folder)
        # generate path to the error_log_file
        self.error_log_file = os.path.join(self.cgras_data_folder, 'detector_system_error.log')
    
    @staticmethod
    def _create_platform_subfolders(platform_home):
        system_folder = os.path.join(platform_home, ApplicationFileManager.SYSTEM_SUBFOLDER)
        os.makedirs(system_folder, exist_ok=True)
        data_folder = os.path.join(platform_home, ApplicationFileManager.DATA_SUBFOLDER)
        os.makedirs(data_folder, exist_ok=True)       
        temp_folder = os.path.join(platform_home, ApplicationFileManager.TEMP_SUBFOLDER)
        os.makedirs(temp_folder, exist_ok=True)                   
    
    def get_cgras_home(self) -> str:
        return self.cgras_data_folder    
    
    def get_images_folder(self) -> str:
        return self.images_folder

    def get_database_folder(self) -> str:
        return self.database_folder
        
    def get_detector_folder(self, *args) -> str:
        return self.get_subfolder(self.detector_folder, *args)
    
    def get_detector_subfolder(self, subfolder, *args) -> str:
        if subfolder not in [ApplicationFileManager.SYSTEM_SUBFOLDER, ApplicationFileManager.DATA_SUBFOLDER, ApplicationFileManager.TEMP_SUBFOLDER]:
            raise AssertionError(f'{type(self).__name__}: invalid parameter (subfolder): {subfolder} ')
        return self.get_subfolder(self.detector_folder, subfolder, *args)   
    
    def get_free_disk_space(self) -> float:
        # return the disk space status in GB of the disk where cgras data folder is located 
        total, used, free = shutil.disk_usage(self.cgras_data_folder)
        total = total / (2**30) 
        used = used / (2**30) 
        free = free / (2**30) 
        return total, used, free
    
    def dump_exc_to_error_log(self, header=None):
        try:
            with open(self.error_log_file, 'a') as outfile:
                outfile.write('----------------------------------------------------------\n')
                outfile.write(f'Timestamp: {datetime.now().strftime("%d/%b/%Y %H:%M:%S")}\n')
                if header is not None:
                    outfile.write(f'Title: {header}\n')
                traceback.print_exc(file=outfile)      
                outfile.write('---- End ----\n\n')   
        except:
            logger.warning(f'Unable to dump exception message to the error log file')
    
    @staticmethod
    def get_subfolder(parent_folder:str, *args) -> str:
        """ return the path string of a subfolder of a parent_folder, and create the folder if not exists, with the partial paths specified as
            positional raguments

        :param parent_folder: the parent folder path 
        :type parent_folder: str
        :return: the full path to the subfolder, which has been created if not exists
        :rtype: str
        """
        if args is not None and len(args) > 0:
            parent_folder = os.path.join(parent_folder, *args)
            os.makedirs(parent_folder, exist_ok=True)
        return parent_folder

    # --- copy the images in the images folder and web scripts to the system folder of detector
    def populate_system_assets_folder(self):
        # A file lock prevents two processes (e.g. cgras_detector_node and detector_aux_server_node)
        # from running this concurrently — the rmtree in one would otherwise delete the destination
        # while the other's copytree is writing into it.
        lock_path = os.path.join(self.get_detector_folder(self.SYSTEM_SUBFOLDER), '.populate.lock')
        try:
            with open(lock_path, 'w') as lock_file:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                try:
                    system_folder_path = self.get_detector_folder(self.SYSTEM_SUBFOLDER)
                    logger.info(f'ApplicationFileManager: populating system folder at {system_folder_path}')
                    source_path = os.path.join(os.path.dirname(__file__), 'web/_system/scripts')
                    if os.path.isdir(source_path):
                        shutil.rmtree(self.get_detector_folder(self.SYSTEM_SCRIPTS_SUBFOLDER), ignore_errors=True)
                        shutil.copytree(source_path, self.get_detector_folder(self.SYSTEM_SCRIPTS_SUBFOLDER), dirs_exist_ok=True)
                    source_path = os.path.join(os.path.dirname(__file__), 'web/_system/images')
                    if os.path.isdir(source_path):
                        shutil.rmtree(self.get_detector_folder(self.SYSTEM_IMAGES_SUBFOLDER), ignore_errors=True)
                        shutil.copytree(source_path, self.get_detector_folder(self.SYSTEM_IMAGES_SUBFOLDER), dirs_exist_ok=True)
                finally:
                    fcntl.flock(lock_file, fcntl.LOCK_UN)
            return True
        except Exception as e:
            logger.warning(f'ApplicationFileManager.populate_system_assets_folder: {traceback.format_exc()}')
            return False

    # --- copy the scripts folder to the given destination
    def copy_scripts_folder(self, destination:str):
        try:
            source_path = os.path.join(os.path.dirname(__file__), 'web/_system/scripts')
            logger.info(f'ApplicationFileManager.copy_scripts_folder: copy {source_path} to {destination}')
            shutil.copytree(source_path, destination, dirs_exist_ok=True)
            return True
        except Exception as e:
            logger.warning(f'ApplicationFileManager.copy_scripts_folder: {traceback.format_exc()}')
            return False
