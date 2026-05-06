#!/usr/bin/env python3

# Copyright 2024 - Andrew Kwok Fai LUI, 
# Robotics and Autonomous Systems Group, REF, RI
# and the Queensland University of Technology

__author__ = 'Andrew Lui'
__copyright__ = 'Copyright 2024'
__license__ = 'GPL'
__version__ = '1.0'
__email__ = 'ak.lui@qut.edu.au'
__status__ = 'Development'

import os, datetime, time, shutil, json, numbers, yaml, traceback
import pandas as pd
from enum import Enum
from datetime import datetime as dt
# project modules
import cgras_datatools.db_tools as db_tools
import cgras_datatools.file_tools as file_tools
from cgras_datatools.lock_tools import synchronized
from cgras_datatools.logging_tools import logger
from detector.database_file import DBFile

class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True

# NOTE: The batch_time is an ISO 8601 date time string format '2025-05-29 14:16:00' and the batch_id is derived from the time and cgras_station_id or the importer_id
# NOTE: The date/time is recorded in localtime instead of the usual GMT because the system is not an Internet application and the data is considered locally

# the DDL for creating tables in the detect.db
DETECT_DDL = {
    'tile_sample':
    """
    CREATE TABLE IF NOT EXISTS tile_sample (
        id text PRIMARY KEY,
        tile_id text,
        batch_id text,         
        batch_time text,
        age integer DEFAULT -1,
        species text,
        season text,   
        settle_time text,  
        spawn_time text,
        tab_ncols integer DEFAULT -1,
        tab_nrows integer DEFAULT -1,
        importer_id text,
        operator text,
        create_time text,
        modify_time text,
        status integer DEFAULT -1,
        priority text,
        remarks text DEFAULT '',
        metadata text DEFAULT NULL,
        UNIQUE (tile_id, batch_id)
    );
    """,
    
    'source_image':
    """
    CREATE TABLE IF NOT EXISTS source_image (
        id integer PRIMARY KEY AUTOINCREMENT,
        capture_id text,
        tile_sample_id text, 
        capture_x integer,
        capture_y integer,
        file_path text,     
        metadata text,  
        UNIQUE (capture_id),
        CONSTRAINT fk_tile_sample_id
            FOREIGN KEY (tile_sample_id) REFERENCES tile_sample (id) ON DELETE CASCADE
    );
    """,
    
    'task_record':
    """
    CREATE TABLE IF NOT EXISTS task_record (
        id integer PRIMARY KEY AUTOINCREMENT,
        task_type int,
        task_object text, 
        start_time text,
        used_time real,
        status int,
        remarks text,
        metadata text
    );
    """, 
    
    'yolo_model':
    """
    CREATE TABLE IF NOT EXISTS yolo_model (
        id integer PRIMARY KEY AUTOINCREMENT,
        name text,
        model_file_path text,
        input_image_width integer,
        input_image_height interger,
        species text,
        start_day integer DEFAULT 0,
        end_day integer DEFAULT -1,
        classes_map_yaml TEXT DEFAULT NULL,
        remarks text,
        predict_params_yaml text,
        keep_object_filter_yaml text,
        is_active integer DEFAULT 1,
        UNIQUE (name)
    );
    """,
    # coral_classes text,
    # dead_coral_classes text,

    'tile_sample_detect_stat':
    """
    CREATE TABLE IF NOT EXISTS tile_sample_detect_stat (
        tile_sample_id text PRIMARY KEY,
        tile_pixel_x integer,
        tile_pixel_y integer,
        coral_alive_count integer DEFAULT 0,
        coral_dead_count integer DEFAULT 0,
        other_count integer DEFAULT 0,
        duplicates_removed integer DEFAULT 0,
        stat_yaml text DEFAULT NULL,
        CONSTRAINT fk_tile_sample_id
            FOREIGN KEY (tile_sample_id) REFERENCES tile_sample (id) ON DELETE CASCADE
    );
    """,

    'detected_object':
    """
    CREATE TABLE IF NOT EXISTS detected_object (
        id integer PRIMARY KEY AUTOINCREMENT,
        tile_sample_id text,
        yolo_class text,
        coral_class text,
        present_class text,
        centre_x real,
        centre_y real,
        corner_x1 real,
        corner_y1 real,
        size_x real,
        size_y real,
        contour_area real DEFAULT NULL,
        confidence real DEFAULT 0,
        metadata text DEFAULT NULL,
        CONSTRAINT fk_tile_sample_id
            FOREIGN KEY (tile_sample_id) REFERENCES tile_sample (id) ON DELETE CASCADE
    );
    """, 
    
    'error_flag':
    """
    CREATE TABLE IF NOT EXISTS error_flag (
        id integer,
        object text,
        level integer DEFAULT 0,
        update_time text,
        remarks text,
        PRIMARY KEY (id, object)
    );
    """,    
    
    'health_model':             # NOT USED
    """
    CREATE TABLE IF NOT EXISTS health_model (
        species text PRIMARY KEY,
        func_name text,
        func_def text DEFAULT NULL,
        UNIQUE(func_name)
    );
    """,

    'cache_tile_health_stat':     # NOT USED
    """
    CREATE TABLE IF NOT EXISTS cache_tile_health_stat (
        tile_id text PRIMARY KEY,
        season text,
        species text,
        settle_time text,
        coral_count_start integer,
        age_start integer,
        coral_count_latest integer,
        dead_coral_count_latest integer,
        other_object_count_latest integer,
        age_latest integer,
        batch_time_latest text,
        loss_rate_whole real,
        loss_rate_recent real,
        num_samples integer DEFAULT 0,
        health_index real DEFAULT NULL,
        count_yaml_data text DEFAULT NULL
    );
    """,
}

# the constants defined for storing task types in the task_record table
class TaskTypes(Enum):
    DETECT_CORALS = 0 
    IMPORT_TILES = 1
    ASSESS_HEALTH = 2

# the constants defined for storing presentation class in the detected_object table
class ClassHierarchyPresentation(Enum):
    OTHER = 'OTHER'
    ALIVE_CORAL = 'ALIVE_CORAL'
    DEAD_CORAL = 'DEAD_CORAL'
    MASKED = 'MASKED'

# the constants defined for storing coral class in the detected_object table
class ClassHierarchyCoral(Enum):
    POLYP_SINGLE = 'POLYP_SINGLE'
    POLYP_MULTI = 'POLYP_MULTI'
    POLYP_KEYPART = 'POLYP_KEYPART'
    DEAD_CORAL = 'DEAD_CORAL'
    OTHER = 'OTHER'
    UNDEFINED = 'UNDEFINED'

# the constants defined for storing status in different tables
class TaskStatusNames(Enum):
    UNKNOWN = -1
    PENDING = 0
    SUCCESS = 1
    FAIL = 2
    RESOLVABLE_FAIL = 3
    REJECTED = 4
    
class SampleStatusNames(Enum):
    ALL = -1
    QUEUED = 0
    DONE = 1
    FLAGGED = 2  # ABORTED may be due to interrupted by user or by a recoverable error (not from the data itself) such as no suitable YOLO model
    REJECTED = 4  # REJECTED may be due to rejected by user or rejected by the system if a non-recoverable error is found in the input data

class CoralObject():
    """ CoralObject models an object in the tile images detected by an object detection model. It comprises locational information including the index in the image grid, the index of the blob in each image,
        the bounding box in the blob. It also contains locational information in the tile frame of reference includig bounding box and centre. It contains the class id and the class name string of the object.
    """
    def __init__(self, preserve_fraction=False, **kwargs):
        """ the constructor

        :param preserve_fraction: determine if fractions in the parameters are truncated to integers, defaults to False
        :type preserve_fraction: bool, optional
        """
        # populates the model parameters from keyword input parameters
        self.blob_row_index = kwargs.get('blob_row_index', None)    # the row and column index of the blob where this coral object was detected
        self.blob_col_index = kwargs.get('blob_col_index', None)    
        self.image_row_index = kwargs.get('image_row_index', None)  # the row and column index of the image where this coral object was detected
        self.image_col_index = kwargs.get('image_col_index', None)
        self.cls_id =  kwargs.get('cls_id', None)                   # the class id of the coral object as specified in the detection model
        self.yolo_class = kwargs.get('yolo_class', None)        # the detection model level class of the object
        self.coral_class = kwargs.get('coral_class', None)          # the coral level class of the object
        self.present_class = kwargs.get('present_class', None)      # the presentation level class of the object
        self.bbox_in_blob = kwargs.get('bbox_in_blob', None)        # the bounding box of the coral object in the image blob space (x1, y1, x2, y2)
        self.bbox_in_image = kwargs.get('bbox_in_image', None)      # the bounding box of the coral object in the image space
        self.bbox_in_tile = kwargs.get('bbox_in_tile', None)        # the bounding box of the coral object in the tile space
        self.bbox = self.bbox_in_tile
        self.centre = kwargs.get('centre', None)                    # the centre of the coral object in the file space
        self.size = kwargs.get('size', None)                        # the size of the coral object (xdim, ydim)
        self.bbox_normalized = kwargs.get('bbox_normalized', None)  # the normalized bounding box of the coral object in the tile space
        self.centre_normalized = kwargs.get('centre_normalized', None) # the normalized centre of the coral object in the tile space    
        self.size_normalized = kwargs.get('size_normalized', None) # the normalized size of the coral object in the tile space
        self.points_normalized = kwargs.get('points_normalized', None) # a list of normalized points that defines the contour of the object
        self.contour_area_normalized = kwargs.get('contour_area_normalized', None) # the normalized area of the coral object in the tile space based on the contour
        self.confidence = kwargs.get('confidence', None)            # the confidence of the detection
        self.invalidated = False
        self.inside_of = None                                      # the object is inside of the set object, only relevant during COD model building
        self.index_str = f'{self.image_col_index, self.image_row_index, self.blob_col_index, self.blob_row_index}'
        if not preserve_fraction:
            self._convert_to_int()
        
    def _convert_to_int(self):
        """ internal function for converting some parameters to int type using truncation
        """
        self.bbox_in_blob = self._convert_list_to_int(self.bbox_in_blob) if self.bbox_in_blob is not None else self.bbox_in_blob
        self.bbox = self._convert_list_to_int(self.bbox) if self.bbox is not None else self.bbox
        self.size = self._convert_list_to_int(self.size) if self.size is not None else self.size
        self.centre = self._convert_list_to_int(self.centre) if self.centre is not None else self.centre
        
    @staticmethod
    def _convert_list_to_int(value_list:list) -> list:
        """ internal function for converting the numbers in a list to integers

        :param value_list: the list of numbers
        :type value_list: list
        :return: a new list containing the same numbers truncated to integers
        :rtype: list
        """
        return [int(x) for x in value_list]
    
    def __repr__(self):
        """ Print the content of this coral object

        """
        inv_str = '[INV]' if self.invalidated else '[VAD]' 
        bbox_int = [int(x) for x in self.bbox]
        centre_int = [int(x) for x in self.centre]
        centre_in_blob = (self.bbox_in_blob[0] + self.size[0] // 2, self.bbox_in_blob[1] + self.size[1] // 2)
        area = int(self.size[0] * self.size[1])
        result = f'{inv_str} {self.image_col_index, self.image_row_index, self.blob_col_index, self.blob_row_index} ({self.yolo_class}) bbox ({bbox_int}) centre ({centre_int})'
        if self.centre_normalized is not None:
            result += f'({self.centre_normalized[0]:.5f}, {self.centre_normalized[0]:.5f}) blob ({centre_in_blob}) area ({area})'
        else:
            result += f' blob ({centre_in_blob}) area ({area})'
        return result
    

# Model the data access object for the tables in this database 
class DetectorDAO():
    def __init__(self, db_file:str, **kwargs):
        self.db_file = db_file
        self._migrate_yolo_model_columns()

    def _migrate_yolo_model_columns(self):
        try:
            with db_tools.create_connection(self.db_file) as conn:
                c = conn.cursor()
                existing = {row[1] for row in c.execute('PRAGMA table_info(yolo_model)').fetchall()}
                if 'is_active' not in existing:
                    c.execute('ALTER TABLE yolo_model ADD COLUMN is_active integer DEFAULT 1')
                    c.execute('UPDATE yolo_model SET is_active = 1')
                    conn.commit()
        except Exception:
            pass

    # functions for validate the database
    # return True if there is at least one tile, one tank, one station, and one pattern for the operation
    @synchronized
    def validate_db(self):
        with db_tools.create_connection(self.db_file) as conn:       
            c = conn.cursor() 
            result = c.execute('SELECT COUNT(*) FROM general_config').fetchone()
            if not result or result[0] == 0:
                return False         
        return True

    # - table: tile_sample

    # a function to compute the id of a tile_sample given its tile_id and the batch_id
    @staticmethod
    def compute_tile_sample_id(tile_id:str, batch_id:str) -> str:
        tile_sample_id = f'{tile_id}_{batch_id}'
        return tile_sample_id
    
    # add a record to the tile_sample table, with species normalized to lower case
    @synchronized
    def add_tile_sample(self, tile_id:str, batch_id:str, batch_time:str, age:int, species:str, season:str, tab_ncols:int, tab_nrows:int, settle_time:str, spawn_time:str='', importer_id:str='', operator:str='', 
                        status:int=SampleStatusNames.QUEUED.value, remarks:str='', metadata=None):
        if metadata is not None and not isinstance(metadata, str):
            metadata = json.dumps(metadata)
        sql = 'INSERT INTO tile_sample (id, tile_id, batch_id, batch_time, age, species, season, tab_ncols, tab_nrows, metadata, settle_time, spawn_time, importer_id, operator, status, remarks, create_time, modify_time, priority) \
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, DATETIME("now", "localtime"), DATETIME("now", "localtime"), DATETIME("now", "localtime"))'
        tile_sample_id = self.compute_tile_sample_id(tile_id, batch_id)
        return db_tools.update(self.db_file, sql, (tile_sample_id, tile_id, batch_id, batch_time, age, species.lower(), season, tab_ncols, tab_nrows, metadata, settle_time, spawn_time, importer_id, operator, status, remarks))
    
    # return True if a record of tile_sample exists given the tile_id and the batch_id
    @synchronized
    def exist_tile_sample(self, tile_id:str, batch_id:str) -> dict:
        sql = 'SELECT * FROM tile_sample WHERE id = ?'
        tile_sample_id = self.compute_tile_sample_id(tile_id, batch_id)
        result = db_tools.query_for_object(self.db_file, sql, (tile_sample_id,))
        return True if result is not None else False
    
    # return the number of records of tile sample given a particular status
    @synchronized
    def count_tile_samples(self, status:int=SampleStatusNames.QUEUED.value) -> int:
        if isinstance(status, SampleStatusNames):
            status = status.value
        sql = 'SELECT COUNT(*) FROM tile_sample WHERE status = ?'
        result = db_tools.query_for_object(self.db_file, sql, (status,))
        return result if result is not None else 0
    
    # return a record of tile sample given the id of the tile_sample as a dixt
    @synchronized
    def get_tile_sample(self, tile_sample_id:str) -> dict:
        sql = 'SELECT * FROM tile_sample WHERE id = ?'
        tile_sample_dict = db_tools.query_for_dict(self.db_file, sql, (tile_sample_id,))
        if 'metadata' in tile_sample_dict:
            try:
                tile_sample_dict['metadata'] = json.loads(tile_sample_dict['metadata'])
            except:
                tile_sample_dict['metadata'] = {}
        return tile_sample_dict
    
    # return the list of unique season found in the table
    @synchronized
    def list_seasons_in_tile_sample(self) -> list:
        sql = 'SELECT DISTINCT(season) FROM tile_sample ORDER BY batch_time DESC'
        return db_tools.query_for_list(self.db_file, sql)     
    
    @synchronized
    def list_species_in_tile_sample(self) -> list:
        sql = 'SELECT DISTINCT(species) FROM tile_sample ORDER BY species ASC'
        return db_tools.query_for_list(self.db_file, sql)                  

    @synchronized
    def exist_species_in_tile_sample(self, species) -> list:
        sql = 'SELECT COUNT(*) FROM tile_sample WHERE species = ?'
        result = db_tools.query_for_object(self.db_file, sql, (species,))
        return result >= 1  
    
    @synchronized
    def list_tiles_in_tile_sample(self, season_title:str=None) -> list:
        if season_title is None:
            sql = 'SELECT tile_id, season, species, settle_time FROM tile_sample GROUP BY tile_id'
            return db_tools.query(self.db_file, sql)        
        else:
            sql = 'SELECT tile_id, season, species, settle_time FROM tile_sample WHERE season = ? GROUP BY tile_id'
            return db_tools.query(self.db_file, sql, (season_title,))                    
    
    @synchronized
    def get_tile_info_from_tile_sample(self, tile_id:str, to_dataframe:bool=False) -> pd.DataFrame:
        sql = 'SELECT tile_id, species, season, settle_time, spawn_time, tab_ncols, tab_nrows, remarks FROM tile_sample WHERE tile_id = ? LIMIT 1'
        if to_dataframe:
            return db_tools.query(self.db_file, sql, (tile_id,))                
        return db_tools.query_for_dict(self.db_file, sql, (tile_id,))     
    
    @synchronized
    def get_tile_info_from_tile_sample_as_df(self, tile_id:str) -> pd.DataFrame:
        tile_info = self.get_tile_info_from_tile_sample(tile_id)
        model = pd.DataFrame(columns=('Tile ID', tile_id))
        if tile_info is not None:
            model.loc[1] = ['Species', tile_info['species']]
            model.loc[2] = ['Season', tile_info['season']]                
            model.loc[3] = ['Settled On', tile_info['settle_time']]
            model.loc[4] = ['Spawned On', tile_info['spawn_time']]
            num_tabs = f"{tile_info['tab_ncols']} x {tile_info['tab_nrows']}"
            model.loc[5] = ['Tabs', num_tabs]
        return model

    # return a list of records of tile sample of which the tile id is as given, and the number of records is bounded by the limit parameter
    @synchronized
    def get_tile_sample_of_tile_id(self, tile_id:str, limit:int=1) -> dict:
        return db_tools.query_for_list_of_dicts(self.db_file, 'SELECT * FROM tile_sample WHERE tile_id = ? LIMIT ?', (tile_id, limit))

    @synchronized
    def list_tile_samples_of_tile_id(self, tile_id:str, status:int=None) -> dict:
        if status is not None:
            return db_tools.query(self.db_file, 'SELECT * FROM tile_sample WHERE tile_id = ? AND status = ?', (tile_id, status,))
        else:
            return db_tools.query(self.db_file, 'SELECT * FROM tile_sample WHERE tile_id = ?', (tile_id,))
    
    # return a list of unique tile_id found in the tile_sample table
    @synchronized
    def get_distinct_tile_id_as_list(self, season) -> list:    
        tile_id_list = db_tools.query_for_list(self.db_file, 'SELECT DISTINCT(tile_id) FROM tile_sample WHERE season = ?', (season,))
        return tile_id_list

    # return a dataframe of records given the season title, the status, and maximum records to return
    @synchronized
    def list_tile_samples(self, season_title:str=None, status:int=SampleStatusNames.QUEUED.value, limit=None, order_by_recent=False) -> pd.DataFrame:
        if status is not None:
            sql = 'SELECT * FROM tile_sample WHERE status = ?'
            param_list = [status] 
        else:
            sql = 'SELECT * FROM tile_sample'
            param_list = []                
        
        if season_title:
            if param_list:
                sql += ' AND season = ?'
            else:
                sql += ' WHERE season = ?'
            param_list.append(season_title)
        if order_by_recent:
            sql += ' ORDER BY modify_time DESC'
        else:
            sql += ' ORDER BY priority ASC'
        if limit is None or not isinstance(limit, numbers.Number):
            ...
        else:
            sql += ' LIMIT ?'
            param_list.append(limit)
        return db_tools.query(self.db_file, sql, tuple(param_list)) 
        
    # returns the record of the next pending tile sample, if exists, as a dict
    @synchronized
    def query_next_pending_tile_sample(self) -> dict:
        sql = 'SELECT * FROM tile_sample WHERE status = ? ORDER BY priority ASC LIMIT 1'
        return db_tools.query_for_dict(self.db_file, sql, (SampleStatusNames.QUEUED.value,))                        
    
    # delete all records int he tile_sample table
    @synchronized
    def clear_tile_sample_table(self):
        return db_tools.clear_table(self.db_file, 'tile_sample')    
    
    # update the importer and the oeprator of a tile_sample record given its id
    @synchronized
    def update_importer(self, tile_sample_id:str, importer_id:str, operator:str=None):
        sql = 'UPDATE tile_sample SET importer_id = ?, operator = ? WHERE id = ?'
        return db_tools.update(self.db_file, sql, (importer_id, operator, tile_sample_id))   
    
    # update the status of the tile_sample given its tile_sample_id 
    @synchronized
    def update_tile_sample_status(self, tile_sample_id:str, status:int, remarks:str=None):
        with db_tools.create_connection(self.db_file) as conn:
            c = conn.cursor()
            if remarks is None:
                c.execute('UPDATE tile_sample SET status = ?, modify_time = DATETIME("now", "localtime") WHERE id = ?', (status, tile_sample_id,))
            else:
                c.execute('UPDATE tile_sample SET status = ?, remarks = ?, modify_time = DATETIME("now", "localtime") WHERE id = ?', (status, remarks, tile_sample_id,))
            return True  
        
    @synchronized
    def clear_tile_sample_data(self, tile_sample_id:str):
        with db_tools.create_connection(self.db_file) as conn:
            c = conn.cursor()
            c.execute('DELETE FROM tile_sample_detect_stat WHERE tile_sample_id = ?', (tile_sample_id,))
            c.execute('DELETE FROM detected_object WHERE tile_sample_id = ?', (tile_sample_id,))
            return True
    
    # delete the record of tile_sample given the id
    @synchronized
    def delete_tile_sample(self, tile_sample_id:str):
        ######
        sql = 'DELETE FROM tile_sample WHERE id = ?'
        return db_tools.update(self.db_file, sql, (tile_sample_id,))   
    
    # update the priority field of a tile_sample record given its id
    @synchronized
    def set_top_priority(self, tile_sample_id:str):
        sql = 'UPDATE tile_sample SET priority = (SELECT DATETIME(MIN(priority), "-5 minute") FROM tile_sample WHERE status = ?) WHERE id = ?'
        status = SampleStatusNames.QUEUED
        return db_tools.update(self.db_file, sql, (status.value, tile_sample_id,))   
    
    # return the records from a query based on search keys including season, status, tile_Id, batch_id, and the period
    
    @synchronized
    def query_processed_tile_samples(self, season_title:str=None, status:int=None, tile_id:str=None, batch_id:str=None, the_period:int=None, limit:int=None) -> pd.DataFrame:
        # go through each input parameters and, if defined, included in the query
        param_list = []
        if status is not None:
            if type(status) in [list, tuple]:
                status_list = ','.join('?' * len(status))
                sql = f'SELECT * FROM tile_sample WHERE status IN ({status_list})'
                param_list.extend(status)            
            else:
                sql = 'SELECT * FROM tile_sample WHERE status = ?'
                param_list.append(status)
        else:
            sql = 'SELECT * FROM tile_sample WHERE status <> ?'
            param_list.append(SampleStatusNames.QUEUED.value)
            # param_list.append(SampleStatusNames.DONE.value)
        if season_title:
            sql += ' AND season = ?'
            param_list.append(season_title)            
        if tile_id:
            sql += ' AND tile_id LIKE ?'
            param_list.append(f'%{tile_id}%')
        if batch_id:
            sql += ' AND batch_id LIKE ?'
            param_list.append(f'%{batch_id}%')
        if the_period != 0:
            sql += ' AND create_time >= DATE("now", ?)'
            param_list.append(f'{the_period} days')
        sql += ' ORDER BY modify_time DESC, priority DESC'
        if type(limit) == int:
            sql += ' LIMIT ?'
            param_list.append(limit)
        return db_tools.query(self.db_file, sql, tuple(param_list))      
    
    # - table: source_image
    @synchronized
    def add_source_image(self, capture_id:str, tile_sample_id:str, capture_x:int, capture_y:int, file_path:str, metadata=None) -> int:
        with db_tools.create_connection(self.db_file) as conn:
            c = conn.cursor()
            c.execute('INSERT INTO source_image (capture_id, tile_sample_id, capture_x, capture_y, file_path, metadata) '
                      'VALUES (?, ?, ?, ?, ?, ?)', 
                      (capture_id, tile_sample_id, capture_x, capture_y, file_path, metadata,))
            conn.commit()
            id = c.lastrowid
        return id
    
    @synchronized
    def query_source_images_of_tile_sample(self, tile_sample_id:str) -> dict:
        sql = 'SELECT * FROM source_image WHERE tile_sample_id = ?'
        return db_tools.query_for_list_of_dicts(self.db_file, sql, (tile_sample_id,))
    
    @synchronized
    def delete_source_images_of_tile_sample(self, tile_sample_id:str) -> int:
        sql = 'DELETE FROM source_image WHERE tile_sample_id = ?'
        return db_tools.update(self.db_file, sql, (tile_sample_id,))  
    
    # composite operation: validate yaml file for a new tile sample
    @synchronized
    def validate_tile_sample_import(self, tile_sample_data:dict):
        error_list = []
        # load and validate the data in the yaml config file which has been converted to a YamlConfig object
        tile_id = tile_sample_data.get('tile_id', None)
        batch_id = tile_sample_data.get('batch_id', None)
        batch_time = tile_sample_data.get('batch_time', None)
        species = tile_sample_data.get('species', None)
        season = tile_sample_data.get('season', None)
        settle_time = tile_sample_data.get('settle_time', None)
        num_tabs = tile_sample_data.get('num_tabs', None)
        tile_size = tile_sample_data.get('tile_size', None)
        frame_size = tile_sample_data.get('frame_size', None)
        
        importer_id = tile_sample_data.get('importer_id', 'Unknown')
        operator = tile_sample_data.get('operator', 'Unknown') 
        remarks = tile_sample_data.get('remarks', '')
        images_dict = dict()
        image_files_parent_folder = tile_sample_data.get('image_files_parent_folder', None)
        yaml_images_list = tile_sample_data.get('images', None)
        # validate the first tier data
        if tile_id is None or batch_id is None or yaml_images_list is None:
            error_list.append(f'One of the mandatory fields (tile_id, batch_id, and images) is missing in the yaml file')
        if batch_time is None:
            error_list.append(f'One of the mandatory fields (batch_time) is missing in the yaml file')            
        if species is None or season is None or settle_time is None:
            error_list.append(f'One of the mandatory fields (species, season, settle_time) is missing in the yaml file')       
        if num_tabs is None or type(num_tabs) not in (list, tuple) or len(num_tabs) != 2 or not all(n > 0 and isinstance(n, numbers.Number) for n in num_tabs):
            error_list.append(f'One of the mandatory fields (num_tabs) is not a tuple of 2 positive integers')   
        if tile_size is None or frame_size is None:
            error_list.append(f'One of the mandatory fields (tile_size, frame_size) is missing in the yaml file')               
        # iterate through the images list in the yaml file
        max_x, max_y = -1, -1
        for index, yaml_images in enumerate(yaml_images_list):
            x, y = yaml_images.get('x', None), yaml_images.get('y', None)
            max_x, max_y = max(max_x, x), max(max_y, y)
            filepath = yaml_images.get('file', None)
            if image_files_parent_folder:
                filepath = os.path.join(image_files_parent_folder, filepath)
            if x is None or y is None or filepath is None:
                error_list.append(f'An image entry must include these fields (x, y, file) and one of them is missing at entry {index}') 
            else:
                if not os.path.isfile(filepath):
                    error_list.append(f'The file path given for image at ({x},{y}) does not exist: {filepath}')
                else: 
                    images_dict[(x, y)] = filepath
        # adding the images_dict to the yaml data
        # yaml_data['images_dict'] = images_dict
        # validate the images list
        if max_x == -1 or max_y == -1:
            error_list.append(f'An image entry must include these fields (x, y, file) and one of them is missing at entry {index}') 
        if (max_x + 1) * (max_y + 1) != len(images_dict):
            error_list.append(f'Some image index (x, y) is missing: the indices are expected to span from (0, 0) to ({max_x, max_y})')
        # start adding image data to list of lists
        if self.exist_tile_sample(tile_id, batch_id):
            error_list.append(f'The tile_sample (tile_id={tile_id}, batch_id={batch_id}) exists in the system (previously imported))')
        for index_y in range(max_y + 1):    
            for index_x in range(max_x + 1): 
                if (index_x, index_y) not in images_dict:
                    error_list.append(f'The image index ({index_x, index_y}) is missing from the images list') 
        # add age to the yaml import file
        try:
            tile_sample_data['age'] = (pd.to_datetime(tile_sample_data['batch_time']) - pd.to_datetime(tile_sample_data['settle_time'])).days
        except:
            error_list.append(f'One of the parameters (batch_time, settle_time) is an invalid datetime format ({tile_sample_data["batch_time"]}, {tile_sample_data["settle_time"]})')
        if error_list:
            model = pd.DataFrame(columns=('#', 'Errors'))
            for index, error in enumerate(error_list):
                model.loc[index + 1] = [index, error]
            return False, model
        else:
            # generate a model for display
            model = pd.DataFrame(columns=('Parameters', 'Values'))
            model.loc[1] = ['tile_id', tile_id]
            model.loc[2] = ['batch_id', batch_id]
            model.loc[3] = ['batch_time', batch_time]
            model.loc[4] = ['species', species]
            model.loc[5] = ['season', season]     
            model.loc[6] = ['remarks', remarks]       
            model.loc[7] = ['capture grid dim', f'{max_x + 1} x {max_y + 1}']
            model.loc[8] = ['num images', len(images_dict)]
            return True, model
                
    # composite operation: import yaml file for a new tile sample
    @synchronized
    def import_tile_sample_yaml(self, tile_sample_data:dict) -> bool:
        error_list = []
        # load and validate the data in the yaml config file which has been converted to a YamlConfig object
        tile_id = tile_sample_data.get('tile_id', None)
        batch_id = tile_sample_data.get('batch_id', None)
        batch_time = tile_sample_data.get('batch_time', None)
        age = tile_sample_data.get('age', -1)
        species = tile_sample_data.get('species').lower()
        
        season = tile_sample_data.get('season', None)
        settle_time = tile_sample_data.get('settle_time', None)
        num_tabs = tile_sample_data.get('num_tabs', None)
        tile_size = tile_sample_data.get('tile_size', None)
        frame_size = tile_sample_data.get('frame_size', None)
        
        spawn_time = tile_sample_data.get('spawn_time', None)
        importer_id = tile_sample_data.get('importer_id', 'Unknown')
        operator = tile_sample_data.get('operator', 'Unknown') 
        remarks = tile_sample_data.get('remarks', '')
        image_files_parent_folder = tile_sample_data.get('image_files_parent_folder', None)
        tile_images_list = tile_sample_data.get('images', None)
        # construct metadata dict
        metadata = {
            'num_tabs': num_tabs,
            'tile_size': tile_size,
            'frame_size': frame_size
        }
        try:
            with db_tools.create_connection(self.db_file) as conn:
                c = conn.cursor()
                tile_sample_id = self.compute_tile_sample_id(tile_id, batch_id)
                metadata = json.dumps(metadata)
                sql = 'INSERT INTO tile_sample (id, tile_id, batch_id, batch_time, age, species, season, tab_ncols, tab_nrows, metadata, settle_time, spawn_time, importer_id, operator, status, remarks, create_time, modify_time, priority) \
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, DATETIME("now", "localtime"), DATETIME("now", "localtime"), DATETIME("now", "localtime"))'
                c.execute(sql, (tile_sample_id, tile_id, batch_id, batch_time, age, species.lower(), season, num_tabs[0], num_tabs[1], metadata, settle_time, spawn_time, 
                                importer_id, operator, SampleStatusNames.QUEUED.value, remarks))
                # self.add_tile_sample(tile_id, batch_id, batch_time, age, species, season, num_tabs[0], num_tabs[1], settle_time, spawn_time, importer_id, operator, metadata=metadata)
                sql = 'DELETE FROM source_image WHERE tile_sample_id = ?'
                c.execute(sql, (tile_sample_id,))
                # self.delete_source_images_of_tile_sample(tile_sample_id)
                for index, tile_image in enumerate(tile_images_list):
                    x, y = tile_image.get('x', None), tile_image.get('y', None)
                    filepath = tile_image.get('file', None)
                    # metadata in yaml string format
                    metadata = tile_image.get('metadata', {}) # dump an empty dictionary as default
                    metadata_yaml = yaml.dump(metadata, Dumper=NoAliasDumper)
                    if image_files_parent_folder is not None:
                        filepath = os.path.join(image_files_parent_folder, filepath)
                    capture_id = tile_image.get('capture_id', f'{tile_sample_id}-{x}-{y}')
                    c.execute('INSERT INTO source_image (capture_id, tile_sample_id, capture_x, capture_y, file_path, metadata) '
                        'VALUES (?, ?, ?, ?, ?, ?)', (capture_id, tile_sample_id, x, y, filepath, metadata_yaml,))
                    # self.add_source_image(capture_id, tile_sample_id, x, y, filepath, metadata_yaml)
                conn.commit()
            return True
        except Exception as e:
            # traceback.print_exc()
            # logger.warning(e)
            return False
        
    # - composite operation: obtain sample info for a tile id
    @synchronized
    def get_tile_sample_stat_as_df(self, tile_id:str) -> pd.DataFrame:
        model = pd.DataFrame(columns=('', 'Values'))
        model.loc[1] = ['Total Samples', 0]
        # retrieve date range of samples
        sql = 'SELECT MIN(batch_time) AS min, MAX(batch_time) as max FROM tile_sample WHERE tile_id = ?'
        date_range = db_tools.query_for_dict(self.db_file, sql, (tile_id,))
        if date_range['min'] is not None:
            model.loc[2] = ['Oldest Sample', date_range['min']]
            model.loc[3] = ['Latest Sample', date_range['max']]
            current_index = 4
        else:
            current_index = 2
        # retrieve count statistics of the status
        sql = 'SELECT status, COUNT(*) AS count FROM tile_sample WHERE tile_id = ? GROUP BY status ORDER BY status'
        count_list = db_tools.query_for_list_of_dicts(self.db_file, sql, (tile_id,))
        total = 0
        for index, count_dict in enumerate(count_list):
            total += count_dict['count']
            model.loc[index + current_index] = [f'# {SampleStatusNames(count_dict["status"]).name} Status', count_dict['count']]
        model.loc[1] = ['Total Samples', total]
        return model
    
    # - composite operation: obtain the coral count trend table
    @synchronized
    def get_coral_count_trend_as_df(self, tile_id:str) -> pd.DataFrame: 
        sql = 'SELECT T.batch_time, T.age, S.coral_alive_count, S.tile_sample_id \
            FROM tile_sample_detect_stat S, tile_sample T WHERE T.tile_id = ?  \
            AND S.tile_sample_id = T.id ORDER BY T.batch_time ASC'
        return db_tools.query(self.db_file, sql, (tile_id,)) 
        
    # - table: yolo_model
    @synchronized
    def add_yolo_model(self, name:str, model_file_path:str, species:str, start_day:int, end_day:int, input_image_width:int, input_image_height:int, 
                       classes_map:dict, remarks:str, predict_params_dict:dict, keep_object_filter_dict: dict) -> int:  
        classes_map = {} if classes_map is None else classes_map
        classes_map_yaml = yaml.dump(classes_map, Dumper=NoAliasDumper)

        predict_params_dict = {} if predict_params_dict is None else predict_params_dict
        predict_params_yaml = yaml.dump(predict_params_dict, Dumper=NoAliasDumper)
        
        keep_object_filter_dict = {} if keep_object_filter_dict is None else keep_object_filter_dict
        keep_object_filter_yaml = yaml.dump(keep_object_filter_dict, Dumper=NoAliasDumper)
             
        with db_tools.create_connection(self.db_file) as conn:
            c = conn.cursor()
            c.execute('INSERT INTO yolo_model (name, model_file_path, species, start_day, end_day, input_image_width, input_image_height, classes_map_yaml, remarks, predict_params_yaml, keep_object_filter_yaml) '
                      'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', 
                      (name, model_file_path, species, start_day, end_day, input_image_width, input_image_height, classes_map_yaml, remarks, predict_params_yaml, keep_object_filter_yaml,))
            conn.commit()
            id = c.lastrowid
        return id
    
    @synchronized
    def list_yolo_model(self) -> pd.DataFrame:
        sql = 'SELECT * FROM yolo_model ORDER BY species ASC, start_day ASC'
        return db_tools.query(self.db_file, sql)    
    
    def populate_default_yolo_classes_map(classes_map:dict) -> None:
        if classes_map is None:
            classes_map = {}
        valid_class_categories = ['polyp_single', 'polyp_multi', 'polyp_keypart', 'coral_dead']
        for class_cat in valid_class_categories:
            if class_cat not in classes_map:
                classes_map[class_cat] = []
                
    @synchronized
    def set_yolo_model_active(self, name:str, is_active:int) -> int:
        sql = 'UPDATE yolo_model SET is_active = ? WHERE name = ?'
        return db_tools.update(self.db_file, sql, (is_active, name,))

    @synchronized
    def query_yolo_model(self, species, days_since_settle) -> list:
        sql = 'SELECT * FROM yolo_model WHERE species = ? AND (? >= start_day) AND (end_day == -1 or ? <= end_day) AND is_active = 1 ORDER BY start_day ASC'
        result_list = db_tools.query_for_list_of_dicts(self.db_file, sql, (species, days_since_settle, days_since_settle,))  
        for result in result_list: 
            try:
                result['classes_map'] = yaml.load(result['classes_map_yaml'], Loader=yaml.Loader)
            except:     
                result['classes_map'] = {}
            try:
                result['predict_params'] = yaml.load(result['predict_params_yaml'], Loader=yaml.Loader)
            except:     
                result['predict_params'] = {}  
            try:
                result['keep_object_filter'] = yaml.load(result['keep_object_filter_yaml'], Loader=yaml.Loader)
            except:     
                result['keep_object_filter'] = {}                               
            
        return result_list
    
    @synchronized
    def get_yolo_model(self, name) -> list:
        sql = 'SELECT * FROM yolo_model WHERE name = ?'
        result = db_tools.query_for_dict(self.db_file, sql, (name,))  
        if result is None:
            return None
        try:
            result['classes_map'] = yaml.load(result['classes_map_yaml'], Loader=yaml.Loader)
        except:     
            result['classes_map'] = {}   
        try:
            result['predict_params'] = yaml.load(result['predict_params_yaml'], Loader=yaml.Loader)
        except:     
            result['predict_params'] = {}  
        try:
            result['keep_object_filter'] = yaml.load(result['keep_object_filter_yaml'], Loader=yaml.Loader)
        except:     
            result['keep_object_filter'] = {}                                
        return result
    
    @synchronized
    def delete_yolo_model(self, name:str) -> int:
        sql = 'DELETE FROM yolo_model WHERE name = ?'
        return db_tools.update(self.db_file, sql, (name,))   
    
    @synchronized
    def update_yolo_model(self, name:str, species:str, start_day:int, end_day:int) -> int:
        sql = 'UPDATE yolo_model SET species = ?, start_day = ?, end_day = ? WHERE name = ?'
        return db_tools.update(self.db_file, sql, (species, start_day, end_day, name,))  

    @staticmethod
    def get_acceptable_yolo_predict_params_list():
        return ['conf', 'iou', 'agnostic_nms']
    
    @staticmethod
    def get_acceptable_keep_object_filter_list():
        return ['apply', 'aspect_ratio_max', 'area_min']    

    # composite operation: validate yaml file for a new yolo model
    @synchronized
    def validate_yolo_model_file_import(self, yolo_spec_data:dict) -> tuple:
        error_list = []
        # load and validate the data in the yaml config file which has been converted to a YamlConfig object
        name = yolo_spec_data.get('name', None)
        file = yolo_spec_data.get('file', None)
        species = yolo_spec_data.get('species', None)
        valid_start_day = yolo_spec_data.get('valid_start_day', None)
        valid_end_day = yolo_spec_data.get('valid_end_day', None)
        if valid_start_day is None or not isinstance(valid_start_day, numbers.Number):
            valid_start_day = yolo_spec_data['valid_end_day'] = 0
        if valid_end_day is None or not isinstance(valid_end_day, numbers.Number):
            valid_end_day = yolo_spec_data['valid_end_day'] = -1        

        input_image_width = yolo_spec_data.get('input_image_width', None)
        input_image_height = yolo_spec_data.get('input_image_height', None)
        classes_map = yolo_spec_data.get('classes_map', {}) 
        remarks = yolo_spec_data.get('remarks', None)  
        # extract the yolo predict params list
        predict_params_dict = yolo_spec_data.get('yolo_predict_params', {}) 
        accept_params_list = self.get_acceptable_yolo_predict_params_list()
        for key in list(predict_params_dict.keys()):
            if key not in accept_params_list or predict_params_dict[key] is None:
                del predict_params_dict[key]    
        # extract the keep object filter 
        keep_object_filter_dict = yolo_spec_data.get('keep_object_filter', {})
        accept_params_list = self.get_acceptable_keep_object_filter_list()
        for key in list(keep_object_filter_dict.keys()):
            if key not in accept_params_list or keep_object_filter_dict[key] is None:
                del keep_object_filter_dict[key]            
        # validate data
        if name is None or file is None or species is None:
            error_list.append(f'One of the mandatory fields (name, file, species) is missing in the yaml file')
        if input_image_width is None or input_image_height is None or not isinstance(input_image_width, numbers.Integral) or not isinstance(input_image_height, numbers.Integral):
            error_list.append(f'One of the mandatory fields (input_image_width, input_image_height) is missing or not a number in the yaml file')
        if self.get_yolo_model(name) is not None:
            error_list.append(f'The yolo model name "{name}" is already defined in the system.  Change the name to a new one.')
        if error_list:
            model = pd.DataFrame(columns=('#', 'Errors'))
            for index, error in enumerate(error_list):
                model.loc[index + 1] = [index, error]
            return False, model
        else:
            model = pd.DataFrame(columns=('Parameters', 'Values'))
            model.loc[1] = ['name', name]
            model.loc[2] = ['file', file]
            model.loc[3] = ['species', species]
            model.loc[4] = ['valid period', self.get_period_str(valid_start_day, valid_end_day)]
            model.loc[5] = ['input image size', f'{input_image_width}(W) x {input_image_height}(H)']
            row_index = 6
            if predict_params_dict:
                model.loc[row_index] = ['YOLO predict params', str(predict_params_dict)]
                row_index += 1
            if keep_object_filter_dict:
                model.loc[row_index] = ['keep object filter', str(keep_object_filter_dict)]
                row_index += 1
            if len(classes_map) == 0:
                model.loc[row_index] = ['classes map', 'not set']
            else:
                model.loc[row_index] = ['classes map', '']
                row_index += 1
                # iterate through the classes_map
                for index, class_map_key in enumerate(classes_map.keys()):
                    try:
                        ClassHierarchyCoral(class_map_key)
                        class_map_values = classes_map[class_map_key]
                        model.loc[row_index] = ['', f'{class_map_key} [{" ".join(class_map_values)}]']
                    except:
                        model.loc[row_index] = ['', f'{class_map_key} is an invalid coral class name']
                    row_index += 1
                            
            if remarks is not None:
                model.loc[row_index] = ['remarks', remarks]   
            
            return True, model
    
    @staticmethod
    def get_period_str(valid_start_day, valid_end_day):
        valid_start_day = 0 if valid_start_day is None else valid_start_day
        valid_end_day = -1 if valid_end_day is None else valid_end_day
        if valid_start_day == 0 and valid_end_day == -1:
            period = 'the whole period'
        elif valid_start_day == 0:
            period = f'from start to day {valid_end_day}'
        elif valid_end_day == -1:
            period = f'from day {valid_start_day} to the end'
        else:
            period = f'from day {valid_start_day} to day {valid_end_day}'
        return period
    
    # composite operation: import yaml file for a yolo model
    @synchronized
    def import_yolo_model_yaml(self, yolo_spec_data:dict, default_start_day:int, default_end_day:int) -> bool:
        error_list = []
        # load and validate the data in the yaml config file which has been converted to a YamlConfig object
        name = yolo_spec_data.get('name', None)
        model_file_path = yolo_spec_data.get('file', None)
        species = yolo_spec_data.get('species').lower()
        valid_start_day = yolo_spec_data.get('valid_start_day', default_start_day)
        valid_end_day = yolo_spec_data.get('valid_end_day', default_end_day)
        input_image_width = yolo_spec_data.get('input_image_width')
        input_image_height = yolo_spec_data.get('input_image_height')  
        classes_map = yolo_spec_data.get('classes_map', {}) 
        remarks = yolo_spec_data.get('remarks', None)  
        # populate the classes map with the default class names
        for class_name in ClassHierarchyCoral:
            if class_name == ClassHierarchyCoral.UNDEFINED:
                continue
            if class_name.name not in classes_map:
                classes_map[class_name.name] = []
        # get the yolo predict params dict, the parameter list is already validated and trimmed in validate_yolo_model_file_import  
        predict_params_dict = yolo_spec_data.get('yolo_predict_params', {}) 
        keep_object_filter_dict = yolo_spec_data.get('keep_object_filter', {}) 
        # add yolo model to the db
        try:
            with db_tools.create_connection(self.db_file) as conn: 
                conn.isolation_level = None  # to turn off auto-commit (may be unnecessary, minor issue, to check)
                if self.add_yolo_model(name, model_file_path, species, valid_start_day, valid_end_day, input_image_width, input_image_height, 
                                       classes_map, remarks, predict_params_dict, keep_object_filter_dict) > 0:
                    return True
            logger.warning(f'Failed to add yolo model to the database')
            return False
        except Exception as e:
            logger.warning(e)
            return False
        
    # - table: detected_objet
    @synchronized
    def add_detected_object(self, tile_sample_id:str, yolo_class:str, coral_class:str, present_class:str, centre_x:float, centre_y:float, 
                            corner_x1:float, corner_y1:float, size_x:float, size_y:float, contour_area:float=-1, confidence:float=0, metadata:str=None) -> int:
        with db_tools.create_connection(self.db_file) as conn:
            c = conn.cursor()
            c.execute('INSERT INTO detected_object (tile_sample_id, yolo_class, coral_class, present_class, centre_x, centre_y, corner_x1, corner_y1, size_x, size_y, contour_area, confidence, metadata) '
                      'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', 
                      (tile_sample_id, yolo_class, coral_class, present_class, centre_x, centre_y, corner_x1, corner_y1, size_x, size_y, contour_area, confidence, metadata))
            conn.commit()
            id = c.lastrowid
        return id
    
        
    @synchronized
    def add_detected_object_from_coral_object_obs(self, tile_sample_id:str, coral_object:CoralObject):
        centre_x, centre_y = coral_object.centre_normalized[0], coral_object.centre_normalized[1]
        corner_x1, corner_y1 = coral_object.bbox_normalized[0], coral_object.bbox_normalized[1]
        size_x, size_y = coral_object.size_normalized[0], coral_object.size_normalized[1] 
        contour_area, confidence, metadata = -1, 0, None
        if hasattr(coral_object, 'contour_area_normalized'):
            contour_area = coral_object.contour_area_normalized
        if hasattr(coral_object, 'points_normalized'):
            points_normalized = coral_object.points_normalized
            metadata = yaml.dump(points_normalized, Dumper=yaml.Dumper)
        if hasattr(coral_object, 'confidence'):
            confidence = coral_object.confidence
            
        return self.add_detected_object(tile_sample_id, coral_object.yolo_class, coral_object.coral_class, coral_object.present_class,
                                        centre_x, centre_y, corner_x1, corner_y1, size_x, size_y, contour_area, confidence, metadata)        

    def add_detected_object_from_coral_object_list(self, tile_sample_id, coral_object_list, stat:dict=None, exclude_outside_of_tile:bool=True):
        if stat is None:
            stat = {
                'coral_alive_count': 0,
                'coral_dead_count': 0,
                'other_count': 0,
                'masked': 0,
            } 
        with db_tools.create_connection(self.db_file) as conn:
            c = conn.cursor()        
            coral_object:CoralObject
            for coral_object in coral_object_list:
                if coral_object.invalidated:
                    continue
                # exclude the objects that are outside of the tile area, which is between (0, 0) and (1, 1)
                centre_x, centre_y = coral_object.centre_normalized[0], coral_object.centre_normalized[1]
                if exclude_outside_of_tile and (centre_x < 0 or centre_x >= 1 or centre_y < 0 or centre_y >= 1):
                    continue
                corner_x1, corner_y1 = coral_object.bbox_normalized[0], coral_object.bbox_normalized[1]
                size_x, size_y = coral_object.bbox_normalized[2] - corner_x1, coral_object.bbox_normalized[3] - corner_y1
                # tally the coral types
                if coral_object.present_class == ClassHierarchyPresentation.ALIVE_CORAL.value:
                    stat['coral_alive_count'] += 1
                elif coral_object.present_class == ClassHierarchyPresentation.DEAD_CORAL.value:
                    stat['coral_dead_count'] += 1
                elif coral_object.present_class == ClassHierarchyPresentation.OTHER.value:
                    stat['other_count'] += 1
                elif coral_object.present_class == ClassHierarchyPresentation.MASKED.value:
                    stat['masked'] += 1 
                # handle contour points and contour size
                if hasattr(coral_object, 'contour_area_normalized'):
                    contour_area = coral_object.contour_area_normalized
                if hasattr(coral_object, 'points_normalized'):
                    points_normalized = coral_object.points_normalized
                    metadata = yaml.dump(points_normalized, Dumper=yaml.Dumper)
                if hasattr(coral_object, 'confidence'):
                    confidence = coral_object.confidence       
                c.execute('INSERT INTO detected_object (tile_sample_id, yolo_class, coral_class, present_class, centre_x, centre_y, corner_x1, corner_y1, size_x, size_y, contour_area, confidence, metadata) '
                      'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', 
                      (tile_sample_id, coral_object.yolo_class, coral_object.coral_class, coral_object.present_class, centre_x, centre_y, corner_x1, corner_y1, size_x, size_y, contour_area, confidence, metadata))
            conn.commit()
        return stat

    @synchronized
    def delete_detected_objects_of_tile_sample(self, tile_sample_id:str) -> int:
        sql = 'DELETE FROM detected_object WHERE tile_sample_id = ?'
        return db_tools.update(self.db_file, sql, (tile_sample_id,))  
    
    @synchronized
    def query_detected_objects(self, tile_sample_id:str, yolo_classes=None, coral_classes:list=None, present_classes:list=None) -> pd.DataFrame:
        param_list = [tile_sample_id]
        sql = 'SELECT * FROM detected_object WHERE tile_sample_id = ? '
        if yolo_classes is not None:
            if type(yolo_classes) in [list, tuple]:
                yolo_classes_list = ','.join('?' * len(yolo_classes))
                sql += f' AND yolo_class IN ({yolo_classes_list})'
                param_list.extend(yolo_classes)            
            else:
                sql += ' AND yolo_class = ?'
                param_list.append(yolo_classes)
        if coral_classes is not None:
            if type(coral_classes) in [list, tuple]:
                coral_classes_list = ','.join('?' * len(coral_classes))
                sql += f' AND coral_class IN ({coral_classes_list})'
                param_list.extend(coral_classes)            
            else:
                sql += ' AND coral_class = ?'
                param_list.append(coral_classes)
        if present_classes is not None:
            if type(present_classes) in [list, tuple]:
                present_classes_list = ','.join('?' * len(present_classes))
                sql += f' AND present_class IN ({present_classes_list})'
                param_list.extend(present_classes)            
            else:
                sql += ' AND present_class = ?'
                param_list.append(present_classes)                
                
        return db_tools.query(self.db_file, sql, tuple(param_list))
    
    @synchronized
    def query_detected_objects_as_coral_objects(self, tile_sample_id:str, yolo_classes:list=None, coral_classes:list=None, present_classes:list=None) -> list:
        detected_object_list = self.query_detected_objects(tile_sample_id, yolo_classes, coral_classes, present_classes).to_dict('records')
        coral_object_list = []
        for detected_object in detected_object_list:
            contour_area = detected_object.get('contour_area', -1)
            confidence = detected_object.get('confidence', 0)
            metadata = detected_object.get('metadata', None)
            
            contour_points = None
            if metadata is not None:
                try:
                    contour_points = yaml.load(metadata, Loader=yaml.Loader)
                except:
                    ...
            # create the object from db results
            coral_object = CoralObject(
                yolo_class = detected_object['yolo_class'],
                coral_class = detected_object['coral_class'],
                present_class = detected_object['present_class'],
                centre = (detected_object['centre_x'], detected_object['centre_y'],),
                bbox_normalized = (detected_object['corner_x1'], detected_object['corner_y1'], detected_object['corner_x1'] + detected_object['size_x'], detected_object['corner_y1'] + detected_object['size_y']),
                centre_normalized = (detected_object['centre_x'], detected_object['centre_y'],),
                size_normalized = (detected_object['size_x'], detected_object['size_y'],),
                points_normalized = contour_points,
                contour_area_normalized = contour_area,
                confidence = confidence
            )
            coral_object_list.append(coral_object)
        return coral_object_list
    
    @synchronized
    def list_coral_classes(self, tile_sample_id:str=None) -> pd.DataFrame:
        if tile_sample_id is None:
            sql = 'SELECT DISTINCT(coral_class) FROM detected_object'
            return db_tools.query_for_list(self.db_file, sql)          
        else:
            sql = 'SELECT DISTINCT(coral_class) FROM detected_object WHERE tile_sample_id = ?'
            return db_tools.query_for_list(self.db_file, sql, (tile_sample_id,))    
             
    @synchronized
    def list_yolo_classes(self, tile_sample_id:str=None) -> pd.DataFrame:
        if tile_sample_id is None:
            sql = 'SELECT DISTINCT(yolo_class) FROM detected_object'
            return db_tools.query_for_list(self.db_file, sql)          
        else:
            sql = 'SELECT DISTINCT(yolo_class) FROM detected_object WHERE tile_sample_id = ?'
            return db_tools.query_for_list(self.db_file, sql, (tile_sample_id,))    

    # - table: tile sample stat
    @synchronized
    def update_tile_sample_detect_stat(self, tile_sample_id:str, tile_pixel_x, tile_pixel_y, coral_alive_count, coral_dead_count, 
                                       other_count, duplicates_removed, stat_yaml) -> int:
        sql = 'REPLACE INTO tile_sample_detect_stat(tile_sample_id, tile_pixel_x, tile_pixel_y, coral_alive_count, coral_dead_count, \
            other_count, duplicates_removed, stat_yaml) VALUES (?, ?, ?, ?, ?, ?, ?, ?)'
        return db_tools.update(self.db_file, sql, (tile_sample_id, tile_pixel_x, tile_pixel_y, coral_alive_count, coral_dead_count, 
                                                   other_count, duplicates_removed, stat_yaml))

    @synchronized
    def get_tile_sample_detect_stat(self, tile_sample_id:str) -> dict:
        sql = 'SELECT * FROM tile_sample_detect_stat WHERE tile_sample_id = ?'
        return db_tools.query_for_dict(self.db_file, sql, (tile_sample_id,))      


    # - composite operation: update cac
    #he
    @synchronized
    def get_detect_stat_of_tile_id(self, tile_id:str) -> list:
        sql = 'SELECT * FROM tile_sample_detect_stat S LEFT OUTER JOIN tile_sample T ON S.tile_sample_id = T.id WHERE T.tile_id = ? ORDER BY T.batch_time ASC'
        stat_list = db_tools.query_for_list_of_dicts(self.db_file, sql, (tile_id,))
        return stat_list

    @synchronized
    def update_basic_cache_tile_health_stat(self, tile_id, season, species, settle_time):
        return db_tools.update(self.db_file, 'REPLACE INTO cache_tile_health_stat(tile_id, season, species, settle_time) VALUES (?, ?, ?, ?)', (tile_id, season, species, settle_time))

    @synchronized
    def update_cache_tile_health_stat(self, tile_id, season, species, settle_time, coral_count_start, age_start, coral_count_latest, dead_coral_count_latest,
                                            other_object_count_latest, age_latest, batch_time_latest, loss_rate_whole, loss_rate_recent, num_samples, health_index, count_data):
        count_data = yaml.dump(count_data, Dumper=NoAliasDumper)
        sql = 'REPLACE INTO cache_tile_health_stat (tile_id, season, species, settle_time, coral_count_start, age_start, coral_count_latest, dead_coral_count_latest, other_object_count_latest, \
                age_latest, batch_time_latest, loss_rate_whole, loss_rate_recent, num_samples, health_index, count_yaml_data) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'
        db_tools.update(self.db_file, sql, (tile_id, season, species, settle_time, coral_count_start, age_start, coral_count_latest, dead_coral_count_latest,
                                            other_object_count_latest, age_latest, batch_time_latest, loss_rate_whole, loss_rate_recent, num_samples, health_index, count_data)) 

    @synchronized
    def add_tile_df_to_cache_tile_health(self, tile_df:list=None):
        for index, row in tile_df.iterrows():
            sql = 'INSERT OR REPLACE INTO cache_tile_health_stat (tile_id, season, species, settle_time) SELECT ?, ?, ?, ? WHERE NOT EXISTS \
                (SELECT * FROM cache_tile_health_stat WHERE tile_id = ?)'
            db_tools.update(self.db_file, sql, (row['tile_id'], row['season'], row['species'], row['settle_time'], row['tile_id'],))

    @synchronized
    def list_all_cache_tile_health(self, season:str=None) -> list:
        if season is None:
            sql = 'SELECT tile_id, species, settle_time, num_samples, coral_count_latest, age_latest, loss_rate_whole, loss_rate_recent, health_index \
                    FROM cache_tile_health_stat GROUP BY tile_id ORDER BY tile_id ASC'
            return db_tools.query(self.db_file, sql,) 
        else:
            sql = 'SELECT tile_id, species, settle_time, num_samples, coral_count_latest, age_latest, loss_rate_whole, loss_rate_recent, health_index \
                    FROM cache_tile_health_stat WHERE season = ? GROUP BY tile_id ORDER BY tile_id ASC'
            return db_tools.query(self.db_file, sql, (season,))
        
    @synchronized
    def list_tiles_in_cache_tile_health(self, season_title:str=None) -> list:
        if type(season_title) == str:
            sql = 'SELECT MIN(settle_time) AS settle_time, tile_id, species FROM cache_tile_health_stat WHERE season = ? GROUP BY tile_id ORDER BY tile_id ASC'
            return db_tools.query(self.db_file, sql, (season_title,))   
        else:
            sql = 'SELECT MIN(settle_time) AS settle_time, tile_id, species FROM cache_tile_health_stat GROUP BY tile_id ORDER BY tile_id ASC'
            return db_tools.query(self.db_file, sql)  

    # - table: health_model
    @synchronized
    def list_health_model(self) -> pd.DataFrame:
        sql = 'SELECT * FROM health_model ORDER BY species ASC'
        return db_tools.query(self.db_file, sql)    
    
    
    @synchronized
    def get_health_model(self, species) -> dict:
        sql = 'SELECT * FROM health_model WHERE species = ?'
        result = db_tools.query_for_dict(self.db_file, sql, (species,))   
        return result
    
    @synchronized
    def delete_health_model(self, species:str) -> int:
        sql = 'DELETE FROM health_model WHERE species = ?'
        return db_tools.update(self.db_file, sql, (species,))   
    
    @synchronized
    def exist_health_model_func_name(self, func_name) -> bool:
        sql = 'SELECT COUNT(*) FROM health_model WHERE func_name = ?'
        result = db_tools.query_for_object(self.db_file, sql, (func_name,))
        return result >= 1
    
    # composite operation: validate yaml file for a new health model
    @synchronized
    def validate_health_model_file_import(self, yaml_data:dict) -> tuple:
        error_list = []
        # load and validate the data in the yaml config file which has been converted to a YamlConfig object
        species = yaml_data.get('species', None)
        func_name = yaml_data.get('func_name', None)
        func_def = yaml_data.get('func_def', None)        
        # validate data
        if species is None or func_name is None or func_def is None:
            error_list.append(f'One of the mandatory fields (species, func_name, func_def) is missing in the yaml file')
             
        if error_list:
            model = pd.DataFrame(columns=('#', 'Errors'))
            for index, error in enumerate(error_list):
                model.loc[index + 1] = [index, error]
            return False, model
        else:
            model = pd.DataFrame(columns=('Parameters', 'Values'))
            model.loc[1] = ['func_name', func_name]
            model.loc[2] = ['species', species]
            model.loc[3] = ['func_def', func_def]
            return True, model
        
    # composite operation: import yaml file for a health model
    @synchronized
    def import_health_model_yaml(self, yaml_data:dict) -> bool:
        # load and validate the data in the yaml config file which has been converted to a YamlConfig object
        species = yaml_data.get('species', None)
        func_name = yaml_data.get('func_name', None)
        func_def = yaml_data.get('func_def', None)   
        try:
            with db_tools.create_connection(self.db_file) as conn: 
                conn.isolation_level = None  # to turn off auto-commit (may be unnecessary, minor issue, to check)
                c = conn.cursor()
                c.execute('REPLACE INTO health_model (species, func_name, func_def) VALUES (?, ?, ?)', (species, func_name, func_def,))
                conn.commit()
                return True
        except Exception as e:
            logger.warning(f'Failed to add health model to the database')
            return False

    # - table: source_image
    @synchronized
    def add_task_record(self, task_type:int, task_object:str, start_time:str, used_time:float, status:int, remarks:str=None, metadata=None) -> int:
        with db_tools.create_connection(self.db_file) as conn:
            c = conn.cursor()
            if metadata is not None:
                metadata = yaml.dump(metadata, Dumper=NoAliasDumper)
            c.execute('INSERT INTO task_record (task_type, task_object, start_time, used_time, status, remarks, metadata) '
                      'VALUES (?, ?, ?, ?, ?, ?, ?)', 
                      (task_type, task_object, start_time, used_time, status, remarks, metadata,))
            conn.commit()
            id = c.lastrowid
        return id

    @synchronized
    def list_recent_task_records(self, limit=None) -> pd.DataFrame:
        if limit is None:
            sql = 'SELECT task_type, task_object, start_time, used_time, status, remarks FROM task_record ORDER BY start_time DESC'
            return db_tools.query(self.db_file, sql)  
        else:
            sql = 'SELECT task_type, task_object, start_time, used_time, status, remarks FROM task_record ORDER BY start_time DESC LIMIT ?'
            return db_tools.query(self.db_file, sql, (limit,))
    
    @synchronized
    def clear_all_task_records(self):
        sql = 'DELETE FROM task_record'
        return db_tools.update(self.db_file, sql)   

    # - composite operation: obtain task record statistics
    @synchronized
    def get_task_records_stat_as_df(self) -> pd.DataFrame:
        model = pd.DataFrame(columns=('', 'Values'))
        row_index = 1
        # number of pending samples
        with db_tools.create_connection(self.db_file) as conn:
            # count pending samples
            sql = 'SELECT COUNT(*) FROM tile_sample WHERE status = ?'
            result = db_tools.query_for_object(self.db_file, sql, (SampleStatusNames.QUEUED.value,))
            result = 0 if result is None else result
            model.loc[row_index] = [f'# PENDING SAMPLES', result]
            row_index += 1            
            # number of tasks completed
            sql = 'SELECT status, COUNT(*) as count FROM task_record WHERE task_type = ? GROUP BY status ORDER BY status'
            results = conn.execute(sql, (TaskTypes.DETECT_CORALS.value,)).fetchall()
            for num_status in results:
                status_name = TaskStatusNames(num_status[0]).name
                model.loc[row_index] = [f'# {status_name}', num_status[1]]
                row_index += 1            
            # mean duration of DETECT_CORAL task
            sql = 'SELECT AVG(used_time) as mean_duration FROM task_record WHERE task_type = ? AND status = ?'
            mean_duration = db_tools.query_for_object(self.db_file, sql, (TaskTypes.DETECT_CORALS.value, TaskStatusNames.SUCCESS.value))
            if mean_duration is not None:
                model.loc[row_index] = ['Mean Time (SUCCESS)', f'{mean_duration:.1f} s']
                row_index += 1
        return model
    
    # - table: error flag
    @synchronized
    def set_error_flag(self, id:int, obj:str=None, remarks:str=None, level:int=0) -> int:
        sql = 'REPLACE INTO error_flag(id, object, update_time, remarks, level) VALUES (?, ?, DATETIME("now", "localtime"), ?, ?)'
        return db_tools.update(self.db_file, sql, (id, obj, remarks, level,))    

    @synchronized
    def list_error_flags(self) -> int:
        sql = 'SELECT * FROM error_flag ORDER BY update_time DESC'
        return db_tools.query(self.db_file, sql)
    
    @synchronized
    def unset_error_flag(self, id:int, obj:str=None) -> int:
        if obj is None or obj == '':
            sql = 'DELETE FROM error_flag WHERE id = ? AND object IS NULL'
            return db_tools.update(self.db_file, sql, (id,)) 
        else:
            sql = 'DELETE FROM error_flag WHERE id = ? AND object = ?'
            return db_tools.update(self.db_file, sql, (id, obj))       

    @synchronized
    def clear_error_flags(self) -> int:
        sql = 'DELETE FROM error_flag'
        return db_tools.update(self.db_file, sql)  

# ------------------------------------------------
def manage_tables():
    CGRAS_HOME = '/home/qcr/cgras_data'
    DATABASE_FOLDER = os.path.join(CGRAS_HOME, 'database')
    DETECT_DBFM = DBFile(DATABASE_FOLDER, 'detector.db', DETECT_DDL)
    DETECT_DAO = DetectorDAO(DETECT_DBFM.db_file)
    DETECT_DBFM.drop_tables(['yolo_model'])
    tables_name = DETECT_DBFM.list_tables_name()
    logger.info(f'tables: {tables_name}')
    DETECT_DBFM.create_tables(['yolo_model'])
    DETECT_DBFM.dump_all_tables()       

# The main program for testing the clearing
# of database tables and creating them
if __name__ == '__main__':
    manage_tables()

