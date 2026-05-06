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

import os, sys, json
from detector.database_file import DBFile
from cgras_datatools import db_tools
from detector.model import APP_FILE_MANAGER, DETECT_DBFM, DetectorDAO

COORDINATOR_DBFILE = os.path.join(APP_FILE_MANAGER.database_folder, 'coordinator.db')
COORDINATOR_DBFM = DBFile(APP_FILE_MANAGER.database_folder, 'coordinator.db', {})
DETECT_DBFILE = os.path.join(APP_FILE_MANAGER.database_folder, 'detector.db')

# The function that supports interactive execution of sql statements 
def run_db():
    while True:
        print(f'''
        (E): Exit
        (1): COORDINATOR
        (2): DETECT
        (3): Setup Database
        ''')
        command = input('Select DB: ')
        if command == 'E':
            sys.exit(0)
        elif command == '1':
            db_file = COORDINATOR_DBFILE
        elif command == '2':
            db_file = DETECT_DBFILE
        elif command == '3':
            DETECT_DBFM.create_tables()
            continue

        print(f'''
        (E): Exit
        (Q): Run Query
        (U): Run Update
        ''')

        command = input('Command: ')
        if command == 'E':
            sys.exit(0)
        if db_file == DETECT_DBFILE:
            table_names = DETECT_DBFM.list_tables_name()
        else:
            table_names = COORDINATOR_DBFM.list_tables_name()
        print(f'Table names: {table_names}')
        if command == 'Q':
            while True:
                sql = input('Enter SQL: ')
                if not sql: break
                try:
                    df = db_tools.query(db_file, sql)
                    print(df)
                except Exception as e:
                    print(f'Error: {e}')
        elif command == 'U':
            while True:
                sql = input('Enter SQL: ')
                if not sql: break
                try:
                    result = db_tools.update(db_file, sql)
                    print(result)
                except Exception as e:
                    print(f'Error: {e}')
                    
def run_update_tile_sample():
    sql = 'SELECT * FROM tile_sample'
    tile_sample_list = db_tools.query_for_list_of_dicts(DETECT_DBFILE, sql)
    for tile_sample_dict in tile_sample_list:
        tile_id, batch_id = tile_sample_dict['tile_id'], tile_sample_dict['batch_id']
        tab_ncols, tab_nrows = tile_sample_dict['tab_ncols'], tile_sample_dict['tab_nrows']
        metadata_dict = {
            'num_tabs': (tab_ncols, tab_nrows),
            'tile_size': (280, 280),
            'frame_size': (294, 294),
        }
        metadata = json.dumps(metadata_dict)
        sql = 'UPDATE tile_sample SET metadata = ? WHERE tile_id = ? AND batch_id = ?'
        db_tools.update(DETECT_DBFILE, sql, (metadata, tile_id, batch_id,))
        sql = 'SELECT metadata FROM tile_sample'
        df = db_tools.query(DETECT_DBFILE, sql)
        print(df)
        
# function to change database table definitions based on release 0.1.0 to a new release
def run_update_table_definitions_010():
    sql = 'ALTER TABLE tile_sample_detect_stat RENAME COLUMN coral_object_count TO coral_alive_count'
    db_tools.update(DETECT_DBFILE, sql)
    sql = 'ALTER TABLE tile_sample_detect_stat RENAME COLUMN dead_coral_object_count TO coral_dead_count'
    db_tools.update(DETECT_DBFILE, sql)
    sql = 'ALTER TABLE tile_sample_detect_stat RENAME COLUMN other_object_count TO other_count'
    db_tools.update(DETECT_DBFILE, sql)        
    sql = 'ALTER TABLE tile_sample_detect_stat RENAME COLUMN yaml_data TO stat_yaml'
    db_tools.update(DETECT_DBFILE, sql)     
    sql = 'ALTER TABLE yolo_model DROP COLUMN coral_classes'
    db_tools.update(DETECT_DBFILE, sql)     
    sql = 'ALTER TABLE yolo_model DROP COLUMN dead_coral_classes'
    db_tools.update(DETECT_DBFILE, sql)     
    sql = 'ALTER TABLE yolo_model ADD COLUMN classes_map_yaml text DEFAULT NULL'
    db_tools.update(DETECT_DBFILE, sql)     

# function to add the predict_params_yaml and keep_object_filter_yaml columns added after release 0.1.0
def run_update_table_definitions_add_yolo_columns():
    sql = 'ALTER TABLE yolo_model ADD COLUMN predict_params_yaml text DEFAULT NULL'
    db_tools.update(DETECT_DBFILE, sql)
    sql = 'ALTER TABLE yolo_model ADD COLUMN keep_object_filter_yaml text DEFAULT NULL'
    db_tools.update(DETECT_DBFILE, sql)

# ------------------------------------------------
# The main program for running a command line 
# program for executing sql statements
if __name__ == '__main__':
    run_db()
    # run_update_tile_sample()
    # run_update_table_definitions_010()
