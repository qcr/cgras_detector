# Copyright 2024 - Andrew Kwok Fai LUI, 
# Robotics and Autonomous Systems Group, REF, RI
# and the Queensland University of Technology

__author__ = 'Andrew Lui'
__copyright__ = 'Copyright 2024'
__license__ = 'GPL'
__version__ = '1.0'
__email__ = 'ak.lui@qut.edu.au'
__status__ = 'Development'

import os, shutil, psutil, random
import pandas as pd
import torch
# dash modules
import dash
from dash import html, dcc, Input, Output, State, dash_table, ctx
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
# project modules
from detector.model import STATE, COORDINATOR_STATE, SystemStates, APP_FILE_MANAGER

class MonitorStateBlock():
    def __init__(self, app, prefix):
        self.app = app 
        self.prefix = prefix = prefix + 'msb'
        self.update_store_id = prefix + 'update_store'
        # model variables
        self.banner_message = ''
        self.current_detect_message = None
        # the tables
        _left_datatable = dash_table.DataTable(id=prefix+'left_table', editable=False, row_selectable=False, cell_selectable=False, style_cell={'fontSize': 14})
        _right_datatable = dash_table.DataTable(id=prefix+'right_table', editable=False, row_selectable=False, cell_selectable=False, style_cell={'fontSize': 14})
        # define widgets 
 
        self._panel = dbc.Row([
            html.H4(dbc.Badge('SYSTEM RESOURCES', className='ms-2 mb-4', color='white', text_color='secondary')),
            dcc.Store(id=self.update_store_id),
            dbc.Col([_left_datatable], className='col-5'),
            dbc.Col([_right_datatable], className='col-7'),
            html.P(' ', id=prefix+'state_message', className='mt-3 text-center'),
        ], className='mx-auto text-center')

        self.app.callback([Output(self.prefix+'state_message', 'children')],
                        [Input(self.update_store_id, 'data')])(self._update_state_message())        
        
        self.app.callback([Output(self.prefix+'left_table', 'data'),
                           Output(self.prefix+'right_table', 'data'),],
                        [Input(self.update_store_id, 'data')])(self._update_table_content())          

    def get_panel(self):
        return self._panel
    
    def register_trigger(self, trigger_id:str):
        # define callbacks for the datatable data
        self.app.callback([Output(self.update_store_id, 'data'),],
                            [Input(trigger_id, 'data')], prevent_initial_call=True, allow_duplicate=True)(self._update_panel())
        
    def _process_button_pressed(self):
        def process_button_pressed(button):
            return (True, 0,)
        return process_button_pressed
    
    def _update_panel(self):
        def update_panel(data):
            return (data,)
        return update_panel
    
    def _update_state_message(self):
        def update_state_message(data):
            state:SystemStates = STATE.get()
            previous_state:SystemStates = STATE.get_previous_state()
      
            if state in [SystemStates.READY, SystemStates.AUTO_START, SystemStates.CLICK_START]:
                self.current_detect_message = None
                if previous_state not in [SystemStates.READY, SystemStates.AUTO_START, SystemStates.CLICK_START] or random.random() < 0.2:
                    self.banner_message = [
                        'Waiting for coral babies to count.',
                        'Counting corals to assess their health and save our time.',
                        'Reef restoration for healthy oceans.'
                    ][random.randrange(0, 3)]
            elif state in [SystemStates.POLL_DETECT, SystemStates.POLL_IMPORT_SAMPLE]:
                if previous_state not in [SystemStates.POLL_DETECT, SystemStates.POLL_IMPORT_SAMPLE] or random.random() < 0.2:
                    self.banner_message = [
                        'Can we correlate the spatial distribution of corals to benthic community factors?',
                        'Making large-scale reef restoration a little bit more viable.',
                        'How can we improve coral growth?'
                    ][random.randrange(0, 3)]
            elif state in [SystemStates.SUSPENDED]:
                self.current_detect_message = None
                self.banner_message = 'I am on leave now as the system is suspended.'
            else:
                if self.current_detect_message is None:
                    self.current_detect_message = [
                        'Busy counting corals.',
                        'It used to take 1-2 hours to manually count a single coral tile.',
                        'Using AI to count tiny corals automatically.'
                    ][random.randrange(0, 3)]
                self.banner_message = self.current_detect_message
            return (self.banner_message,)
        return update_state_message 

    # generate the model for the left status table display
    def _define_right_model(self):
        # query cpu percent and memory percent
        cpu_count = psutil.cpu_count()
        cpu_count_physical = psutil.cpu_count(logical=False)
        cpu_percent = psutil.cpu_percent()
        mem_total = psutil.virtual_memory().available
        mem_percent = psutil.virtual_memory().percent    
        model = pd.DataFrame(columns=('System Status', ''))
        model.loc[1] = ['CPU Count', f'{cpu_count} ({cpu_count_physical})']
        model.loc[2] = ['CPU Usage', f'{cpu_percent:.1f} %']
        model.loc[3] = ['Virtual Memory', f'{mem_total / (2**30):.2f} GB'] 
        model.loc[4] = ['Virtual Memory Usage', f'{mem_percent:.1f} %']      
        total, used, free = APP_FILE_MANAGER.get_free_disk_space()
        free_diskspace_percent = free * 100 / total if total > 0 else 0
        model.loc[5] = ['Free/Total Disk Space', f'{free:.1f}/{total:.1f} GB ({free_diskspace_percent:.1f} %)']
        return model

    # generate the model for the left status table display
    def _define_left_model(self):
        # obtain the states of the two platforms
        state:SystemStates = STATE.get()
        previous_state:SystemStates = STATE.get_previous_state()
        capturer_state = COORDINATOR_STATE.get()   
        model = pd.DataFrame(columns=('Other Status', ''))
        model.loc[1] = ['CCVS Status', f'{state.name} ']
        model.loc[2] = ['IAS Status', f'{capturer_state.name}']
        if not torch.cuda.is_available():
            model.loc[3] = ['GPU/CUDA', f'Unavailable']
        else:
            device_count = torch.cuda.device_count()
            model.loc[3] = ['GPU/CUDA', f'Available']
            model.loc[4] = ['# GPU Devices', f'{device_count}']
            for index in range(device_count):
              model.loc[5 + index] = [f'', f'{torch.cuda.get_device_name(index)}']
        return model    

    def _update_table_content(self):
        def update_table_content(data):
            left_model = self._define_left_model()
            right_model = self._define_right_model()
            return (left_model.to_dict('records'), right_model.to_dict('records'),)
        return update_table_content

    # the callback for table update
    # def _update_content_bar(self):
    #     def update_content_bar(data):
    #         # obtain the states of the two platforms
    #         state:SystemStates = STATE.get()
    #         previous_state:SystemStates = STATE.get_previous_state()
    #         capturer_state = COORDINATOR_STATE.get()
    #         # query cpu percent and memory percent
    #         cpu_percent = psutil.cpu_percent()
    #         mem_percent = psutil.virtual_memory().percent        
    #         if state in [SystemStates.READY, SystemStates.AUTO_START, SystemStates.CLICK_START]:
    #             self.current_detect_message = None
    #             if previous_state not in [SystemStates.READY, SystemStates.AUTO_START, SystemStates.CLICK_START] or random.random() < 0.2:
    #                 self.banner_message = [
    #                     'I am consuming electricity but there is no coral babies to look after. Can you feel my guilt?',
    #                     'I am wasting my talent here. Got nothing to do. I should be nursing coral babies. ',
    #                     'I turn into a couch potato. Is it my destiny?'
    #                 ][random.randrange(0, 3)]
    #         elif state in [SystemStates.POLL_DETECT, SystemStates.POLL_IMPORT_SAMPLE]:
    #             if previous_state not in [SystemStates.POLL_DETECT, SystemStates.POLL_IMPORT_SAMPLE] or random.random() < 0.2:
    #                 self.banner_message = [
    #                     'I just asked my supervisor for more coral larvae but got nothing. I am feeling insecure.',
    #                     'No new coral again! Should I look elsewhere for coral babies?',
    #                     'Can we make corals more productive? '
    #                 ][random.randrange(0, 3)]
    #         elif state in [SystemStates.SUSPENDED]:
    #             self.current_detect_message = None
    #             self.banner_message = 'I am on leave now as the system is suspended. Ping me if you want but I am not reading messages.'
    #         else:
    #             if self.current_detect_message is None:
    #                 self.current_detect_message = [
    #                     'Leave me alone. I am working hard to keep large and small corals happy in the playpen.',
    #                     'The corals are noisy. No time to entertain you now.',
    #                     'Shhh! The corals are sleeping and their tentacles are being counted.'
    #                 ][random.randrange(0, 3)]
    #             self.banner_message = self.current_detect_message
    #         return (state.name, capturer_state.name, cpu_percent, f'{cpu_percent} %', mem_percent, f'{mem_percent} %', 
    #                 self.banner_message,)
    #     return update_content_bar 