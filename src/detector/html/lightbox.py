# Copyright 2024 - Andrew Kwok Fai LUI, 
# Robotics and Autonomous Systems Group, REF, RI
# and the Queensland University of Technology

__author__ = 'Andrew Lui'
__copyright__ = 'Copyright 2024'
__license__ = 'GPL'
__version__ = '1.0'
__email__ = 'ak.lui@qut.edu.au'
__status__ = 'Development'

import os, math, random, re
import numpy as np
import cv2
from collections import defaultdict

class LightboxHelper():
    
    # replace the placeholders in the html string based on the mapping specified in rep_dict
    @staticmethod
    def _multiple_replace(html, rep_dict):
        pattern = re.compile("|".join([re.escape(k) for k in sorted(rep_dict, key=len, reverse=True)]), flags=re.DOTALL)
        return pattern.sub(lambda x: rep_dict[x.group(0)], html)

    @staticmethod
    def generate_single_image_lightbox(save_html_to:str, image_url:str):
        parent_folder = os.path.dirname(__file__)
        with open(os.path.join(parent_folder, 'lightbox_single_image.html'), 'r') as file:
            html = file.read()
        html = html.replace('[[IMAGE_URL]]', image_url)
        with open(save_html_to, 'w') as file:
            file.write(html)

    @staticmethod
    def generate_multi_images_lightbox(save_html_to:str, image_dict_list:list, page_title:str=None, metadata_dict_list:list=None):
        """ generate html for displaying images in a lightbox, 

        :param save_html_to: the target folder where the html is saved
        :param image_dict_list: a list of dict that contains at least the keys 'src', and 'title'
        :param metadata_dict_list: optional list of dict that contains metadata
        """
        parent_folder = os.path.dirname(__file__)
        with open(os.path.join(parent_folder, 'lightbox_multi_image.html'), 'r') as file:
            html = file.read()
        # generate the item list in the call to nanogallery2 jquery call
        item_list = []
        for image_dict in image_dict_list:
            src = image_dict['src']
            title = image_dict['title']
            if metadata_dict_list is None:
                item = f"{{src: '{src}', title: '{title}'}},\r\n"
            else:         
                customData = f"{title}-meta" 
                item = f"{{src: '{src}', title: '{title}', customData: '{customData}'}},\r\n"
            item_list.append(item)
        # replace the placeholder in the html template
        html = html.replace('[[JQUERY_ITEMS]]', ''.join(item_list))
        # replace the placeholder title
        page_title = 'Image Browser' if page_title is None else page_title
        html = html.replace('[[PAGE TITLE]]', page_title)
        # generate the metadata modal popup for each image if metadata_dict_list is not None
        if metadata_dict_list:
            metadata_modal_list = []
            for image_dict, metadata_dict in zip(image_dict_list, metadata_dict_list):
                title = image_dict['title']
                customData = f"{title}-meta"
                metadata_table_html = LightboxHelper._dict_to_html(metadata_dict)
                metadata_modal_html = LightboxHelper._generate_image_metadata_html(customData, metadata_table_html)
                metadata_modal_list.append(metadata_modal_html)
            html = html.replace('[[MODAL_DIV_LIST]]', '\r\n'.join(metadata_modal_list))
        else:
            html = html.replace('[[MODAL_DIV_LIST]]', '')
        with open(save_html_to, 'w') as file:
            file.write(html) 

    # generate the html part displaying image metadata
    @staticmethod
    def _generate_image_metadata_html(html, customData, metadata_table_html):
        div_template = '''
    <div class="reveal large" id="[[METADATA_ID]]" data-reveal>
      <h3>Image Metadata</h3>
      [[METADATA_TABLE]]
      <button class="close-button" data-close aria-label="Close" type="button">
        <span aria-hidden="true">&times;</span>
      </button>
    </div>
    '''
        rep_dict = {'[[METADATA_ID]]': customData, '[[METADATA_TABLE]]': metadata_table_html} 
        html = LightboxHelper._multiple_replace(div_template, rep_dict) 
        return html

    # generate html from python dict
    @staticmethod
    def _dict_to_html(data:dict):
        html_table = '{}{}<table><tbody><tr></tr><tr></tr></tbody></table>'.format(''.join(f'{key}' for key in data.keys()), ''.join(f'{value}' for value in data.values()))
        return html_table
    
    @staticmethod
    def generate_message_page(save_html_to: str, page_title: str, message: str):
        parent_folder = os.path.dirname(__file__)
        with open(os.path.join(parent_folder, 'landing_view_image.html'), 'r') as file:
            html = file.read()
        html = html.replace('[[PAGE TITLE]]', page_title)
        html = html.replace('[[PAGE CONTENT]]', f'<p>{message}</p>')
        with open(save_html_to, 'w') as file:
            file.write(html)

    # generate the landing page html that shows a list of links
    @staticmethod
    def generate_landing_page(save_html_to:str, link_dict_list:list, page_title:str):
        parent_folder = os.path.dirname(__file__)
        with open(os.path.join(parent_folder, 'landing_view_image.html'), 'r') as file:
            html = file.read()
        page_title = 'About the Tile Sample' if page_title is None else page_title
        html = html.replace('[[PAGE TITLE]]', page_title)
        link_list = []
        for link_dict in link_dict_list:
            href = link_dict['href']
            text = link_dict['text']
            link_list.append(f'<p><a href="{href}">{text}</a></p>'
                             )
        html = html.replace('[[PAGE CONTENT]]', ''.join(link_list))
        with open(save_html_to, 'w') as file:
            file.write(html)   
