import os
import cv2
import numpy as np
from .reorder_image_from_arcanum_blocks import order_blocks, draw_numbered_blocks, stack_blocks
from .decrypt import decrypt_file_openssl
from newspaper_segmentation_client import run_newspaper_segmentation


class PortadaRedrawImageForOcr(object):
    def __init__(self, input_path='', pconfig=None):
        self._config = None
        if len(input_path) > 0:
            self._image_path = input_path
            self._image = cv2.imread(input_path)
        else:
            self.image = None
            self._image_path = ''
        self.config = pconfig

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, val):
        self._image = val

    @property
    def image_path(self):
        return self._image_path

    @image_path.setter
    def image_path(self, val):
        self._image_path = val
        self.image = cv2.imread(val)

    @property
    def config(self):
        # if self._config is None:
        #     #search config
        #     self.config = PortadaRedrawImageForOcr.__get_config_content()
        return self._config

    @image_path.setter
    def config(self, val):
        self._config = val

    def __verify_image(self):
        if self.image is None:
            raise Exception("Error: Image is not specified.")

    # @staticmethod
    # def __get_config_content():
    #     #Directory global /etc/

    def save_image(self, image_path=''):
        """
        Save the image from 'self.image' to 'image_path'. By default, image_path is equal to 'self.image_path'
        :param image_path: the image path where save the image
        :return: None
        """
        self.__verify_image()
        if len(image_path) == 0:
            image_path = self.image_path
        cv2.imwrite(image_path, self.image)

    def process_image(self):
        self.__verify_image()
        # get json blocks from arcanum
        arcanum_json = self.get_arcanum_blocks()
        corrected_blocks = order_blocks(self.image, arcanum_json)
        # self.image = draw_numbered_blocks(corrected_blocks, self.image)
        self.image = stack_blocks(self.image, corrected_blocks)
    def get_arcanum_blocks(self):
        decrypt_key = os.environ['MUNACRA_TERCES']
        arcanum_key = decrypt_file_openssl(self._config['arcanum_key_path'], decrypt_key)
        return run_newspaper_segmentation(self.image_path, arcanum_key)

