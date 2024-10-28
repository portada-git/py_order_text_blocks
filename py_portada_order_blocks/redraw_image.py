import os
import cv2

from .reorder_image_from_arcanum_blocks import order_blocks, stack_blocks
from .decrypt import decrypt_file_openssl
from newspaper_segmentation_client import run_newspaper_segmentation_on_image
from py_image_utils.image_utilities_cv import resize_image_percent_til_size, convert_ordered_block_stack_to_cv2
from pathlib import Path
from PIL import Image


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
        self.image_blocks = []

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, val):
        self._image = val
        self.image_blocks = []

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

    @config.setter
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

    def save_block_images(self, dir_name="", image_name=""):
        self.__verify_image()
        for bi in self.image_blocks:
            if len(image_name) == 0:
                image_name = bi["file_name"]

            if len(dir_name) > 0:
                image_path = Path(dir_name).joinpath(Path(image_name).stem)
            else:
                image_path = Path(image_name)

            with open("{file_name}_{count:03d}{extension}".format(file_name=image_path, count=bi["count"],
                                                                   extension=bi["extension"]), "wb") as bf:
                bf.write(bi["image"])

    def pre_process_image(self, image, size, ext=""):
        if len(ext)==0:
            img, _, _, _ = resize_image_percent_til_size(image, size)
        else:
            img, _, _, _ = resize_image_percent_til_size(image, size, ext)
        return img

    def process_image(self):
        self.__verify_image()
        # get json blocks from arcanum
        ext = Path(self.image_path).suffix
        if len(ext) == 0:
            ext = ".jpg"
        image = self.pre_process_image(self.image, 6291456, ext)
        arcanum_json = self.get_arcanum_blocks(image)
        corrected_blocks = order_blocks(self.image, arcanum_json)
        # corrected_blocks = post_process(corrected_blocks, proc)
        # self.image = draw_numbered_blocks(corrected_blocks, self.image)
        blocks = convert_ordered_block_stack_to_cv2(self.image, corrected_blocks)
        count = 0
        file_name = Path(self.image_path).stem
        for block in blocks:
            self.image_blocks.append(
                dict(file_name=file_name, extension=ext, count=count, image=cv2.imencode(ext, block)[1]))
            count = count + 1

    def get_arcanum_blocks(self, image=None):
        decrypt_key = os.environ['MUNACRA_TERCES']
        arcanum_key = decrypt_file_openssl(self._config['arcanum_key_path'], decrypt_key)
        if image is None:
            image = self.image

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pil_img = Image.fromarray(image)

        return run_newspaper_segmentation_on_image(pil_img, arcanum_key)
