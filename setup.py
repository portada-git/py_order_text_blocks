from setuptools import setup
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name='py_portada_order_blocks',
    version='0.0.8',
    description='tools for creating images of documents with only one column to avoid ordering problems when they are processed by OCR within the PortADa project',
    author='PortADa team',
    author_email='jcbportada@gmail.com',
    license='MIT',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/portada-git/py_portada_order_text_blocks",
    packages=['py_portada_order_blocks','py_portada_order_blocks.reorder_image_from_arcanum_blocks'],
    py_modules=['redraw_image'],
    install_requires=[ 
        'opencv-python >= 4.8,<4.9',
        'cryptography~=42.0.7',
        'numpy~=1.26.4',
        'arcanum-newspaper-segmentation-client~=1.8.4',
	'py_image_utils @ git+https://github.com/portada-git/py_image_utils#egg=py_image_utils'
    ], 
    python_requires='>=3.9',
    zip_safe=False)
