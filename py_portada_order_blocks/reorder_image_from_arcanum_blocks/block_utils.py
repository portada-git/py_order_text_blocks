import json
from typing import Tuple
from typing import Union

import cv2
import numpy as np


def get_blocks_from_json(json_file_path: Union[str, dict]) -> list[dict]:
    """
    Extracts blocks from a JSON file containing article data.

    Parameters:
        json_file_path (str): The file path to the JSON file containing the data.

    Returns:
        list[dict]: A list of dictionaries representing the extracted blocks.
    """

    if type(json_file_path) is str:
        # Load the JSON data
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)
    else:
        data = json_file_path

    # Extract all the blocks from the articles
    all_blocks = []
    for article in data['articles']:
        all_blocks.extend(article['blocks'])

    return all_blocks


def get_block_coordinates(block: dict, img_width: int,
                          img_height: int) -> Tuple[float, float,
                                                    float, float]:
    """
    Calculate the coordinates of a block within an image.

    Parameters:
        block (dict): A dictionary representing the block with 'bounds'
                      information.
        img_width (int): The width of the image with the blocks.
        img_height (int): The height of the image with the blocks.

    Returns:
        Tuple[float, float, float, float]: A tuple representing the
                                           coordinates of the block
                                           (xmin, ymin, width, height).
    """

    bounds = block['bounds']
    xmin = int(bounds[0] * img_width)
    ymin = int(bounds[1] * img_height)
    xmax = int(bounds[2] * img_width)
    ymax = int(bounds[3] * img_height)
    w = xmax - xmin
    h = ymax - ymin
    return xmin, ymin, w, h


def get_block_label(block: dict) -> str:
    """
    Extracts the label of a block from its dictionary representation.

    Parameters:
        block (dict): A dictionary representing the block.

    Returns:
        str: The label of the block.
    """
    return block['label']


def get_contour_coordinates(contour: np.ndarray) -> Tuple[float, float,
                                                          float, float]:
    """
    Calculate the coordinates of a contour's bounding rectangle.

    Parameters:
        contour (np.ndarray): An array representing the contour.

    Returns:
        Tuple[float, float, float, float]: A tuple representing the
                                           coordinates of the bounding
                                           rectangle (x, y, width, height).
    """

    x, y, w, h = cv2.boundingRect(contour)
    return x, y, w, h


def is_contour_too_thin(contour: np.ndarray, threshold: int) -> bool:
    """
    Checks if a contour is too thin based on a given threshold.

    Parameters:
        contour (np.ndarray): An array representing the contour.
        threshold (int): The threshold value to compare the contour's width
                         against.

    Returns:
        bool: True if the contour is too thin, False otherwise.
    """

    _, _, w, _ = get_contour_coordinates(contour)
    return w < threshold


def is_mostly_contained(rect1: Tuple[float, float, float, float],
                        rect2: Tuple[float, float, float, float],
                        threshold: float = 0.75) -> bool:
    """
    Checks if rect1 is mostly contained within rect2.

    Parameters:
        rect1 (Tuple[float, float, float, float]): Coordinates
                                                   (x, y, width, height)
                                                   of the first rectangle.
        rect2 (Tuple[float, float, float, float]): Coordinates
                                                   (x, y, width, height)
                                                   of the second rectangle.
        threshold (float, optional): Threshold value for determining
                                     containment. Defaults to 0.75.

    Returns:
        bool: True if rect1 is mostly contained within rect2, False otherwise.
    """

    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    # Calculate the intersection rectangle
    intersection_rect = [max(x1, x2), max(y1, y2), min(x1 + w1, x2 + w2),
                         min(y1 + h1, y2 + h2)]

    # Calculate the area of intersection rectangle
    intersection_area = (max(0, intersection_rect[2] - intersection_rect[0]) *
                         max(0, intersection_rect[3] - intersection_rect[1]))

    # Calculate the area of rect1
    rect1_area = w1 * h1

    # Check if more than the threshold of the rect1 area
    # is within the rect2 area
    return intersection_area / rect1_area > threshold


def is_contour_above(contour1, contour2):
    """
    Check if contour1 is above contour2.

    Parameters:
        contour1: tuple
            The bounding rectangle coordinates of contour1 (x, y, w, h).
        contour2: tuple
            The bounding rectangle coordinates of contour2 (x, y, w, h).

    Returns:
        bool:
            True if contour1 is above contour2, False otherwise.
    """
    _, y1, _, h1 = cv2.boundingRect(contour1)
    _, y2, _, _ = cv2.boundingRect(contour2)
    return y1 + h1 < y2


def remove_tables_from_blocks(blocks: list[dict]) -> list[dict]:
    """
    Removes blocks labeled as 'Table' and title blocks followed by
    tables from the list of blocks.

    Parameters:
        blocks (list[dict]): A list of dictionaries representing blocks.

    Returns:
        list[dict]: A list of dictionaries representing blocks
                    with tables removed.
    """

    corrected_blocks = []
    for i, block in enumerate(blocks):
        # Skip blocks labeled as 'Table'
        if get_block_label(block) == 'Table':
            continue
        # Skip title blocks followed by tables
        elif (get_block_label(block) == 'Title' and i+1 < len(blocks)
                and get_block_label(blocks[i+1]) == 'Table'):
            continue
        # Append non-table blocks to the list of corrected blocks
        corrected_blocks.append(block)
    return corrected_blocks


def remove_overlapping_blocks(blocks: list[dict], img_width: int,
                              img_height: int) -> list[dict]:
    """
    Removes blocks that are mostly contained within other blocks
    from the list of blocks.

    Parameters:
        blocks (list[dict]): A list of dictionaries representing blocks.

    Returns:
        list[dict]: A list of dictionaries representing blocks with
                    overlapping blocks removed.
    """

    blocks_to_remove = []
    # Identify overlapping blocks
    for i, block1 in enumerate(blocks):
        for j, block2 in enumerate(blocks):
            if i != j and is_mostly_contained(get_block_coordinates(block1,
                                                                    img_width,
                                                                    img_height
                                                                    ),
                                              get_block_coordinates(block2,
                                                                    img_width,
                                                                    img_height
                                                                    ),
                                              threshold=0.5):
                blocks_to_remove.append(i)

    # Remove the identified blocks
    filtered_blocks = [block for i, block in enumerate(blocks)
                       if i not in blocks_to_remove]

    return filtered_blocks


def remove_artifacts(blocks: list[dict]) -> list[dict]:
    """
    Removes blocks labeled as 'Artifact' from the list of blocks.

    Parameters:
        blocks (list[dict]): A list of dictionaries representing blocks.

    Returns:
        list[dict]: A list of dictionaries representing blocks
                    with artifacts removed.
    """

    # Remove blocks labeled as 'Artifact'
    filtered_blocks = [block for block in blocks
                       if get_block_label(block) != 'Artifact']

    return filtered_blocks
