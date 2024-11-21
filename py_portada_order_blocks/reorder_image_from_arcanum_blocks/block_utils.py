import json
from typing import Tuple, Union, Optional
import re
from datetime import date
import copy

import cv2
import numpy as np


def get_blocks_from_json(json_file_path: Union[str, dict]) -> list[dict]:
    """
    Extracts blocks from a JSON file containing article data.

    Parameters:
        json_file_path (Union[str, dict]): The file path to the JSON file containing the data.

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
    if 'articles' not in data:
        return data
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
                        threshold: float = 0.4) -> bool:
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
                                     containment. Defaults to 0.4.

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

    if rect1_area <= 0:
        return False

    # Check if more than the threshold of the rect1 area
    # is within the rect2 area
    return intersection_area / rect1_area > threshold


def is_contour_above(contour1: Tuple[int, int, int, int], contour2: Tuple[int, int, int, int]) -> bool:
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


def merge_overlapping_blocks(blocks: list[dict], img_width: int, img_height:int,
                             threshold: float = 0.5) -> list[dict]:
    """
    Merges blocks that are mostly contained within other blocks
    from the list of blocks.

    Parameters:
        blocks (list[dict]): A list of dictionaries representing blocks.
        img_width (int): The width of the image with the blocks.
        img_height (int): The height of the image with the blocks.
        threshold (float, optional): Threshold value for determining
                                     overlap. Defaults to 0.5.

    Returns:
        list[dict]: A list of dictionaries representing blocks with
                    overlapping blocks merged.
    """
    while True:
        merged_blocks = []
        merged_indices = set()  # Track merged blocks to avoid duplicates
        merged = False  # Flag to track if any merge occurs

        for i in range(len(blocks)):
            if i in merged_indices:
                continue
            
            current_block = blocks[i]
            
            # Iterate over other blocks to check for overlaps
            for j in range(i + 1, len(blocks)):
                if j in merged_indices:
                    continue

                # Get coordinates for the blocks
                current_block_bounds = get_block_coordinates(current_block, img_width, img_height)
                other_block = blocks[j]
                other_block_bounds = get_block_coordinates(other_block, img_width, img_height)

                # Check if blocks overlap
                if (is_mostly_contained(current_block_bounds, other_block_bounds, threshold)
                    or is_mostly_contained(other_block_bounds, current_block_bounds, threshold)):
                    
                    # Merge the current block with the overlapping block
                    current_block = merge_two_blocks(current_block, other_block)
                    
                    merged_indices.add(j)  # Mark other block as merged
                    merged = True  # Set flag to indicate a merge occurred

            # Add the resulting (possibly merged) block to the list
            merged_blocks.append(current_block)

        # If no merges happened in this pass, we’re done
        if not merged:
            break
        
        # Update blocks with the newly merged set for the next iteration
        blocks = merged_blocks

    return merged_blocks


def merge_two_blocks(block1: dict, block2: dict) -> dict:
    """
    Merges two blocks by creating a bounding box that fully encloses both blocks.

    Parameters:
    block1 (dict): A dictionary representing the first block.

    block2 (dict): A dictionary representing the second block.

    Returns:
        dict: A new dictionary representing the merged block, with updated
                bounding coordinates. The `bounds` key contains the merged
                area as [left, top, right, bottom] values, and the `label` key
                is set to `"merged"` to indicate that the block is a result 
                of merging.
    """

    merged_block = copy_block(block1)
    # Merge two blocks by creating a bounding box that covers both blocks
    merged_block['bounds'][0] = min(block1['bounds'][0], block2['bounds'][0])
    merged_block['bounds'][1] = min(block1['bounds'][1], block2['bounds'][1])
    merged_block['bounds'][2] = max(block1['bounds'][2], block2['bounds'][2])
    merged_block['bounds'][3] = max(block1['bounds'][3], block2['bounds'][3])
    merged_block['label'] = "merged"
    return merged_block


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


def extract_date_page_and_newspaper(filename: str) -> Optional[Tuple[date, int, str]]:
    """
    Extracts the date, page number, and newspaper name from the filename.

    This function parses a filename to extract the publication date, page number, and
    newspaper name based on a specific naming convention.

    Parameters:
        filename (str): The filename containing the date, page number,
                            and newspaper name information.

    Returns:
        Optional[Tuple[date, int, str]]: A tuple containing the date, page number,
                                            and newspaper name if the pattern
                                            matches, or None if the filename
                                            format does not match the expected
                                            pattern.
    """
    # Define the pattern for matching the filename and capturing the newspaper name
    pattern = r'(\d{4})_(\d{2})_(\d{2})_([A-Z]+_[A-Z]+)_(\d+).*'

    # Match the filename against the pattern
    match = re.match(pattern, filename)

    if match:
        year = int(match.group(1))
        month = int(match.group(2))
        day = int(match.group(3))
        newspaper_name = match.group(4)
        page_number = int(match.group(5))

        file_date = date(year, month, day)

        return file_date, page_number, newspaper_name
    else:
        return None
    

def is_point_on_block_edge(point: Tuple[int, int],
                           block_bounds: Tuple[int, int, int, int]) -> bool:
    """
    Determines if a point is located on the top or bottom edge of a given rectangular block.

    This function checks whether a specified point lies exactly on either the top or bottom
    horizontal edge of a rectangular block defined by its bounding box coordinates.

    Parameters:
        point (Tuple[int, int]): The (x, y) coordinates of the point to check.
        block_bounds (Tuple[int, int, int, int]): A tuple representing the bounding
                                                    box of the block, defined as
                                                    (x, y, width, height).

    Returns:
        bool: True if the point lies on the top or bottom edge of the block;
                otherwise, False.

    """

    block_x, block_y, block_width, block_height = block_bounds
    point_x, point_y = point

    return (block_x <= point_x <= (block_x + block_width) and
            (point_y == block_y or point_y == (block_y + block_height)))



def find_line_intersection(segment1: Tuple[int, int, int, int],
                           segment2: Tuple[int, int, int, int]
                           ) -> Optional[Tuple[int, int]]:
    """
    Finds the intersection point of two line segments, if it exists.

    This function calculates the intersection point of two line segments given by their
    endpoints. If the segments are parallel or do not intersect within the defined segment
    boundaries, the function returns None.

    Parameters:
        segment1 (Tuple[int, int, int, int]): A tuple representing the first line
                                                segment's endpoints in the format
                                                (x1, y1, x2, y2).
        segment2 (Tuple[int, int, int, int]): A tuple representing the second line
                                                segment's endpoints in the format
                                                (x3, y3, x4, y4).

    Returns:
        Optional[Tuple[int, int]]: A tuple (x, y) representing the intersection
                                    point of the two segments, or None if the
                                    segments do not intersect within their finite
                                    lengths or are parallel.

    Notes:
    - The function uses parametric equations to determine if the segments intersect within
      their bounds.
    - `ua` and `ub` are the parametric variables for the segments; if either is out of
      the range [0, 1], it indicates the intersection lies outside the segments.
    """

    # Unpack segment points
    x1, y1, x2, y2 = segment1
    x3, y3, x4, y4 = segment2

    # Calculate the denominator for the intersection formula
    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if denom == 0:  # Parallel segments
        return None

    # Calculate the intersection parameters
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    if ua < 0 or ua > 1:  # Intersection is outside segment1
        return None

    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
    if ub < 0 or ub > 1:  # Intersection is outside segment2
        return None

    # Calculate the exact intersection point
    x = x1 + ua * (x2 - x1)
    y = y1 + ua * (y2 - y1)
    return (x, y)


def is_valid_block(block_bounds: Tuple[int, int, int, int]) -> bool:
    """
    Checks if a block is valid based on its width and height.

    Parameters:
        block_bounds (Tuple[int, int, int, int]): A tuple representing the
                                                    bounding box of the block in
                                                    the format (x, y, width, height).

    Returns:
        bool: True if both width and height are greater than zero.
                False if either width or height is zero or negative.
    """

    # Unpack width and height
    _, _, width, height = block_bounds

    # Check for positive width and height
    return width > 0 and height > 0


def split_block_if_intersect(block_bounds: Tuple[int, int, int, int], 
                             column_line: list[Tuple[int, int]]
                             ) -> list[Tuple[int, int, int, int]]:
    """
    Splits a block into two parts if a column line intersects it. 
    
    This function determines if a given column line intersects the top and bottom
    edges of a block. If two intersection points are found, the block is split
    into left and right portions based on these points. The function only performs
    the split if both resulting blocks are valid rectangles with a width that is
    at least 20% of the original block’s width.

    Parameters:
        block_bounds (Tuple[int, int, int, int]): A tuple representing the
                                                    bounding box of the block in
                                                    the format (x, y, width, height).
        column_line (List[Tuple[int, int]]): A list of points representing a line
                                                that may intersect the block. Each
                                                point in the list is a tuple
                                                (x, y) representing a contour
                                                along the line.

    Returns:
        list[Tuple[int, int, int, int]]: A list of one or two bounding boxes. If
                                            the block is split, two new bounding
                                            boxes are returned, each covering a
                                            portion of the original block. If no
                                            split occurs, the function returns the
                                            original block as a single bounding
                                            box in a list.

    Notes:
    - The function uses helper functions `find_line_intersection` to identify
        intersection points and `is_point_on_block_edge` to verify that these
        points fall on the block’s top or bottom edges.
    """

    # Unpack the block coordinates
    x, y, w, h = block_bounds

    # Define the block's top and bottom edges
    block_edges = [
        (x, y, x + w, y),        # Top edge
        (x, y + h, x + w, y + h)  # Bottom edge
    ]

    intersection_points = []

    # Check intersections with each line segment in the column line
    for i in range(len(column_line) - 1):
        line = (column_line[i][0][0], column_line[i][0][1], column_line[i + 1][0][0], column_line[i + 1][0][1])
        for block_edge in block_edges:
            intersection = find_line_intersection(line, block_edge)
            if intersection and is_point_on_block_edge(intersection, block_bounds):
                intersection_points.append(intersection)

    # Check for sufficient intersection points
    if len(intersection_points) < 2:
        return [block_bounds]  # No intersection or only one point found

    # Sort intersection points based on x-coordinates for left and right block determination
    intersection_points = np.array(intersection_points)
    intersection_points = intersection_points[np.argsort(intersection_points[:, 0])]

    # Create coordinates for the two new blocks
    new_block1_bounds = (x, y, intersection_points[0][0], y + h)  # Left block
    new_block2_bounds = (intersection_points[1][0], y, x + w, y + h)  # Right block

    # Minimum width for each new block (20% of original width)
    min_width = 0.2 * w

    # Validate both new blocks for non-zero area and sufficient width
    if not (is_valid_block(new_block1_bounds) and is_valid_block(new_block2_bounds)):
        return [block_bounds]  # Return original block if either new block is invalid

    if (new_block1_bounds[2] - new_block1_bounds[0] < min_width
        or new_block2_bounds[2] - new_block2_bounds[0] < min_width):
        return [block_bounds]  # Return original block if either new block is too narrow

    # Return the two split blocks if they are both valid
    return [new_block1_bounds, new_block2_bounds]


def copy_block(block: dict) -> dict:
    """
    Creates a deep copy of a block dictionary, ensuring that all data structures
    within the block are duplicated rather than referenced.

    This function iterates over each key-value pair in the `block` dictionary and
    applies `deepcopy` to each value. This ensures that nested objects (like lists,
    arrays, or dictionaries) are fully copied, allowing modifications to the copy
    without affecting the original `block`.

    Parameters:
        block (dict): A dictionary representing a block.

    Returns:
        dict: A new dictionary that is an independent, deep copy of the input
                `block`. All nested structures are fully duplicated to avoid
                references to the original data.
    """

    # Create an empty dictionary for the copied block
    block_copy = {}

    # Iterate through each key-value pair in the original block and deep copy each value
    for key, value in block.items():
        block_copy[key] = copy.deepcopy(value)

    return block_copy
