import os
import sys
import cv2
import numpy as np

import json

from .block_utils import *
from .line_utils import *


"""
USE EXAMPLE:

from py_portada_order_text_blocks import correct_blocks

corrected_blocks = order_blocks(IMAGE_PATH, JSON_DATA)

input_image = cv2.imread(IMAGE_PATH)

# Draw the blocks in the correct order
image_with_blocks = draw_numbered_blocks(corrected_blocks, input_image)
cv2.imwrite("image_with_blocks_path", image_with_blocks)

# Write the updated data to a new JSON file
with open("updated_json_path, 'w') as json_file:
    json.dump(corrected_blocks, json_file, indent=4)
"""


def isolate_red(image: np.ndarray) -> np.ndarray:
    """
    Isolates red components in the input image.

    Parameters:
        image (np.ndarray): Input image represented as a NumPy array.
    Returns:
        np.ndarray: Image with only red components isolated.
    """
    # Create a copy of the image
    isolated_image = image.copy()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range of red color in HSV
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    # Threshold the HSV image to get only red colors
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Bitwise-AND mask and original image
    isolated_image[mask == 0] = [255, 255, 255]  # Set non-red pixels to white

    return isolated_image


def draw_numbered_blocks(blocks: list[dict], image: np.ndarray) -> np.ndarray:
    """
    Draws rectangles around each block in the image and numbers them based
    on their order.

    Parameters:
        blocks (list[dict]): A list of dictionaries representing blocks.
        image (np.ndarray): Input image represented as a NumPy array.

    Returns:
        np.ndarray: A copy of the input image with rectangles drawn around
        each block and numbers added based on order.
    """

    image_with_blocks = image.copy()
    img_height, img_width, _ = image.shape
    # Draw rectangles and add numbers for each block
    for i, block in enumerate(blocks, start=1):
        x, y, w, h = get_block_coordinates(block, img_width, img_height)
        cv2.rectangle(image_with_blocks, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(image_with_blocks, str(i), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

    return image_with_blocks


def stack_blocks(image: np.ndarray, blocks: list[dict]) -> np.ndarray:
    """
    Creates an image with the text blocks stacked vertically.

    Args:
        blocks (list[dict]): A list of dictionaries representing blocks.
        image (np.ndarray): Input image represented as a NumPy array.

    Returns:
        np.ndarray: An image with the blocks stacked vertically.
    """

    img_height, img_width, _ = image.shape
    
    cut_blocks = []
    max_width = 0
    
    for block in blocks:
        x1, y1, w, h = get_block_coordinates(block, img_width, img_height)
        
        # Calculate the bottom-right corner coordinates of each block
        x2 = x1 + w
        y2 = y1 + h

        # Cut the text block region from the image
        cut_block = image[y1:y2, x1:x2]

        # Track the maximum width of the blocks
        if cut_block.shape[1] > max_width:
            max_width = cut_block.shape[1]
        
        # Append the cut block to the list
        cut_blocks.append(cut_block)
    
    # Resize all blocks to have the same width as the widest block
    resized_blocks = []
    for block in cut_blocks:
        if block.shape[1] < max_width:
            padding = max_width - block.shape[1]
            block = cv2.copyMakeBorder(block, 0, 0, 0, padding, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        resized_blocks.append(block)

    # Stack the cut blocks vertically
    stacked_image = np.vstack(resized_blocks)
    
    return stacked_image


def stretch_blocks_horizontally(blocks: list[dict],
                               image: np.ndarray) -> np.ndarray:
    """
    Stretches the blocks horizontally within the image and
    fills them with red color.

    Parameters:
        blocks (list[dict]): A list of dictionaries representing blocks.
        image (np.ndarray): Input image represented as a NumPy array.

    Returns:
        np.ndarray: A copy of the input image with blocks stretched
        horizontally and filled with red color.
    """

    image_with_horizontal_blocks = image.copy()
    img_height, img_width, _ = image.shape
    # Stretch horizontally and fill with red color
    for block in blocks:
        _, y, _, h = get_block_coordinates(block, img_width, img_height)
        cv2.rectangle(image_with_horizontal_blocks, (20, y+4),
                      (img_width-20, y+h-4), (0, 0, 255), cv2.FILLED)

    return image_with_horizontal_blocks


def stretch_blocks_vertically(blocks: list[dict],
                             image: np.ndarray) -> np.ndarray:
    """
    Stretches the blocks vertically within the image and
    fills them with red color.

    Parameters:
        blocks (list[dict]): A list of dictionaries representing blocks
        image (np.ndarray): Input image represented as a NumPy array.

    Returns:
        np.ndarray: A copy of the input image with blocks stretched
        vertically and filled with red color.
    """

    image_with_vertical_blocks = image.copy()
    img_height, img_width, _ = image.shape
    # Stretch vertically and fill with red color
    for block in blocks:
        x, y, w, h = get_block_coordinates(block, img_width, img_height)
        cv2.rectangle(image_with_vertical_blocks, (x+int(w/7), 20),
                      (x+w-int(w/7), img_height), (0, 0, 255), cv2.FILLED)

    return image_with_vertical_blocks


def find_contours(image: np.ndarray, sorting_mode: int) -> list[np.ndarray]:
    """
    Finds contours in the input image and sorts them based on the specified
    mode.

    Parameters:
        image (np.ndarray): Input image represented as a NumPy array.
        sorting_mode (int): Sorting mode for contours.
                            0 for horizontal sorting, 1 for vertical sorting.

    Returns:
        list[np.ndarray]: list of contours sorted based on the specified mode.
    """

    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding
    thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV +
                           cv2.THRESH_OTSU)[1]

    # Find contours
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours based on the specified mode
    cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[sorting_mode])

    return cnts


def find_horizontal_contours(blocks: list[dict],
                             image: np.ndarray) -> list[np.ndarray]:
    """
    Finds horizontal contours within the image after stretching blocks
    horizontally and filling them with red color.

    Parameters:
        blocks (list[dict]): A list of dictionaries representing blocks.
        image (np.ndarray): Input image represented as a NumPy array.

    Returns:
        list[np.ndarray]: A list of numpy arrays representing horizontal
                          contours, sorted from top to bottom.
    """

    image_with_horizontal_blocks = stretch_blocks_horizontally(blocks, image)

    isolated_image = isolate_red(image_with_horizontal_blocks)

    # Find horizontal contours and sorts them from top to bottom
    horizontal_cnts = find_contours(isolated_image, 1)

    return horizontal_cnts


def find_blocks_in_contours(blocks: list[dict], cnts: list[np.ndarray],
                            img_width: int,
                            img_height: int) -> list[list[dict]]:
    """
    Finds blocks that are mostly contained within each contour.

    Parameters:
        blocks (list[dict]): A list of dictionaries representing blocks.
        cnts (list[np.ndarray]): A list of numpy arrays representing contours
                                 detected in the image.
        img_width (int): The width of the image in which the blocks and
                           contours are located.
        img_height (int): The height of the image in which the blocks and
                            contours are located.

    Returns:
        list[list[dict]]: A list of lists, where each inner list contains
                          dictionaries representing blocks that are mostly
                          contained within each contour.
    """

    blocks_in_cnts = []
    for cnt in cnts:
        blocks_within_contour = []
        for block in blocks:
            if is_mostly_contained(get_block_coordinates(block, img_width,
                                                         img_height),
                                   get_contour_coordinates(cnt)):
                blocks_within_contour.append(block)
        blocks_in_cnts.append(blocks_within_contour)
    return blocks_in_cnts


def find_vertical_contours_array(blocks_in_horizontal_cnts: list[list[dict]],
                                 image: np.ndarray) -> list[list[np.ndarray]]:
    """
    Finds vertical contours within each contour where horizontal blocks
    are located.

    Parameters:
        blocks_in_horizontal_cnts (list[list[dict]]): A list of lists, where
                                                      each inner list
                                                      represents blocks
                                                      contained within a
                                                      horizontal contour.
        image (np.ndarray): Input image represented as a NumPy array.

    Returns:
        list[list[np.ndarray]]: A list of lists. Each inner list contains
                                numpy arrays representing
                                vertical contours found within each of the
                                horizontal contours.

    Notes:
        - Horizontal contours need to be sorted from top to bottom.
        - Within each horizontal contour, vertical contours are sorted from
          left to right.
    """

    vertical_cnts_array = []
    for i, blocks in enumerate(blocks_in_horizontal_cnts):
        contour_image = stretch_blocks_vertically(blocks, image)

        isolated_image = isolate_red(contour_image)

        vertical_cnts = find_contours(isolated_image, 0)
        vertical_cnts_array.append(vertical_cnts)
    return vertical_cnts_array


def remove_overlaping_contours(contours_array:
                               list[list[np.ndarray]]
                               ) -> list[list[np.ndarray]]:
    """
    Removes overlapping contours within a list of contours.

    Parameters:
        contours_array (list[list[np.ndarray]]): A list of lists, where each
                                                 inner list contains numpy
                                                 arrays representing contours.

    Returns:
        list[list[np.ndarray]]: A list of lists. Each inner list contains
                                numpy arrays representing contours with
                                overlapping contours removed.

    """

    new_cnts_array = []

    for contours in contours_array:
        contours_to_remove = set()

        # Iterate through each pair of contours to identify
        # overlapping contours
        for i, cnt1 in enumerate(contours):
            for cnt2 in contours[i + 1:]:
                if is_mostly_contained(get_contour_coordinates(cnt2),
                                       get_contour_coordinates(cnt1)):
                    contours_to_remove.add(cv2.boundingRect(cnt2))

        # Remove overlapping contours from the list of contours
        new_cnts = [cnt for cnt in contours
                    if cv2.boundingRect(cnt) not in contours_to_remove]
        new_cnts_array.append(new_cnts)

    return new_cnts_array


def correct_vertical_contour_order(vertical_cnts_array:
                                   list[list[np.ndarray]]
                                   ) -> list[list[np.ndarray]]:
    """
    Corrects the vertical contour order based on their position relative
    to each other.

    Parameters:
        vertical_cnts_array (list[list[np.ndarray]]): A list of lists, where
                                                      each inner list contains
                                                      numpy arrays
                                                      representing vertical
                                                      contours within each
                                                      horizontal contour.

    Returns:
        list[list[np.ndarray]]: list[list[np.ndarray]]: A list of lists.
                                                        Each inner list
                                                        contains numpy arrays
                                                        representing vertical
                                                        contours in the
                                                        correct order.

    """

    new_vertical_cnts_array = []

    for vertical_cnts in vertical_cnts_array:
        cnts_above = []
        cnts_below = []

        # Iterate through each vertical contour to determine its position
        # relative to others
        for i, vc in enumerate(vertical_cnts):
            is_below = any(is_contour_above(vc2, vc)
                           for vc2 in vertical_cnts)
            if is_below:
                cnts_below.append(vc)
            else:
                cnts_above.append(vc)

        # Append the corrected order of contours to the
        # new vertical contour array
        new_vertical_cnts_array.append(cnts_above + cnts_below)

    return new_vertical_cnts_array


def find_aligned_blocks(blocks: list[dict], horizontal_cnts: list[np.ndarray],
                        vertical_cnts_array: list[list[np.ndarray]],
                        img_width: int, img_height: int) -> list[dict]:
    """
    Finds blocks aligned both horizontally and vertically within the detected
    contours.

    Parameters:
        blocks (list[dict]): A list of dictionaries representing blocks.
                             The blocks need to be sorted from top to bottom.
        horizontal_cnts (list[np.ndarray]): A list of numpy arrays
                                            representing horizontal contours.
        vertical_cnts_array (list[list[np.ndarray]]): A list of lists, where
                                                      each inner list contains
                                                      numpy arrays
                                                      representing vertical
                                                      contours within each
                                                      horizontal contour.
        img_width (int): The width of the image in which the blocks and
                           contours are located.
        img_height (int): The height of the image in which the blocks and
                            contours are located.

    Returns:
        list[dict]: A list of dictionaries representing blocks that are
                    aligned both horizontally and vertically within the
                    detected contours.

    Notes:
        - Horizontal contours need to be sorted from top to bottom.
        - Within each horizontal contour, vertical contours need to be sorted
          from left to right.
        - Blocks within each vertical contour are sorted from top to bottom.
    """

    aligned_blocks = []
    for i, hc in enumerate(horizontal_cnts):
        vertical_cnts = vertical_cnts_array[i]
        for vc in vertical_cnts:
            if is_contour_too_thin(vc, 0.05*img_width):
                continue
            aligned_blocks.extend([block for block in blocks
                                  if is_mostly_contained(
                                    get_block_coordinates(block, img_width,
                                                          img_height),
                                    get_contour_coordinates(hc)
                                    )
                                   and is_mostly_contained(
                                    get_block_coordinates(block, img_width,
                                                          img_height),
                                    get_contour_coordinates(vc)
                                    )])
            blocks = [x for x in blocks if x not in aligned_blocks]
    return aligned_blocks


def correct_block_order(blocks: list[dict], image: np.ndarray,
                        horizontal_cnts: list[np.ndarray]) -> list[dict]:
    """
    Corrects the order of blocks within the image based on their alignment
    within detected contours.

    Parameters:
        blocks (list[dict]): A list of dictionaries representing blocks.
        image (np.ndarray): Input image represented as a NumPy array.
        horizontal_cnts (list[np.ndarray]): A list of numpy arrays
                                            representing horizontal contours.

    Returns:
        list[dict]: A list of dictionaries representing blocks with corrected
                    order based on alignment within detected contours.

    Notes:
        - Horizontal contours are sorted from top to bottom.
        - Within each horizontal contour, vertical contours are sorted from
          left to right.
        - Blocks within each vertical contour are sorted from top to bottom.
    """

    img_height, img_width, _ = image.shape

    blocks_in_horizontal_cnts = find_blocks_in_contours(blocks,
                                                        horizontal_cnts,
                                                        img_width,
                                                        img_height)

    vertical_cnts_array = find_vertical_contours_array(
                            blocks_in_horizontal_cnts, image)

    vertical_cnts_array = remove_overlaping_contours(vertical_cnts_array)

    vertical_cnts_array = correct_vertical_contour_order(vertical_cnts_array)

    # Sort the blocks based on their y-coordinates
    sorted_blocks = sorted(blocks, key=lambda x: x['bounds'][1])

    aligned_blocks = find_aligned_blocks(sorted_blocks, horizontal_cnts,
                                         vertical_cnts_array, img_width,
                                         img_height)
    
    aligned_blocks = stretch_all_blocks(aligned_blocks, horizontal_cnts, vertical_cnts_array, img_width, img_height)

    return aligned_blocks


def split_sections(line: np.ndarray, img_width: int, img_height: int
                   ) -> list[np.ndarray]:
    """
    Splits an image into three distinct areas based on bounding rectangle of the
    contour line.

    This function takes a contour line represented as a NumPy array and the
    dimensions of an image, then computes the bounding rectangle of the contour.
    It constructs three separate contours that represent the areas of the image
    divided by the contour line.

    Parameters:
        line (np.ndarray): A NumPy array representing the coordinates of the
                            contour line.
        img_width (int): The width of the image.
        img_height (int): The height of the image.

    Returns:
        list[np.ndarray]: A list containing two NumPy arrays, each representing a
                            set of contour points that define two sections of the
                            image. The first section is to the left of the contour
                            line, and the second section is to the right. The
                            function does not return a section for the lower right
                            area as it does not contain relevant information.

    Notes:
    - The function constructs three contours based on the bounding rectangle of the provided line:
      - `cnt1`: A contour representing the area to the left of the contour line.
      - `cnt2`: A contour representing the area above the contour line to the right.
      - `cnt3`: A contour representing the area below and to the right of the contour line,
        which is not used as it does not contain relevant information.
    - The function returns only the first two contours, as `cnt3` is deemed unnecessary.
    """

    x, y, w, h = cv2.boundingRect(line)

    cnt1 = np.array([
        [0, 0],
        [x, 0],
        [x, img_height],
        [0, img_height]
    ])

    cnt2 = np.array([
        [x, 0],
        [img_width, 0],
        [img_width, y],
        [x, y]
    ])

    # This never contains information about Arrivals so we don't need it.
    cnt3 = np.array([
        [x, y],
        [img_width, y],
        [img_width, img_height],
        [x, img_height]
    ])

    return [cnt1, cnt2]


def split_wrong_blocks(blocks: list[dict], image: np.ndarray,
                       column_lines: list[np.ndarray]) -> list[dict]:
    """
    Splits blocks in an image if they intersect with specified column lines, returning a refined list of non-overlapping blocks.

    This function iteratively checks each block against a list of column lines, splitting any block that intersects with a column line. 
    For each intersection, two new blocks are created, adjusted to fit either side of the intersecting line. The process continues 
    until no further splits occur, ensuring each block remains distinct and aligned within the image boundaries.

    Parameters:
        blocks (list[dict]): A list of dictionaries representing blocks.
        image (np.ndarray): Input image represented as a NumPy array.
        column_lines (list[np.ndarray]): A list of column lines represented as NumPy arrays.
                                            Each line is defined by a set of points and serves
                                            as a boundary for splitting any intersecting blocks.

    Returns:
        list[dict]: A list of final split blocks. Each block is represented as a dictionary
        with its updated bounding box coordinates and a label field indicating whether it was split.

    Notes:
    - The function continues to split blocks until no further splits occur across any column lines, ensuring stable results.
    - Each block's coordinates are normalized with respect to the original image dimensions, allowing easy re-scaling.
    """
    
    img_height, img_width, _ = image.shape
    final_blocks = []
    has_split_occurred = True

    while has_split_occurred:  # Continue looping until no blocks are split
        has_split_occurred = False
        new_blocks = []

        for block in blocks:
            block_was_split = False

            # Check each column line for intersection with the current block
            for column_line in column_lines:
                block_bounds = get_block_coordinates(block, img_width, img_height)
                split_result = split_block_if_intersect(block_bounds, column_line)

                # If there was a split, create new blocks and mark a split occurred
                if len(split_result) > 1:
                    new_block1 = copy_block(block)
                    new_block1['bounds'][0] = split_result[0][0] / img_width
                    new_block1['bounds'][1] = split_result[0][1] / img_height
                    new_block1['bounds'][2] = split_result[0][2] / img_width
                    new_block1['bounds'][3] = split_result[0][3] / img_height
                    new_block1['label'] = "was_split"

                    new_block2 = copy_block(block)
                    new_block2['bounds'][0] = split_result[1][0] / img_width
                    new_block2['bounds'][1] = split_result[1][1] / img_height
                    new_block2['bounds'][2] = split_result[1][2] / img_width
                    new_block2['bounds'][3] = split_result[1][3] / img_height
                    new_block2['label'] = "was_split"

                    # Add newly created blocks to `new_blocks` list for further evaluation
                    new_blocks.append(new_block1)
                    new_blocks.append(new_block2)

                    # Indicate that a split occurred and break to re-evaluate from scratch
                    has_split_occurred = True
                    block_was_split = True
                    break

            # If the block wasn't split by any column line, add it to `final_blocks`
            if not block_was_split:
                final_blocks.append(block)

        # Replace `blocks` with the new split blocks for the next iteration
        blocks = new_blocks

    return final_blocks


def stretch_all_blocks(blocks: list[dict], horizontal_cnts: list[np.ndarray],
                       vertical_cnts_array: list[list[np.ndarray]],
                       img_width: int, img_height: int) -> list[dict]:
    """
    Adjusts blocks in an image by stretching or merging them to prevent information loss due to page curvature
    and preserve sentence continuity within columns.

    This function processes a list of blocks, stretching each block within a column downwards to align with the starting 
    position of the next block in the same column. This helps avoid information loss between blocks. For blocks located 
    in the first column, a leftward stretch is applied to account for possible distortion due to page curvature. Additionally, 
    consecutive blocks within the same column that are adjacent are merged to maintain sentence integrity across the 
    column, preventing unintentional breaks.

    Parameters:
        blocks (list[dict]): A list of dictionaries representing blocks.
                             The blocks need to be sorted from top to bottom.
        horizontal_cnts (list[np.ndarray]): A list of numpy arrays
                                            representing horizontal contours.
        vertical_cnts_array (list[list[np.ndarray]]): A list of lists, where
                                                      each inner list contains
                                                      numpy arrays
                                                      representing vertical
                                                      contours within each
                                                      horizontal contour.
        img_width (int): The width of the image in which the blocks and
                           contours are located.
        img_height (int): The height of the image in which the blocks and
                            contours are located.

    Returns:
        list[dict]: A list of dictionaries representing adjusted blocks with
                    updated bounding coordinates. Blocks that were merged 
                    are removed from the final list to prevent overlap.

    Notes:
    -----
    - Blocks within each column are stretched downwards to prevent information gaps between consecutive blocks.
    - Blocks in the leftmost column are slightly stretched to the left to account for information loss due to page curvature.
    - Adjacent blocks within the same column are merged if they appear contiguous, ensuring sentence continuity.
    """
        
    blocks_to_remove = []
    for hc_index, hc in enumerate(horizontal_cnts):
        vertical_cnts = vertical_cnts_array[hc_index]
        for vc_index, vc in enumerate(vertical_cnts):
            for j, block in enumerate(blocks):
                block_bounds = get_block_coordinates(block, img_width, img_height)
                hc_coords = get_contour_coordinates(hc)
                vc_coords = get_contour_coordinates(vc)
                if not is_mostly_contained(block_bounds, hc_coords):
                    continue
                if not is_mostly_contained(block_bounds, vc_coords):
                    continue
                # stretch blocks in the leftmost column to account for curvature
                if vc_index == 0 and block['bounds'][0] < 0.1:
                    block['bounds'][0] -= 0.2*block['bounds'][0]
                if j + 1 < len(blocks):
                    block2_coords = get_block_coordinates(blocks[j+1], img_width, img_height)
                    if not is_mostly_contained(block2_coords, hc_coords):
                        continue
                    if not is_mostly_contained(block2_coords, vc_coords):
                        continue
                    if (block['bounds'][3] < blocks[j+1]['bounds'][1]):
                        block['bounds'][3] = blocks[j+1]['bounds'][1]
                    if (block['bounds'][2] < blocks[j+1]['bounds'][0]):
                        block['bounds'][0] = min(block['bounds'][0], blocks[j+1]['bounds'][0])
                        block['bounds'][1] = min(block['bounds'][1], blocks[j+1]['bounds'][1])
                        block['bounds'][2] = max(block['bounds'][2], blocks[j+1]['bounds'][2])
                        block['bounds'][3] = max(block['bounds'][3], blocks[j+1]['bounds'][3])
                        blocks_to_remove.append(j+1)
                if j - 1 >= 0:
                    block2_coords = get_block_coordinates(blocks[j-1], img_width, img_height)
                    if not is_mostly_contained(block2_coords, hc_coords):
                        continue
                    if not is_mostly_contained(block2_coords, vc_coords):
                        continue
                    if (block['bounds'][3] < blocks[j-1]['bounds'][1]):
                        block['bounds'][3] = blocks[j-1]['bounds'][1]
                    if (block['bounds'][2] < blocks[j-1]['bounds'][0]):
                        block['bounds'][0] = min(block['bounds'][0], blocks[j-1]['bounds'][0])
                        block['bounds'][1] = min(block['bounds'][1], blocks[j-1]['bounds'][1])
                        block['bounds'][2] = max(block['bounds'][2], blocks[j-1]['bounds'][2])
                        block['bounds'][3] = max(block['bounds'][3], blocks[j-1]['bounds'][3])
                        blocks_to_remove.append(j-1)

    stretched_blocks = [block for i, block in enumerate(blocks)
                       if i not in blocks_to_remove]
    return stretched_blocks


def should_split_sections(newspaper: str, newspaper_date: date, 
                          num_horizontal_seg: int, 
                          image: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Determines whether to split newspaper sections based on specified criteria, 
    and identifies the relevant horizontal line if a split is needed.
    
    Parameters:
        newspaper (str): The code name of the newspaper, used to apply specific
                            rules.
        newspaper_date (date): The publication date of the newspaper, used to
                                assess if the split should occur.
        num_horizontal_seg (int): The number of horizontal segments detected in
                                    the image.
        image (np.ndarray): The newspaper image in which horizontal lines are
                            detected.
    
    Returns:
        Tuple[bool, Optional[np.ndarray]]: A tuple where:
                                            - The first element is a boolean
                                                indicating if sections should be
                                                split.
                                            - The second element is the horizontal
                                                line (if found) that indicates
                                                where the split should occur, or
                                                None if no such line is found.
    """
    # USED ONLY FOR LE SEMAPHORE DE MARSEILLE
    # The date after which the blocks of the newspaper image might need to be
    # split into left and right for better block ordering
    sm_target_date = date(1892, 3, 20)

    img_height, img_width, _ = image.shape

    # Initialize split indicator and line variable
    split_the_sections = False
    split_line = None

    # Check if the newspaper and date meet the criteria for splitting
    if newspaper == "MAR_SM" and newspaper_date > sm_target_date and num_horizontal_seg == 1:
        # Detect horizontal lines in the image
        horizontal_lines = find_horizontal_lines(image)

        # Identify a valid horizontal line for splitting based on position
        for horizontal_line in horizontal_lines:
            x, y, w, h = cv2.boundingRect(horizontal_line)
            if x > 0.4 * img_width and (0.3 * img_height <= y <= 0.8 * img_height):
                split_the_sections = True
                split_line = horizontal_line
                break

    return split_the_sections, split_line


def order_blocks(image_path: Union[str, np.ndarray], json_path: Union[str, dict]) -> list[dict]:
    """
    Orders the blocks detected in an image based on their position.

    Parameters:
        image_path (str|np.ndarray): Path to the image file or image already loaded.
        json_path (str|dict): Path to the JSON file or a dict containing block information.

    Returns:
        list[dict]: A list of dictionaries representing blocks in
                    reading order.

    """

    # Retrieve blocks from JSON
    blocks = get_blocks_from_json(json_path)

    # Load the image
    if type(image_path) is str:
        image = cv2.imread(image_path)
    else:
        image = image_path

    img_height, img_width, _ = image.shape

    # Filter blocks
    # Remove artifacts from the blocks
    filtered_blocks = remove_artifacts(blocks)

    # Merge overlapping blocks
    filtered_blocks = merge_overlapping_blocks(filtered_blocks, img_width, img_height)

    # Find horizontal contours and sorts them from top to bottom
    horizontal_cnts = find_horizontal_contours(filtered_blocks, image)

    # Find the column lines within each horizontal contour
    column_lines = find_column_lines_per_horizontal_contour(horizontal_cnts, image)

    # Split the blocks intersected by a column line
    seperated_blocks = split_wrong_blocks(filtered_blocks, image, column_lines)

    newspaper_date, page, newspaper = extract_date_page_and_newspaper(image_path)

    # Check if horizontal contours should change, this is only used for Le Semaphore
    split_the_sections, split_line = should_split_sections(newspaper, newspaper_date,
                                                           len(horizontal_cnts), image)

    # Change horizontal contours based on the split_line
    if split_the_sections:
        horizontal_cnts = split_sections(split_line, img_width, img_height)

    # Correct block order based on horizontal contours
    corrected_blocks = correct_block_order(seperated_blocks, image,
                                           horizontal_cnts)

    return corrected_blocks


def main():
    YEAR = "1850"
    JSON_DATA = f"Jsons/{YEAR}.json"
    IMAGE_PATH = f"images/{YEAR}_deskewed.jpg"

    corrected_blocks = order_blocks(IMAGE_PATH, JSON_DATA)

    image = cv2.imread(IMAGE_PATH)

    # Draw the blocks in the correct order
    copy = draw_numbered_blocks(corrected_blocks, image)
    cv2.imwrite(f"Corrected_blocks_{YEAR}.jpg", copy)

    stack_copy = stack_blocks(image, corrected_blocks)
    cv2.imwrite(f"Stacked_blocks_{YEAR}.jpg", stack_copy)

    # Write the updated data to a new JSON file
    with open(f"Jsons/updated_{YEAR}.json", 'w') as json_file:
        json.dump(corrected_blocks, json_file, indent=4)


if __name__ == "__main__":
    main()