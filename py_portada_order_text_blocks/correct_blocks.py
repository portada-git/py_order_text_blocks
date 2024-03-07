import cv2
import numpy as np

import json

from block_utils import *


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

    Args:
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


def strech_blocks_horizontally(blocks: list[dict],
                               image: np.ndarray) -> np.ndarray:
    """
    Stretches the blocks horizontally within the image and
    fills them with red color.

    Args:
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
        cv2.rectangle(image_with_horizontal_blocks, (20, y+10),
                      (img_width-20, y+h-10), (0, 0, 255), cv2.FILLED)

    return image_with_horizontal_blocks


def strech_blocks_vertically(blocks: list[dict],
                             image: np.ndarray) -> np.ndarray:
    """
    Stretches the blocks vertically within the image and
    fills them with red color.

    Args:
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
        cv2.rectangle(image_with_vertical_blocks, (x+20, y-12),
                      (x+w-20, y+h+25), (0, 0, 255), cv2.FILLED)

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

    Args:
        blocks (list[dict]): A list of dictionaries representing blocks.
        image (np.ndarray): Input image represented as a NumPy array.

    Returns:
        list[np.ndarray]: A list of numpy arrays representing horizontal
                          contours, sorted from top to bottom.
    """

    image_with_horizontal_blocks = strech_blocks_horizontally(blocks, image)

    isolated_image = isolate_red(image_with_horizontal_blocks)

    # Find horizontal contours and sorts them from top to bottom
    horizontal_cnts = find_contours(isolated_image, 1)

    return horizontal_cnts


def find_blocks_in_contours(blocks: list[dict], cnts: list[np.ndarray],
                            img_width: int,
                            img_height: int) -> list[list[dict]]:
    """
    Finds blocks that are mostly contained within each contour.

    Args:
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

    Args:
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

    Note:
        - Horizontal contours need to be sorted from top to bottom.
        - Within each horizontal contour, vertical contours are sorted from
          left to right.
    """

    vertical_cnts_array = []
    for i, blocks in enumerate(blocks_in_horizontal_cnts):
        contour_image = strech_blocks_vertically(blocks, image)

        # Uncomment this to see the images of the vertical contours.
        # cv2.imwrite(f"vertical_cnts_{i}.jpg", contour_image)

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


def remove_thin_contours(contours_array: list[list[np.ndarray]],
                         threshold: int) -> list[list[np.ndarray]]:
    """
    Removes thin contours within a list of contours based on a threshold.

    Parameters:
        contours_array (list[list[np.ndarray]]): A list of lists, where each
                                                 inner list contains numpy
                                                 arrays representing contours.
        threshold (int): Threshold value to determine if a contour is too thin.

    Returns:
        list[list[np.ndarray]]: list[list[np.ndarray]]: A list of lists.
                                                        Each inner list
                                                        contains numpy
                                                        arrays representing
                                                        contours with thin
                                                        contours removed.

    """

    new_cnts_array = []

    for contours in contours_array:
        contours_to_remove = set()

        # Iterate through each contour to identify thin contours
        for cnt in contours:
            if is_contour_too_thin(cnt, threshold):
                contours_to_remove.add(cv2.boundingRect(cnt))

        # Remove thin contours from the list of contours
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
                           for vc2 in vertical_cnts[:i])
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

    Args:
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

    Note:
        - Horizontal contours need to be sorted from top to bottom.
        - Within each horizontal contour, vertical contours need to be sorted
          from left to right.
        - Blocks within each vertical contour are sorted from top to bottom.
    """

    aligned_blocks = []
    for i, hc in enumerate(horizontal_cnts):
        vertical_cnts = vertical_cnts_array[i]
        for vc in vertical_cnts:
            if is_contour_too_thin(vc, img_width/15):
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

    Args:
        blocks (list[dict]): A list of dictionaries representing blocks.
        image (np.ndarray): Input image represented as a NumPy array.
        horizontal_cnts (list[np.ndarray]): A list of numpy arrays
                                            representing horizontal contours.

    Returns:
        list[dict]: A list of dictionaries representing blocks with corrected
                    order based on alignment within detected contours.

    Note:
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

    vertical_cnts_array = remove_thin_contours(vertical_cnts_array,
                                               img_width/15)

    vertical_cnts_array = correct_vertical_contour_order(vertical_cnts_array)

    # Sort the blocks based on their y-coordinates
    sorted_blocks = sorted(blocks, key=lambda x: x['bounds'][1])

    aligned_blocks = find_aligned_blocks(sorted_blocks, horizontal_cnts,
                                         vertical_cnts_array, img_width,
                                         img_height)

    return aligned_blocks


def order_blocks(image_path: str, json_path: str) -> list[dict]:
    """
    Orders the blocks detected in an image based on their position.

    Parameters:
        image_path (str): Path to the image file.
        json_path (str): Path to the JSON file containing block information.

    Returns:
        list[dict]: A list of dictionaries representing blocks in
                    reading order.

    """

    # Retrieve blocks from JSON
    blocks = get_blocks_from_json(json_path)

    # Load the image
    image = cv2.imread(image_path)
    img_height, img_width, _ = image.shape

    # Filter blocks
    # Remove overlapping blocks
    filtered_blocks = remove_overlapping_blocks(blocks, img_width,
                                                img_height)
    # Remove artifacts from the blocks
    filtered_blocks = remove_artifacts(filtered_blocks)

    # Find horizontal contours and sorts them from top to bottom
    horizontal_cnts = find_horizontal_contours(filtered_blocks, image)

    # Correct block order based on horizontal contours
    corrected_blocks = correct_block_order(filtered_blocks, image,
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

    # Write the updated data to a new JSON file
    with open(f"Jsons/updated_{YEAR}.json", 'w') as json_file:
        json.dump(corrected_blocks, json_file, indent=4)


if __name__ == "__main__":
    main()
