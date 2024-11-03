import cv2
import numpy as np


def merge_contours(contours: list[np.ndarray], gap: int = 50) -> list[np.ndarray]:
    """
    Merges a list of contours based on proximity, combining contours that are
    close horizontally within a specified gap distance.

    The function processes each contour from left to right based on the
    x-coordinate of their bounding boxes. Contours are merged if the horizontal
    distance between them is less than or equal to the specified `gap` value.

    Parameters:
        contours (list[np.ndarray]): A list of contours, where each contour is
                                        represented as an ndarray of points. Each
                                        contour is an array of shape (N, 1, 2)
                                        where N is the number of points in the
                                        contour.
        gap (int, optional): The maximum horizontal distance between contours to
                                allow merging. Contours with an x-coordinate
                                difference less than or equal to this value will
                                be merged into a single contour. Defaults to 50.

    Returns:
        list[np.ndarray]: A list of merged contours. Each merged contour combines
                            neighboring contours that were within the specified
                            gap distance from each other.
    
    Notes:
    - If `contours` is empty, an empty list is returned.
    """

    # Sort contours by their x-coordinate, left-to-right
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    
    # Initialize list to store merged contours
    merged_contours = []

    # Return early if no contours are provided
    if len(contours) < 1:
        return merged_contours
    
    # Initialize the first contour as the current contour for merging
    current_contour = contours[0]
    
    for contour in contours[1:]:
        # Get bounding boxes for the current and next contour
        x1, y1, w1, h1 = cv2.boundingRect(current_contour)
        x2, y2, w2, h2 = cv2.boundingRect(contour)
        
        # Check if contours are close enough horizontally and vertically disjoint for merging
        if (x2 - (x1 + w1) <= gap) and ((y1 + h1 < y2) or (y1 > y2 + h2)):
            # Merge contours by stacking points
            current_contour = np.vstack((current_contour, contour))
        else:
            # Finalize the current contour and move to the next
            merged_contours.append(current_contour)
            current_contour = contour
    
    # Append the last contour after looping
    merged_contours.append(current_contour)
    
    return merged_contours


def filter_contour_points(contour: np.ndarray) -> np.ndarray:
    """
    Filters out points in the contour that are more than 50 pixels away 
    from the weighted average x-coordinate. If two consecutive points have an
    x-distance greater than 50 pixels, the point furthest from the average 
    is removed, and the process repeats until no consecutive points exceed
    the 50-pixel x-distance threshold.

    Parameters:
        contour (np.ndarray): An array of shape (N, 1, 2), where each inner array
                                represents a point (x, y).

    Returns:
        np.ndarray: A filtered array of points that meet the distance criteria.
    """

    while True:
        # Extract x and y coordinates
        x_coords = contour[:, 0, 0]
        y_coords = contour[:, 0, 1]
        
        # Calculate vertical distances between consecutive points
        vertical_distances = np.abs(np.diff(y_coords))
        
        # Calculate weighted average x-coordinate
        weighted_average_x = np.average(x_coords[:-1], weights=vertical_distances)
        
        # Create a mask for points within 50 pixels of the weighted average x
        mask = np.abs(x_coords - weighted_average_x) <= 50
        contour = contour[mask]

        # Re-calculate x_coords after filtering
        x_coords = contour[:, 0, 0]
        
        # Check for any consecutive points with x-distance > 50
        x_diffs = np.abs(np.diff(x_coords))
        
        # Find indices of consecutive points with x-distance > 50
        problematic_indices = np.where(x_diffs > 50)[0]
        
        # If no problematic pairs are found, break the loop
        if len(problematic_indices) == 0:
            break
        
        # Remove the point furthest from the weighted average x in problematic pairs
        for idx in problematic_indices[::-1]:  # Reverse to avoid re-indexing issues
            if np.abs(x_coords[idx] - weighted_average_x) > np.abs(x_coords[idx + 1] - weighted_average_x):
                contour = np.delete(contour, idx, axis=0)
            else:
                contour = np.delete(contour, idx + 1, axis=0)
                
    return contour


def find_column_lines(image: np.ndarray) -> list[np.ndarray]:
    """
    Detects and extracts contours representing vertical column lines from an
    image, specifically targeting narrow and tall regions that likely indicate
    lines seperating columns.

    Parameters:
        image (np.ndarray): A BGR image array (height, width, 3) representing the
                            input image from which column lines will be detected.

    Returns:
        list[np.ndarray]: A list of filtered contours, where each contour
                            represents a vertical column line. Each contour
                            is a NumPy array of points defining the line.

    Notes:
    ------
    - This function assumes the input image contains structured text or elements 
      aligned in vertical columns.
    - Functions `remove_horizontal_clusters`, `merge_contours`, and 
      `filter_contour_points` are used to refine the contours further, ensuring
      only relevant vertical structures are retained.
    - Adjust `min_contour_height` or `w < 200` as needed to fine-tune detection
      for different document layouts.
    """

    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Edge detection to find contours
    edges = cv2.Canny(gray, 100, 200)

    # Find contours from edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize list to store valid column-like contours
    filtered_contours = []
    
    # Define minimum contour height as 5% of image height to filter out small contours
    min_contour_height = image.shape[0] * 0.05

    for contour in contours:
        # Get bounding box for the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Filter based on bounding box dimensions to identify vertical column shapes
        if h > min_contour_height and w < 200:
            filtered_contours.append(contour)

    # Remove horizontal noise from the detected contours
    vertical_contours = remove_horizontal_clusters(filtered_contours, 20, 10)
    vertical_contours = remove_horizontal_clusters(vertical_contours, 20, 20)

    # Merge close vertical contours
    merged_contours = merge_contours(vertical_contours)

    # Final filtering of contour points to clean up the output
    filtered_contours = []
    for contour in merged_contours:
        filtered_contours.append(filter_contour_points(contour))
    
    return filtered_contours


def find_horizontal_lines(image: np.ndarray) -> list[np.ndarray]:
    """
    Detects and extracts contours representing horizontal lines in an image, 
    typically aimed at identifying separators or section dividers within 
    document layouts.

    Parameters:
        image (np.ndarray): A BGR image array (height, width, 3) from which
                            horizontal lines will be detected.

    Returns:
        list[np.ndarray]: A list of sorted contours, each representing a detected
                            horizontal line in the image. Contours are sorted by
                            their y-coordinate (top to bottom).

    Notes:
    ------
    - This function is designed to work with structured layouts, such as document 
      pages with visible line dividers.
    - `min_contour_width` and `max_contour_width` can be adjusted based on the 
      typical width of horizontal lines in the document layout.

    """

    # Convert to grayscale for simpler processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Perform edge detection using Canny edge detector
    edges = cv2.Canny(blur, 50, 150, apertureSize=3, L2gradient=True)

    # Find contours in the edge image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize list to store valid horizontal line contours
    filtered_contours = []
    
    # Define minimum and maximum contour width based on image width to target horizontal lines
    min_contour_width = image.shape[1] * 0.20
    max_contour_width = image.shape[1] * 0.6

    for contour in contours:
        # Get bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Filter based on contour dimensions to detect horizontal lines
        if min_contour_width <= w <= max_contour_width and h < 50:
            filtered_contours.append(contour)

    # Sort contours by y-coordinate (top to bottom)
    sorted_contours = sorted(filtered_contours, key=lambda c: cv2.boundingRect(c)[1])

    return sorted_contours


def remove_horizontal_clusters(contours: list[np.ndarray],
                               horizontal_span_threshold: int,
                               vertical_span_threshold: int) -> list[np.ndarray]:
    """
    Filters out horizontal clusters within contours based on specified 
    thresholds, effectively removing portions of contours that span widely 
    in the horizontal direction but have minimal vertical variation.

    Parameters:
        contours (list[np.ndarray]): A list of contours (each as an array of
                                        points) from which horizontal clusters
                                        will be removed.
    
        horizontal_span_threshold (int): The maximum allowed horizontal span
                                            (difference in x-coordinates) for 
                                            a cluster to be retained. Clusters
                                            with a wider span are filtered out.
    
        vertical_span_threshold (int): The maximum allowed vertical span
                                        (difference in y-coordinates) for a
                                        cluster to be retained. Clusters exceeding
                                        this threshold are kept, as they may
                                        represent vertical structures.

    Returns:
        list[np.ndarray]: A list of filtered contours, with horizontal clusters
                            that exceed the given thresholds removed.
    
    Notes:
    ------
    - The `horizontal_span_threshold` and `vertical_span_threshold` should be 
      chosen based on the specific image characteristics, as they control the 
      type and extent of horizontal structures to be filtered.
    - `filtered_contours` will contain contours with large horizontal clusters removed, 
      which can be useful in eliminating unwanted horizontal noise.
    """

    # Initialize the list to store filtered contours
    filtered_contours = []

    for contour in contours:
        # Temporary list to hold filtered points for the current contour
        filtered_contour = []
        cluster = [contour[0]]  # Start with the first point as a new cluster

        # Iterate over the remaining points in the contour
        for i in range(1, len(contour)):
            current_point = contour[i][0]
            prev_point = contour[i - 1][0]
            
            # Check if the current point is close to the previous one vertically
            if abs(current_point[1] - prev_point[1]) <= vertical_span_threshold:
                cluster.append(contour[i])  # Add point to the current cluster
            else:
                # Calculate the x-span and y-span of the cluster
                x_coords = [point[0][0] for point in cluster]
                y_coords = [point[0][1] for point in cluster]
                x_span = max(x_coords) - min(x_coords)
                y_span = max(y_coords) - min(y_coords)

                # Discard wide, flat clusters based on span thresholds
                if not (x_span > horizontal_span_threshold and y_span <= vertical_span_threshold):
                    filtered_contour.extend(cluster)  # Add valid points to contour

                # Start a new cluster with the current point
                cluster = [contour[i]]

        # Final cluster check at the end of the contour
        x_coords = [point[0][0] for point in cluster]
        y_coords = [point[0][1] for point in cluster]
        x_span = max(x_coords) - min(x_coords)
        y_span = max(y_coords) - min(y_coords)

        if not (x_span > horizontal_span_threshold and y_span <= vertical_span_threshold):
            filtered_contour.extend(cluster)

        # Add filtered contour if it has more than one point
        if len(filtered_contour) > 1:
            filtered_contours.append(np.array(filtered_contour, dtype=np.int32))

    return filtered_contours
