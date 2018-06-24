import cv2 as cv
import numpy as np
from skimage.feature import hog


def find_bounds(image):
    height, width = image.shape
    image_area = height * width

    # Find all the contours in the image
    image, contours, hier = cv.findContours(
        image.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    # Convert back to BGR space
    bounded = cv.cvtColor(image, cv.COLOR_GRAY2BGR)

    # Find the outer most rectangle
    for contour in contours:

        # find the contour approximation
        epsilon = 0.02 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epsilon, True)
        expected_bounding_area = 0.5 * image_area

        # If it's a relatively large rectangle and convex
        if len(approx) == 4 and abs(cv.contourArea(approx)) > expected_bounding_area and cv.isContourConvex(approx):

            # draw it on and return the bounds
            color = (0, 0, 255)
            thickness = 3
            cv.drawContours(bounded, [approx], 0, color, thickness)
            return (approx, bounded)
    
    # Otherwise, there are no bounding rectangles
    return None, bounded


def preprocess_image(image):

    # Grayscale the image
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Blur the image
    blurred_image = cv.GaussianBlur(gray_image, (7, 7), 0)

    # Adaptive threshold the image
    threshold_image = cv.adaptiveThreshold(
        blurred_image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 3, 2)

    # Dilate to fill in gaps
    dilatation_size = 4
    dilation_element = cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (dilatation_size + 1, dilatation_size+1), (dilatation_size, dilatation_size))
    threshold_image = cv.dilate(threshold_image, dilation_element)


    return threshold_image


def find_numbers(image, bounds):

    # If there is a bounding recangle
    if bounds is not None:

        # Construct a region of interest inside the rectangle
        black = np.zeros((image.shape[0], image.shape[1], 1), np.uint8)
        cv.drawContours(black, [bounds], 0, 255, cv.FILLED)
        image = cv.bitwise_and(black, image)

    # Find all the contours in the image or roi
    image, contours, hier = cv.findContours(
        image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Figure out what each contour is
    numbers = []

    for contour in contours:

        # Bound each contour in a rectangle
        rect = cv.boundingRect(contour)
        contour_perim_length = 0.1 * min(image.shape[0], image.shape[1])

        # If the countour is relatively long
        if cv.arcLength(contour, True) > contour_perim_length and contour_is_within_bounds(rect, bounds, image):
            numbers.append(rect)
            
    return numbers

def contour_is_within_bounds(rectangle, bounds, image):
    height, width = image.shape[0], image.shape[1]
    x, y, w, h = rectangle

    # If there is a bounding rectangle and the contour is not fully within bounds 
    if bounds is not None and (not cv.pointPolygonTest(bounds, (x, y), False) > 0 or not cv.pointPolygonTest(bounds, (x + w, y + h), False) > 0):
        return False
    
    # If there is no bounding rectangle and the contour is not fully in the image
    if bounds is None and (x < 0 or y < 0 or x + w > width or y + h > height):
        return False

    # Otherwise the contour is within bounds
    return True



def preprocess_number(roi, length, dilate_width):

    # Resize the region of interest
    roi = cv.resize(roi, (length, length), interpolation=cv.INTER_AREA)

    # Dilate the roi
    roi = cv.dilate(roi, (dilate_width, dilate_width))

    # Convert to histogram of oriented gradients
    roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(
        14, 14), cells_per_block=(1, 1), block_norm='L1')

    # Convert hog to np.array
    roi_hog_fd = np.array([roi_hog_fd], 'float64')
    return roi_hog_fd
