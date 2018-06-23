import cv2 as cv
import numpy as np
from skimage.feature import hog

def find_edges(image):

    # Find all the contours in the image
    image, contours, hier = cv.findContours(image.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    # Convert back to BGR space
    edged = cv.cvtColor(image, cv.COLOR_GRAY2BGR)

    # Find the outer most rectangle
    for contour in contours:
        epsilon = 0.02 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4 and abs(cv.contourArea(approx)) > 1000 and cv.isContourConvex(approx):
            color = (0, 0, 255)
            thickness = 3
            cv.drawContours(edged, [approx], 0, color, thickness)
            return (approx, edged)
    return None, edged

def preprocess_image(image):

    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred_image = cv.GaussianBlur(gray_image, (7, 7), 0)
    threshold_image = cv.adaptiveThreshold(blurred_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,                                             cv.THRESH_BINARY_INV,3,2)
    dilatation_size = 2
    dilation_element = cv.getStructuringElement(cv.MORPH_RECT, (2*dilatation_size + 1, 2*dilatation_size+1), (dilatation_size, dilatation_size))
    threshold_image = cv.dilate(threshold_image, dilation_element)
    return threshold_image

def find_numbers(image, bounds):

    if bounds is not None:

        # construct roi
        black = np.zeros((image.shape[0], image.shape[1], 1), np.uint8)
        cv.drawContours(black, [bounds], 0, 255, cv.FILLED)
        image = cv.bitwise_and(black, image)

    # find all the contours in the image
    image, contours, hier = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    numbers = []
    for contour in contours:
        rect = cv.boundingRect(contour)
        x, y, w, h = rect
        if cv.arcLength(contour, True) > 100:
            if bounds is not None and (not cv.pointPolygonTest(bounds, (x, y), False) > 0 or not cv.pointPolygonTest(bounds, (x + w, y + h), False) > 0):
                    continue
            numbers.append(rect)
    return numbers

def preprocess_number(roi, length, dilate_width):
    roi = cv.resize(roi, (length, length), interpolation=cv.INTER_AREA)
    roi = cv.dilate(roi, (dilate_width, dilate_width))
    roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), block_norm='L1')        
    roi_hog_fd = np.array([roi_hog_fd], 'float64')
    return roi_hog_fd