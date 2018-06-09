import cv2 as cv
import numpy as np
from skimage.feature import hog


def preprocess_image(image):
    threshold = 120
    max_threshold = 255

    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred_image = cv.GaussianBlur(gray_image, (5, 5), 0)
    ret, threshold_image = cv.threshold(blurred_image, threshold, max_threshold, cv.THRESH_BINARY_INV)
    return threshold_image

def find_numbers(image):
    image_height, image_width = image.shape

    # find all the contours in the image
    image, contours, hier = cv.findContours(image.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # bound the contours with rectangles
    rects = [cv.boundingRect(contour) for contour in contours]

    # limit only rectangles not touching the edge and store their heights
    numbers = []
    heights_of_numbers = []
    for rect in rects:
        x, y, w, h = rect
        if x > 0 and y > 0 and x + w < image_width and y + h < image_height:
            heights_of_numbers.append(h)
            numbers.append(rect)

    # find average and standard deviation of number heights
    average_height_of_numbers = np.average(heights_of_numbers)
    std_height_of_numbers = np.std(heights_of_numbers)

    # limit only rectangles within 1 standard deviation in height from rest of rectangles
    numbers = [number for number in numbers if number[3] < average_height_of_numbers + std_height_of_numbers and number[3] > average_height_of_numbers - std_height_of_numbers]
    return numbers

def preprocess_number(roi, length, dilate_width):
    roi = cv.resize(roi, (length, length), interpolation=cv.INTER_AREA)
    roi = cv.dilate(roi, (dilate_width, dilate_width))
    roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), block_norm='L1')
    roi_hog_fd = np.array([roi_hog_fd], 'float64')
    return roi_hog_fd