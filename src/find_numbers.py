import cv2 as cv
import helpers
from colors import *
from CurrencyConverter import CurrencyConverter
from sklearn.externals import joblib
import argparse

parser = argparse.ArgumentParser(description='Find numbers.')
parser.add_argument('image', metavar='i', type=str, help='path to image')

model_file = "digits_cls.pkl"
classifier, preprocessor = joblib.load(model_file)

RESIZED_IMAGE_WIDTH = 800
image = cv.imread(parser.parse_args().image)
height, width, channels = image.shape
resize_factor = RESIZED_IMAGE_WIDTH/width

image = cv.resize(image, (0, 0), fx=resize_factor,
                  fy=resize_factor, interpolation=cv.INTER_AREA)

processed_image = helpers.preprocess_image(image)
bounds, bounded_image = helpers.find_bounds(processed_image)

numbers = helpers.find_numbers(processed_image, bounds)

entire_number = {}

for number in numbers:
    x, y, w, h = number
    side = int(max(w, h) * 1.2)
    x1 = int(x + w // 2 - side // 2)
    y1 = int(y + h // 2 - side // 2)
    x2 = x1 + side
    y2 = y1 + side

    # find region of interest (roi)
    roi = processed_image[y1:y2, x1:x2]

    # if the region of interst has a width or height
    if roi.any():

        # preprocess the number
        roi_hog_fd = preprocessor.transform(
            helpers.preprocess_number(roi, 28, 3))

        # classify the number
        number = classifier.predict(roi_hog_fd)

        # write the text on the original image
        number = str(int(number[0]))
        cv.putText(image, number, (x, y),
                   cv.FONT_HERSHEY_DUPLEX, 2, LIGHT_SEA_GREEN, 3)

        # draw the rectangles
        thickness = 5
        cv.rectangle(image, (x1, y1),
                    (x2, y2), GREEN, thickness)

        entire_number[x] = number

entire_amount = helpers.get_entire_amount(entire_number)

print(str(entire_amount) + ' USD')
to_currency_amount = CurrencyConversion.convert_currency(entire_amount, 'EUR')
print(to_currency_amount)

output = cv.hconcat((image, bounded_image))
cv.imshow('output', output)


print("=========================================")
print("Press 'q' in the image window to exit.")
print("=========================================")
while True:
    if cv.waitKey(1) & 0xFF == ord('q'):
        cv.destroyAllWindows()
        exit()
