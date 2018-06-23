import cv2 as cv
import helpers
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

image = cv.resize(image, (0, 0), fx=resize_factor, fy=resize_factor, interpolation=cv.INTER_AREA)

processed_image = helpers.preprocess_image(image)
bounds, edged = helpers.find_edges(processed_image)

numbers = helpers.find_numbers(processed_image, bounds)

for number in numbers:
    x, y, w, h = number
    length = int(w * 2)
    bound_x = int(x + w // 2 - length // 2)
    bound_y = int(y + h // 2 - length // 2)
    bound_w = bound_x + length
    bound_h = bound_y + length

    # find region of interest (roi)
    roi = processed_image[bound_y:bound_h, bound_x:bound_w]

    # if the region of interst has a width or height
    if roi.any():

        # preprocess the number
        roi_hog_fd = preprocessor.transform(helpers.preprocess_number(roi, 28, 3))

        # classify the number
        number = classifier.predict(roi_hog_fd)

        # write the text on the original image
        cv.putText(image, str(int(number[0])), (x, y), cv.FONT_HERSHEY_DUPLEX, 2, (0, 0, 0), 3)

    # draw the rectangles
    color = (255, 0, 0)
    thickness = 5
    cv.rectangle(image, (bound_x, bound_y), (bound_w, bound_h), color, thickness)

output = cv.hconcat((image, edged))
cv.imshow('output', output)


print("=========================================")
print("Press 'q' in the image window to exit.")
print("=========================================")
while True:
    if cv.waitKey(1) & 0xFF == ord('q'):
        cv.destroyAllWindows()
        exit()