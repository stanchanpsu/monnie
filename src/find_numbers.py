import cv2 as cv
import helpers
from sklearn.externals import joblib
import argparse

parser = argparse.ArgumentParser(description='Find numbers.')
parser.add_argument('image', metavar='i', type=str, help='path to image')

RESIZED_IMAGE_LENGTH = 500

image = cv.imread(parser.parse_args().image)
image = cv.resize(image, (RESIZED_IMAGE_LENGTH, RESIZED_IMAGE_LENGTH), interpolation=cv.INTER_AREA)
processed_image = helpers.preprocess_image(image)

model_file = "digits_cls.pkl"
classifier, preprocessor = joblib.load(model_file)

numbers = helpers.find_numbers(processed_image)

for number in numbers:
    margin = 5
    x, y, w, h = number
    bound_x = x - margin
    bound_y = y - margin
    bound_w = x + w + margin
    bound_h = y + h + margin

    # draw the rectangles
    color = (255, 0, 0)
    thickness = 5
    cv.rectangle(image, (bound_x, bound_y), (bound_w, bound_h), color, thickness)

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

processed_image = cv.cvtColor(processed_image, cv.COLOR_GRAY2BGR)
output = cv.hconcat((image, processed_image))
cv.imshow('output', output)


print("=========================================")
print("Press 'q' in the image window to exit.")
print("=========================================")
while True:
    if cv.waitKey(1) & 0xFF == ord('q'):
        cv.destroyAllWindows()
        exit()