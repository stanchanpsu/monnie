import cv2
from skimage.feature import hog
from sklearn.externals import joblib
import numpy as np
import time

# cap = cv2.VideoCapture(0)
image = cv2.imread('../images/photo_of_numbers.jpg')
image = cv2.resize(image, (500, 500), interpolation=cv2.INTER_AREA)
classifier = "digits_cls.pkl"
clf, pp = joblib.load(classifier)
blank_image = np.zeros((600,800,3), np.uint8)

# while(True):
    # Capture frame-by-frame
#     ret, frame = cap.read()

# Our operations on the frame come here
im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

ret, im_th = cv2.threshold(im_gray, 140, 255, cv2.THRESH_BINARY_INV)
_, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
rects = [cv2.boundingRect(ctr) for ctr in ctrs]
print(ctrs[0])
print(rects[0])

answer = ""
avg_height = 0
total_rects = 0
for rect in rects:
    x, y, w, h = rect
    if x > 0 and y > 0 and x + w < 500 and y + h < 500:
        total_rects += 1
        avg_height += rect[3]
avg_height /= total_rects
for rect in rects:
    # Draw the rectangles
    offset = 5
    x, y, w, h = rect
    if x > 0 and y > 0 and x + w < 500 and y + h < 500 and h > avg_height / 2:
        cv2.rectangle(image, (x - offset, y - offset), (x+w +offset, y+h+offset), (255, 255, 0), 3)
        roi = im_th[y-offset:y+h+offset, x-offset:x+w+offset]
        if roi.any():
            roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            roi = cv2.dilate(roi, (3, 3))
            roi_hog_fd, hog_image = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=True, block_norm="L1")
            roi_hog_fd = pp.transform(np.array([roi_hog_fd], 'float64'))
            nbr = clf.predict(roi_COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

ret, im_th = cv2.threshold(im_gray, 140, 255, cv2.THRESH_BINARY_INV)
_, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
rects = [cv2.boundingRect(ctr) for ctr in ctrs]
print(ctrs[0])
print(rects[0])

answer = ""
avg_height = 0
total_rects = 0
for rect in rects:
    x, y, w, h = rect
    if x > 0 and y > 0 and x + w < 500 and y + h < 500:
        total_rects += 1
        avg_height += rect[3]
avg_height /= total_rects
for rect in rects:
    # Draw the rectangles
    offset = 5
    x, y, w, h = rect
    if x > 0 and y > 0 and x + w < 500 and y + h < 500 and h > avg_height / 2:
        cv2.rectangle(image, (x - offset, y - offset), (x+w +offset, y+h+offset), (255, 255, 0), 3)
        roi = im_th[y-offset:y+h+offset, x-offset:x+w+offset]
        if roi.any():
            roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            roi = cv2.dilate(roi, (3, 3))
            roi_hog_fd, hog_image = hog(roi, oCOLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

ret, im_th = cv2.threshold(im_gray, 140, 255, cv2.THRESH_BINARY_INV)
_, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
rects = [cv2.boundingRect(ctr) for ctr in ctrs]
print(ctrs[0])
print(rects[0])

answer = ""
avg_height = 0
total_rects = 0
for rect in rects:
    x, y, w, h = rect
    if x > 0 and y > 0 and x + w < 500 and y + h < 500:
        total_rects += 1
        avg_height += rect[3]
avg_height /= total_rects
for rect in rects:
    # Draw the rectangles
    offset = 5
    x, y, w, h = rect
    if x > 0 and y > 0 and x + w < 500 and y + h < 500 and h > avg_height / 2:
        cv2.rectangle(image, (x - offset, y - offset), (x+w +offset, y+h+offset), (255, 255, 0), 3)
        roi = im_th[y-offset:y+h+offset, x-offset:x+w+offset]
        if roi.any():
            roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            roi = cv2.dilate(roi, (3, 3))
            roi_hog_fd, hog_image = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=True, block_norm="L1")
            roi_hog_fd = pp.transform(np.array([roi_hog_fd], 'float64'))
            nbr = clf.predict(roi_rientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=True, block_norm="L1")
            roi_hog_fd = pp.transform(np.array([roi_hog_fd], 'float64'))
            nbr = clf.predict(roi_hog_fd)
            cv2.putText(image, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 0), 3)
            answer += str(nbr[0])

x,y,w,h = rects[0]
blank_image.fill(255)
print(answer)
cv2.putText(blank_image, answer, (x, y+h),cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 0), 10)
# cv2.imshow('rectangle', image)
im_th = cv2.cvtColor(im_th,cv2.COLOR_GRAY2BGR)
# blank_image = cv2.cvtColor(blank_image, cv2.COLOR_GRAY2BGR)
show = cv2.hconcat((image, im_th))
cv2.imshow('output', blank_image)
# show = cv2.hconcat((show, blank_image))
#     cv2.imshow('output', blank_image)
#     cv2.imshow('gray', im_gray)
cv2.imshow('show', show)

# Display the resulting frame
#     cv2.imshow('frame',gray)
#     time.sleep(.3)
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
        cv2.destroyAllWindows()
        exit()

# When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()

# image = "images/pinta.jpg"
# im = cv2.imread(image)



# Convert to grayscale and apply Gaussian filtering
# im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
