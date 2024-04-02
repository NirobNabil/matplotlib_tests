import cv2 as cv
import numpy as np

img = cv.imread('images/7.png')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

_, thresh = cv.threshold(gray, 254,255,cv.THRESH_BINARY_INV)


kernel = np.ones((5,5),np.uint8)
# img = cv.erode(img,kernel,iterations = 1)

contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
for i, c in enumerate(contours):
    if cv.contourArea(c) > 300:
        x,y,w,h = cv.boundingRect(c)
        cv.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        rect = cv.minAreaRect(c)
        box = cv.boxPoints(rect)
        cropped = img[ int(box[1][1]):int(box[2][1]), int(box[0][0]):int(box[1][0]) ]
        cv.imwrite(f"cropped-{i}.jpg", cropped)
        # print(cv.boundingRect(c))

img = cv.drawContours(img, contours, -1, (150))
cv.imshow('Image.jpg', img)
cv.waitKey(5000)
# cv.waitKey(0)

flags = [i for i in dir(cv) if i.startswith('COLOR_')]

# cv.imshow('Image', img)
# cv.waitKey(5000)

# print(flags)
