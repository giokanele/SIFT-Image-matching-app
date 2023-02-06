#!/usr/bin/env python3

import cv2
import numpy as np

img = cv2.imread("/home/fizzer/SIFT_app/Screenshot_2023-02-02_12-00-50.png", cv2.IMREAD_GRAYSCALE)

cap = cv2.VideoCapture(0)

#features
sift = cv2.SIFT_create()
kp_image, desc_image = sift.detectAndCompute(img, None)
#img = cv2.drawKeypoints(img, kp_image, img)

#Feature matching
index_param = dict(algorithm=0, trees=5)
search_param = dict()
flann = cv2.FlannBasedMatcher(index_param, search_param) 


while True:
    _,  frame = cap.read()
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #train image
    

    kp_grayFrame, desc_grayframe = sift.detectAndCompute(grayFrame, None) 
    #grayFrame = cv2.drawKeypoints(grayFrame, kp_grayFrame, grayFrame)
    matches = flann.knnMatch(desc_image, desc_grayframe, k=2)

    good_points = []
    for m, n in matches:
        if m.distance < 0.55*n.distance:
            good_points.append(m)
    

    if len(good_points) > 10:
        query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
        train_pts = np.float32([kp_grayFrame[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)

        matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()

        h, w = img.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix)
        homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)

        cv2.imshow("Homography", homography)
    else:
        cv2.imshow("Homography", frame)

    # img3 = cv2.drawMatches(img, kp_image, grayFrame, kp_grayFrame, good_points, grayFrame) 

    # cv2.imshow("gray_frame", img3)
    # cv2.imshow("img", img)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()