# _*_ coding:utf-8 _*_
"""
@Time     : 2022/12/5 14:42
@Author   : Wangxuanye
@File     : SURFMatcher.py
@Project  : Image-Stitching-OpenCV
@Software : PyCharm
@License  : (C)Copyright 2018-2028, Taogroup-NLPR-CASIA
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2022/12/5 14:42        1.0             None
"""

import timeit

import cv2
import numpy as np

class SURFMatcher:
    def __init__(self):
        self.surf = cv2.xfeatures2d.SURF_create()
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=0, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def match(self, i1, i2, direction):

        print("Direction : ", direction)

        image_set_1 = self.get_SURF_features(i1)
        image_set_2 = self.get_SURF_features(i2)
        matches = self.flann.knnMatch(image_set_2["des"], image_set_1["des"], k=2)
        good = []
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                good.append((m.trainIdx, m.queryIdx))

        if len(good) > 4:
            points_current = image_set_2["kp"]
            points_previous = image_set_1["kp"]

            matched_points_current = np.float32(
                [points_current[i].pt for (__, i) in good]
            )
            matched_points_prev = np.float32(
                [points_previous[i].pt for (i, __) in good]
            )

            H, _ = cv2.findHomography(
                matched_points_current, matched_points_prev, cv2.RANSAC, 4
            )
            return H
        return None

    def get_SURF_features(self, im):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        kp, des = self.surf.detectAndCompute(gray, None)
        return {"kp": kp, "des": des}
