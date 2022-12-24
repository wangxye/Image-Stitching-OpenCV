# _*_ coding:utf-8 _*_
"""
@Time     : 2022/12/4 16:09
@Author   : Wangxuanye
@File     : Multi_Image_Stitching.py
@Project  : Image-Stitching-OpenCV
@Software : PyCharm
@License  : (C)Copyright 2018-2028, Taogroup-NLPR-CASIA
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2022/12/4 16:09        1.0             None
"""
import timeit

import cv2
import numpy as np

from SIFIMatcher import SIFTMatcher
from SURFMatcher import SURFMatcher
from ORBMatcher import ORBMatcher

# FEATURE_TYPE = "SURF"
# FEATURE_TYPE = "ORB"
FEATURE_TYPE = "SIFI"


class Stitch:
    def __init__(self, filenames):
        # self.path = args
        # fp = open(self.path, 'r')
        # filenames = [each.rstrip('\r\n') for each in fp.readlines()]
        # filenames = args

        self.filenames = filenames

        print(filenames)
        self.images = [cv2.resize(cv2.imread(each), (480 * 3, 360 * 3)) for each in filenames]
        # self.images = [cv2.imread(each) for each in filenames]
        self.count = len(self.images)
        self.left_list, self.right_list, self.center_im = [], [], None

        if FEATURE_TYPE == "SURF":
            print(FEATURE_TYPE)
            self.matcher_obj = SURFMatcher()
        elif FEATURE_TYPE == "SIFI":
            print(FEATURE_TYPE)
            self.matcher_obj = SIFTMatcher()
        elif FEATURE_TYPE == "ORB":
            print(FEATURE_TYPE)
            self.matcher_obj = ORBMatcher()

        self.leftImage = None
        self.rightImage = None
        # self.prepare_lists()

    def prepare_lists(self):
        print("Number of images : %d" % self.count)
        self.centerIdx = self.count / 2
        print("Center index image : %d" % self.centerIdx)
        self.center_im = self.images[int(self.centerIdx)]
        for i in range(self.count):
            if (i <= self.centerIdx):
                self.left_list.append(self.images[i])
            else:
                self.right_list.append(self.images[i])
        print("Image lists prepared")

    def leftshift(self):
        # self.left_list = reversed(self.left_list)
        a = self.left_list[0]
        tmp = None
        for b in self.left_list[1:]:
            H = self.matcher_obj.match(a, b, 'left')
            # print("Homography is : ", H)
            xh = np.linalg.inv(H)
            # print("Inverse Homography :", xh)
            br = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]))
            br = br / br[-1]
            tl = np.dot(xh, np.array([0, 0, 1]))
            tl = tl / tl[-1]
            bl = np.dot(xh, np.array([0, a.shape[0], 1]))
            bl = bl / bl[-1]
            tr = np.dot(xh, np.array([a.shape[1], 0, 1]))
            tr = tr / tr[-1]
            cx = int(max([0, a.shape[1], tl[0], bl[0], tr[0], br[0]]))
            cy = int(max([0, a.shape[0], tl[1], bl[1], tr[1], br[1]]))
            offset = [abs(int(min([0, a.shape[1], tl[0], bl[0], tr[0], br[0]]))),
                      abs(int(min([0, a.shape[0], tl[1], bl[1], tr[1], br[1]])))]
            dsize = (cx + offset[0], cy + offset[1])
            print("image dsize =>", dsize, "offset", offset)

            tl[0:2] += offset
            bl[0:2] += offset
            tr[0:2] += offset
            br[0:2] += offset
            dstpoints = np.array([tl, bl, tr, br])
            srcpoints = np.array([[0, 0], [0, a.shape[0]], [a.shape[1], 0], [a.shape[1], a.shape[0]]])
            # print('sp',sp,'dp',dp)
            M_off = cv2.findHomography(srcpoints, dstpoints)[0]
            # print('M_off', M_off)
            warped_img2 = cv2.warpPerspective(a, M_off, dsize)
            # cv2.imshow("warped", warped_img2)
            # cv2.waitKey()
            warped_img1 = np.zeros([dsize[1], dsize[0], 3], np.uint8)
            warped_img1[offset[1]:b.shape[0] + offset[1], offset[0]:b.shape[1] + offset[0]] = b
            tmp = blend_linear(warped_img1, warped_img2)
            a = tmp

        self.leftImage = tmp

    def rightshift(self):
        tmp = None
        for each in self.right_list:
            H = self.matcher_obj.match(self.leftImage, each, 'right')
            # print("Homography :", H)
            if H is None:
                continue
            br = np.dot(H, np.array([each.shape[1], each.shape[0], 1]))
            br = br / br[-1]
            tl = np.dot(H, np.array([0, 0, 1]))
            tl = tl / tl[-1]
            bl = np.dot(H, np.array([0, each.shape[0], 1]))
            bl = bl / bl[-1]
            tr = np.dot(H, np.array([each.shape[1], 0, 1]))
            tr = tr / tr[-1]
            cx = int(max([0, self.leftImage.shape[1], tl[0], bl[0], tr[0], br[0]]))
            cy = int(max([0, self.leftImage.shape[0], tl[1], bl[1], tr[1], br[1]]))
            offset = [abs(int(min([0, self.leftImage.shape[1], tl[0], bl[0], tr[0], br[0]]))),
                      abs(int(min([0, self.leftImage.shape[0], tl[1], bl[1], tr[1], br[1]])))]
            dsize = (cx + offset[0], cy + offset[1])
            print("image dsize =>", dsize, "offset", offset)

            tl[0:2] += offset
            bl[0:2] += offset
            tr[0:2] += offset
            br[0:2] += offset
            dstpoints = np.array([tl, bl, tr, br])
            srcpoints = np.array([[0, 0], [0, each.shape[0]], [each.shape[1], 0], [each.shape[1], each.shape[0]]])
            M_off = cv2.findHomography(dstpoints, srcpoints)[0]
            warped_img2 = cv2.warpPerspective(each, M_off, dsize, flags=cv2.WARP_INVERSE_MAP)
            # cv2.imshow("warped", warped_img2)
            # cv2.waitKey()
            warped_img1 = np.zeros([dsize[1], dsize[0], 3], np.uint8)
            warped_img1[offset[1]:self.leftImage.shape[0] + offset[1],
            offset[0]:self.leftImage.shape[1] + offset[0]] = self.leftImage
            tmp = blend_linear(warped_img1, warped_img2)
            self.leftImage = tmp

        self.rightImage = tmp

    def showImage(self, string=None):
        if string == 'left':
            cv2.imshow("left image", self.leftImage)
        elif string == "right":
            cv2.imshow("right Image", self.rightImage)
        cv2.waitKey()

    def stitch(self):
        """
        stitches the images into a panorama
        """

        self.prepare_lists()

        # left stitching
        start = timeit.default_timer()
        self.leftshift()

        self.rightshift()

        stop = timeit.default_timer()
        duration = stop - start
        print("stitching took %.2f seconds." % duration)

        return self.leftImage


def blend_linear(warp_img1, warp_img2):
    img1 = warp_img1
    img2 = warp_img2

    img1mask = ((img1[:, :, 0] | img1[:, :, 1] | img1[:, :, 2]) > 0)
    img2mask = ((img2[:, :, 0] | img2[:, :, 1] | img2[:, :, 2]) > 0)

    r, c = np.nonzero(img1mask)
    out_1_center = [np.mean(r), np.mean(c)]

    r, c = np.nonzero(img2mask)
    out_2_center = [np.mean(r), np.mean(c)]

    vec = np.array(out_2_center) - np.array(out_1_center)
    intsct_mask = img1mask & img2mask

    r, c = np.nonzero(intsct_mask)

    out_wmask = np.zeros(img2mask.shape[:2])
    proj_val = (r - out_1_center[0]) * vec[0] + (c - out_1_center[1]) * vec[1]
    out_wmask[r, c] = (proj_val - (min(proj_val) + (1e-3))) / \
                      ((max(proj_val) - (1e-3)) - (min(proj_val) + (1e-3)))

    # blending
    mask1 = img1mask & (out_wmask == 0)
    mask2 = out_wmask
    mask3 = img2mask & (out_wmask == 0)

    out = np.zeros(img1.shape).astype(np.float32)
    for c in range(3):
        out[:, :, c] = img1[:, :, c] * (mask1 + (1 - mask2) * (mask2 != 0)) + \
                       img2[:, :, c] * (mask2 + mask3)
    return np.uint8(out)


if __name__ == "__main__":
    files = [

        # "images/2-2.jpg",
        # "images/2-3.jpg",

        # "images/3-2.jpg",
        # "images/3-3.jpg",

        # "images/5-1.jpg",
        # "images/5-2.jpg",
        # "images/5-3.jpg",
        # "images/5-4.jpg",

        # "images/6-1.jpg",
        # "images/6-2.jpg",
        # "images/6-3.jpg",

        # "images/7-1.jpg",
        # "images/7-2.jpg",

        "images/8-1.jpg",
        "images/8-2.jpg",

        # "images/9-1.jpg",
        # "images/9-2.jpg",
        # "images/9-3.jpg",
        # "images/9-4.jpg",
        # "images/9-5.jpg",
    ]
    s = Stitch(files)

    panorama = s.stitch()
    print("saving...")

    save_path = "./result/{0}/{0}_".format(FEATURE_TYPE)
    # save_path = "./result/SURF_"
    # save_path = "./result/ORB_"
    for each in files:
        save_path += each[each.rfind('/') + 1:-4] + '_'

    save_path += "panorama.jpg"

    print(save_path)
    cv2.imwrite(save_path, panorama)

