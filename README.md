# Image-Stitching-OpenCV
Simple image stitching algorithm using SIFT/SURF/ORB, homography, KNN and Ransac in Python.
For full details and explanations, you're welcome to read `image_stitching.py` or `Multi_Image_Stitching.py`. 	

The project is to implement a featured based automatic image stitching algorithm. When we input two images with overlapped fields, we expect to obtain a wide seamless panorama.

We use scale invariant features transform(SIFT/SURF/ORB) to extract local features of the input images, K nearest neighbors algorithms to match these features and Random sample consensus(Ransac) to calculate the homograph matrix, which will be used for image warping. Finally we apply a weighted matrix as a mask for image blending.

In addition, we support the stitching of two or more images.

## Dependency
- Python 2 or 3 
- OpenCV 3

## Usage
`python Image_Stitching [/PATH/img1] [/PATH/img2]`

`python Multi_Image_Stitching  [/PATH/img1] [/PATH/img2] ...`

