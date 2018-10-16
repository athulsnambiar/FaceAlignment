"""
Average Face.

This script takes as command line argument the folder containing the
annotations of helen dataset and then creates a file "avg_shape.npy"
containing the average face of the dataset
"""

import numpy as np
import pandas as pd
import sys
import os
from typing import List
import cv2


if len(sys.argv) != 3:
    print("Run as following....")
    print("python3 test_draw_face.py <image path> <numpy array path>")
    sys.exit(0)

if not os.path.exists(sys.argv[1]) or not os.path.isfile(sys.argv[1]):
    print("Image doesn't exist or Invalid")
    sys.exit(0)

if not os.path.exists(sys.argv[2]) or not os.path.isfile(sys.argv[2]):
    print("Annotation doesn't exist or Invalid")
    sys.exit(0)

img_path: str = sys.argv[1]
anno_path: str = sys.argv[2]

image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
#image = np.ones((2000,2000))
shape = np.load(anno_path)
for i in shape:
    cv2.circle(image, (int(i[0]), int(i[1])), 10, (0, 0, 255), 2)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow("image", 500,500)
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
