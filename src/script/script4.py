# Mỗi person 2 ảnh 1 left, 1 right
# Bộ ảnh còn lại làm nhiễu cho unsupervised


import sys
sys.path.append("../")

import os
import cv2
import numpy as np

from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm 
from processing.similar_image import k_medoids_pam_robust, compare_image