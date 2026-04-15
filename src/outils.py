# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 01:06:21 2026

@author: AWOUNANG
"""
import numpy as np
import cv2


def decode_image(file_bytes: bytes):
    np_arr = np.frombuffer(file_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)