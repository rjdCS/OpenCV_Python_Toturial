# coding=utf-8
"""
查看所有被支持的鼠标事件
"""

import cv2
events=[i for i in dir(cv2) if 'EVENT'in i]
print(events)