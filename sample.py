import cv2
import time
from mmdet_inference.mmdet_inference import MMDetInference

mmdet_inference = MMDetInference()
imgs = [
    cv2.imread("test.jpg"),
    cv2.imread("test.jpg"),
    cv2.imread("test.jpg"),
    cv2.imread("test.jpg"),
    cv2.imread("test.jpg"),
    cv2.imread("test.jpg"),
    cv2.imread("test.jpg"),
    cv2.imread("test.jpg"),
    cv2.imread("test.jpg"),
    cv2.imread("test.jpg")
]
start = time.time()
detection = mmdet_inference.detect_get_box_in(imgs, classes=['dog'])
print(detection)
print(time.time() - start)
