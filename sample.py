import cv2
from mmdet_inference.mmdet_inference import MMDetInference

mmdet_inference = MMDetInference()
imgs = [
    cv2.imread("test.jpg")
]
detection = mmdet_inference.detect_get_box_in(imgs)
print(detection)
