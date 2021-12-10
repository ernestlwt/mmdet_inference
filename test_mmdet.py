from mmdet.apis import init_detector, inference_detector
import numpy as np


config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
device = 'cuda:0'

img = 'test.jpg'

model = init_detector(config_file, checkpoint_file, device=device)
result = inference_detector(model, img)

bboxes = np.vstack(result)
labels = [
    np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(result)
]
labels = np.concatenate(labels)

print(bboxes)
bbox_int = bboxes.astype(np.int32)
print(bbox_int)
print()
print(labels)
print(model.CLASSES)


model.show_result(img, result)
model.show_result(img, result, out_file='result.jpg')