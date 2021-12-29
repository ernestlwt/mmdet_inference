import cv2
import time
import argparse
from pathlib import Path

from mmdet_inference.mmdet_inference import MMDetInference

parser = argparse.ArgumentParser()
parser.add_argument('video_path', help='path to video')
parser.add_argument('--thresh', help='OD confidence threshold', default=0.4, type=float)
args = parser.parse_args()

assert args.thresh > 0.0

od1 = MMDetInference(
    config_file="../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py",
    checkpoint_file="../checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
    thresh=args.thresh,
    device='cuda:0'
)

od2 = MMDetInference(
    config_file="../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py",
    checkpoint_file="../checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
    thresh=args.thresh,
    device='cpu'
)

if args.video_path.isdigit():
    vp = int(args.video_path)
else:
    vp = Path(args.video_path)
    assert vp.is_file(),'{} not a file'.format(vp)
    vp = str(vp)
cap = cv2.VideoCapture(vp)
assert cap.isOpened(),'Cannot open video file {}'.format(vp)

cv2.namedWindow('Faster-RCNN FPN OD1', cv2.WINDOW_NORMAL)
cv2.namedWindow('Faster-RCNN FPN OD2', cv2.WINDOW_NORMAL)

while True:
    # Decode
    ret, frame = cap.read()
    if not ret:
        break
    # Inference
    tic = time.perf_counter()
    dets1 = od1.detect_get_box_in([frame], box_format='ltrb')
    toc = time.perf_counter()
    print('OD1 infer duration: {:0.3f}s'.format(toc-tic))
    dets1 = dets1[0]

    tic = time.perf_counter()
    dets2 = od2.detect_get_box_in([frame], box_format='ltrb')
    toc = time.perf_counter()
    print('OD2 infer duration: {:0.3f}s'.format(toc-tic))
    dets2 = dets2[0]

    # Drawing
    show_frame1 = frame.copy()
    for det in dets1:
        ltrb, conf, clsname = det
        l,t,r,b = ltrb
        cv2.rectangle(show_frame1, (int(l),int(t)),(int(r),int(b)), (255,255,0))
        cv2.putText(show_frame1, '{}:{:0.2f}'.format(clsname, conf), (l,b), cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(255,255,0), lineType=2)

    show_frame2 = frame.copy()
    for det in dets2:
        ltrb, conf, clsname = det
        l,t,r,b = ltrb
        cv2.rectangle(show_frame2, (int(l),int(t)),(int(r),int(b)), (255,255,0))
        cv2.putText(show_frame2, '{}:{:0.2f}'.format(clsname, conf), (l,b), cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(255,255,0), lineType=2)

    cv2.imshow('Faster-RCNN FPN OD1', show_frame1)
    cv2.imshow('Faster-RCNN FPN OD2', show_frame2)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()