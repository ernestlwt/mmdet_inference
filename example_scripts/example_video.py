import cv2
import time
import argparse
from pathlib import Path

from mmdet_inference.mmdet_inference import MMDetInference

parser = argparse.ArgumentParser()
parser.add_argument('video_path', help='path to video')
parser.add_argument('--thresh', help='OD confidence threshold', default=0.4, type=float)
parser.add_argument('--out', help='flag to output video', action='store_true')
parser.add_argument('--outpath', help='path to output video', default='out.mp4', type=str)

parser.add_argument('--nodisplay', help='flag to not display', action='store_true')

args = parser.parse_args()

assert args.thresh > 0.0

od = MMDetInference(
    config_file="../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py",
    checkpoint_file="../checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
    thresh=args.thresh
)

if args.video_path.isdigit():
    vp = int(args.video_path)
else:
    vp = Path(args.video_path)
    assert vp.is_file(),'{} not a file'.format(vp)
    vp = str(vp)
cap = cv2.VideoCapture(vp)
assert cap.isOpened(),'Cannot open video file {}'.format(vp)

if args.out:
    outpath = args.outpath
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    frameSize = int(cap.get(3)), int(cap.get(4))
    fps = int(cap.get(5))
    out_vid = cv2.VideoWriter(outpath, fourcc, int(fps), frameSize)
else:
    out_vid = None

if not args.nodisplay:
    cv2.namedWindow('Faster-RCNN FPN', cv2.WINDOW_NORMAL)

while True:
    # Decode
    ret, frame = cap.read()
    if not ret:
        break
    # Inference
    tic = time.perf_counter()
    dets = od.detect_get_box_in([frame], box_format='ltrb', classes=None)
    toc = time.perf_counter()
    print('infer duration: {:0.3f}s'.format(toc-tic))
    dets = dets[0]

    # Drawing
    show_frame = frame.copy()
    for det in dets:
        ltrb, conf, clsname = det
        l,t,r,b = ltrb
        cv2.rectangle(show_frame, (int(l),int(t)),(int(r),int(b)), (255,255,0))
        cv2.putText(show_frame, '{}:{:0.2f}'.format(clsname, conf), (l,b), cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(255,255,0), lineType=2)

    if not args.nodisplay:
        cv2.imshow('Faster-RCNN FPN', show_frame)
        if cv2.waitKey(1) == ord('q'):
            break
    if out_vid:
        out_vid.write(show_frame)

cv2.destroyAllWindows()
if out_vid:
    out_vid.release()