import logging

from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmdet.apis import init_detector, inference_detector
import numpy as np

log_level = logging.INFO
logger = logging.getLogger('mmdet_inference')
logger.setLevel(log_level)
handler = logging.StreamHandler()
handler.setLevel(log_level)
formatter = logging.Formatter('[%(levelname)s] [%(name)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def batch(iterable, bs=1):
    l = len(iterable)
    for ndx in range(0, l, bs):
        yield iterable[ndx:min(ndx + bs, l)]

class MMDetInference:
    _defaults = {
        "config_file": "configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py",
        "checkpoint_file": "checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
        "device": "cuda:0",
        "max_batch_size": 8,
        "bgr": True,
        "thresh": None
    }

    def __init__(self, **kwargs):
        '''
        kwargs
        ---------
        - config_file: str, path to config file
        - checkpoint_file: str, path to checkpoint file
        - device: str, "cpu" or "cuda:0", "cuda:1", etc
        - max_batch_size: int
        - bgr: boolean
        - thresh: float or None
        '''

        self.cfg = self._defaults
        self.cfg.update(kwargs)
        self.cfg.update({"flip_channels": not self.cfg["bgr"]})
        logger.info(f'mmdet_inference config file: {self.cfg["config_file"]}')
        logger.info(f'mmdet_inference checkpoint file: {self.cfg["checkpoint_file"]}')
        logger.info(f'mmdet_inference device: {self.cfg["device"]}')
        try:
            self.model = init_detector(self.cfg["config_file"], self.cfg["checkpoint_file"], device=self.cfg["device"])
        except FileNotFoundError:
            raise FileNotFoundError("Config and/or Checkpoint file not found")
        
        self.model = fuse_conv_bn(self.model)

        self._detect([np.zeros((10,10,3), dtype=np.uint8)])
        logger.info('Model warmed up!')
        

    # TODO: make this detect by batch
    def _detect(self, imgs):
        results = []
        inputs = []
        heights = []
        widths = []

        for img in imgs:
            if self.cfg["flip_channels"]:
                img = img[:,:,::-1]
            height, width = img.shape[:2]
            inputs.append(img)
            heights.append(height)
            widths.append(width)
        
        predictions = inference_detector(self.model, inputs)
        for p, w, h in zip(predictions, widths, heights):
            result = {
                "image_width": w,
                "image_height": h,
                "predictions": p
            }
            results.append(result)
        return results

    def _postprocess(self, results, box_format="ltrb", wanted_classes=None, buffer_ratio=0.0):
        all_dets = []
        for result in results:            
            bboxes = np.vstack(result["predictions"])
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(result["predictions"])
            ]
            labels = np.concatenate(labels)

            im_height = result["image_height"]
            im_width = result["image_width"]

            dets = []
            for bbox, label in zip(bboxes, labels):
                l, t, r, b, score = bbox
                if self.cfg["thresh"] is not None and score < self.cfg['thresh']:
                    continue
                pred_class_name = self.model.CLASSES[label]
                if wanted_classes is not None and pred_class_name not in wanted_classes:
                    continue
                    
                w = r - l + 1
                h = b - t + 1
                width_buffer = w * buffer_ratio
                height_buffer = h * buffer_ratio
                
                l = max( 0.0, l-0.5*width_buffer )
                t = max( 0.0, t-0.5*height_buffer )
                r = min( im_width - 1.0, r + 0.5*width_buffer )
                b = min( im_height - 1.0, b + 0.5*height_buffer )

                box_infos = []
                for c in box_format:
                    if c == 't':
                        box_infos.append( int(round(t)) ) 
                    elif c == 'l':
                        box_infos.append( int(round(l)) )
                    elif c == 'b':
                        box_infos.append( int(round(b)) )
                    elif c == 'r':
                        box_infos.append( int(round(r)) )
                    elif c == 'w':
                        box_infos.append( int(round(w+width_buffer)) )
                    elif c == 'h':
                        box_infos.append( int(round(h+height_buffer)) )
                    else:
                        assert False,'box_format given in detect unrecognised!'
                assert len(box_infos) > 0 ,'box infos is blank'

                dets.append( (box_infos, score, pred_class_name) )
            all_dets.append(dets)
        return all_dets

    def detect_get_box_in(self, images, box_format='ltrb', classes=None, buffer_ratio=0.):
        '''
        Params
        ------
        - images : ndarray-like or list of ndarray-like
        - box_format : string of characters representing format order, where l = left, t = top, r = right, b = bottom, w = width and h = height
        - classes : list of string, classes to focus on
        - buffer : float, proportion of buffer around the width and height of the bounding box
        Returns
        -------
        if one ndarray given, this returns a list (boxes in one image) of tuple (box_infos, score, predicted_class),
        
        else if a list of ndarray given, this return a list (batch) containing the former as the elements,
        where,
            - box_infos : list of floats in the given box format
            - score : float, confidence level of prediction
            - predicted_class : string
        '''
        single = False
        if isinstance(images, list):
            if len(images) <= 0:
                return None
            else:
                assert all(isinstance(im, np.ndarray) for im in images)
        elif isinstance(images, np.ndarray):
            images = [images]
            single = True
        
        all_dets = []
        for this_batch in batch(images, bs=self.cfg["max_batch_size"]):
            result = self._detect(this_batch)
            dets = self._postprocess(result, box_format=box_format, wanted_classes=classes)

            if len(all_dets) > 0:
                all_dets.extend(dets)
            else:
                all_dets = dets
        
        if single:
            return all_dets[0]
        else:
            return all_dets
