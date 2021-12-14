### MMDET inference wrapper

##### TODO
- test with different configs
- weights download
- packagify

Tested with:
```
python 3.8
cudatoolkit 10.2
torch 1.10
torchvision 0.11.1
```

Install mmdet with:
```
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10.0/index.html
pip install mmdet

# additional libraries for mmdet
# for instaboost
pip install instaboostfast
# for panoptic segmentation
pip install git+https://github.com/cocodataset/panopticapi.git
# for LVIS dataset
pip install git+https://github.com/lvis-dataset/lvis-api.git
# for albumentations
pip install albumentations>=0.3.2 --no-binary imgaug,albumentations

```