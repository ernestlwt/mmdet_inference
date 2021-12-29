### MMDET inference wrapper

##### Configuration Files
https://github.com/open-mmlab/mmdetection/tree/master/configs


##### Checkpoint Files
At https://github.com/open-mmlab/mmdetection, look for ./configs/**\<model u are interested in>**/readme.md, there will be a link for you to download

### Notes
```
threshold should be set in config file. but can be added while initializing mmdetinference object too as per levan/det2. might move this to the inference function instead of during model object initialization for greater flexibility
```

Tested with:
```
python 3.8
cudatoolkit 10.2
torch 1.10
torchvision 0.11.1
```

Install dependencies with:
```
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10.0/index.html
pip install mmdet
```

Install this repo with:
```
pip install .

```