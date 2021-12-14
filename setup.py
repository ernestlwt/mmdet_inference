from setuptools import setup, find_packages

setup(
        name='mmdet_inference',
        version='1.0',
        packages=find_packages(exclude=("test",)),
        install_requires=[
            'opencv-python',
            'torch',
            'torchvision',
            'mmcv-full',
            'mmdet'
            ]
    )