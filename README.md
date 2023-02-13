# Fundus-Lesions-Instance-Segmentation

### Installation
```
conda create --name detectron2
conda activate detectron2

conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

git clone https://github.com/marcelosdias/Fundus-Lesions-Instance-Segmentation.git
```
For more information about detectron2 installation: 
https://detectron2.readthedocs.io/en/latest/tutorials/install.html

### Pre-processing steps
1. Generate Annotations
```
cd pre-processing/binary-to-coco-json-converter/
python main.py
cd ../..
```
2. Cropping and Tilling
```
cd pre-processing/cropping-tilling/
python main_coco.py
cd ../..
```
### Training
```
cd model/
python train.py --dataset ddr/idrid --epochs xx
cd ..
```
### Testing
```
cd model/
python test.py --dataset ddr --type valid --iou 0.25
cd ..
```

