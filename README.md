# Fundus-Lesions-Instance-Segmentation

### Installation
```
conda create --name detectron2
conda activate detectron2

conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

git clone https://github.com/marcelosdias/Fundus-Lesions-Instance-Segmentation.git

cd Fundus-Lesions-Instance-Segmentation/
```
Anaconda installation:
https://www.anaconda.com/products/distribution/start-coding-immediately

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
### Predicting
```
cd model/
python predict.py --dataset ddr/idrid --type valid/test --iou 0.25 --file_name 007-2846-100.jpg
cd ..
```

### ✒️ Authors
* **Alejandro Pereira** - Computer Science, Federal University of Pelotas - UFPel, Pelotas, Brazil.
* **Carlos Santos** - Federal Institute of Education, Science and Technology Farroupilha - IFFar, Alegrete, Brazil.
* **Daniel Welfer** - Departament of Applied Computing, Federal University of Santa Maria - UFSM, Santa Maria, Brazil.
* **Marcelo Ribeiro** - Computer Science, Federal University of Pelotas - UFPel, Pelotas, Brazil.
* **Marcelo Dias** - Computer Science, Federal University of Pelotas - UFPel, Pelotas, Brazil.
* **Marilton Aguiar** - Postgraduate Program in Computing, Federal University of Pelotas - UFPel, Pelotas, Brazil.

