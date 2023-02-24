# A New Approach for Fundus Lesions Instance Segmentation

![](https://github.com/marcelosdias/Fundus-Lesions-Instance-Segmentation/blob/main/images/exmple-image.gif)


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

### Instalation Error

If you are using Windows and get this: ERROR: Invalid requirement: "'git+https://github.com/facebookresearch/detectron2.git'"

You can try use this command: python -m pip install git+https://github.com/facebookresearch/detectron2.git

### Download proposed work

- [Download the proposed work trained with ddr](https://github.com/marcelosdias/Fundus-Lesions-Instance-Segmentation/releases/download/proposed-work-ddr/ddr.zip)

- [Download the proposed work trained with idrid](https://github.com/marcelosdias/Fundus-Lesions-Instance-Segmentation/releases/download/proposed-work-idrid/idrid.zip)

You need to paste and extract .zip into the model folder

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
Modes:
* es (Early Stopping): By default stop after 100 epochs
* vl (Validation loss): Generate validation loss during the training
```
cd model/
python train.py --mode es/vl --dataset ddr/idrid --epochs xx
cd ..
```
### Testing
```
cd model/
python test.py --dataset ddr/idrid --type valid --iou 0.25
cd ..
```
### Predicting
```
cd model/
python predict.py --dataset ddr/idrid --type valid/test --iou 0.25 --file_name 007-2846-100.jpg
cd ..
```

### Authors
* **Marcelo Dias** - Computer Science, Federal University of Pelotas - UFPel, Pelotas, Brazil.
* **Carlos Santos** - Federal Institute of Education, Science and Technology Farroupilha - IFFar, Alegrete, Brazil.
* **Marilton Aguiar** - Postgraduate Program in Computing, Federal University of Pelotas - UFPel, Pelotas, Brazil.
* **Daniel Welfer** - Departament of Applied Computing, Federal University of Santa Maria - UFSM, Santa Maria, Brazil.
* **Alejandro Pereira** - Computer Science, Federal University of Pelotas - UFPel, Pelotas, Brazil.
* **Marcelo Ribeiro** - Computer Science, Federal University of Pelotas - UFPel, Pelotas, Brazil.

### Acknowledgements
* https://github.com/facebookresearch/detectron2
* https://github.com/brunobelloni/binary-to-coco-json-converter
* Coordenação de Aperfeiçoamento de Pessoal de Nível Superior - Brasil (CAPES)
* Programa Institucional de Bolsas de Iniciação Científica - PROBIC/FAPERGS
