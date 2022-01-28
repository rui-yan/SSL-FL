# SSL-FL

## Set up environment
- ```conda env create -f environment_beit.yml```
- then ```pip install torch===1.7.1+cu110 torchvision===0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html```

## Datasets and simulated data splits
### Download gdrive for file uploading
Step1: ```wget https://github.com/prasmussen/gdrive/releases/download/2.1.1/gdrive_2.1.1_linux_386.tar.gz```
Step2: ```tar -xvf gdrive_2.1.1_linux_386.tar.gz```
Step3: ```./gdrive about```
Step4: ```./gdrive upload /home/documents/file_name.zip```
Step5: ```./gdrive list```
### Download data from google drive
```pip install gdown```
```gdown https://drive.google.com/uc?id=```
- CIFAR10: [download link](https://drive.google.com/file/d/1elgSFsfDI3Tfdf3BZ44lxBs2BaJ4lf5_/view?usp=sharing)
- Retina: [download link](https://drive.google.com/file/d/10l2A5dW5pdU6dXAjmP-o_dVgEduhsoT2/view?usp=sharing)
- COVIDx: [download link](https://drive.google.com/file/d/1zOHaHVqgTMk_25fciTojq7ZIurpWIFSb/view?usp=sharing)

## BEiT
### IMNET Pretrained models
We provide four BEiT weights pretrained on ImageNet-22k. The models were pretrained with 224x224 resolution.

- `BEiT-base`: #layer=12; hidden=768; FFN factor=4x; #head=12; patch=16x16 (#parameters: 86M)
- `BEiT-large`: #layer=24; hidden=1024; FFN factor=4x; #head=16; patch=16x16 (#parameters: 304M)
- 
Download checkpoints that are **self-supervised pretrained** on ImageNet-22k:
- BEiT-base: [beit_base_patch16_224_pt22k](https://unilm.blob.core.windows.net/beit/beit_base_patch16_224_pt22k.pth)
- BEiT-large: [beit_large_patch16_224_pt22k](https://unilm.blob.core.windows.net/beit/beit_large_patch16_224_pt22k.pth)

Dall-e Tokenizers: [download link](https://drive.google.com/file/d/1DkXJTQC7ELCoBUwq8j4XNoxe7dkPUEdr/view?usp=sharing)
