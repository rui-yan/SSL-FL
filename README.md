# SSL-FL

## Set up environment
- ```conda env create -f environment_beit/yml```
- then ```pip install torch===1.7.1+cu110 torchvision===0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html```

## Datasets and simulated data splits
- CIFAR10: [download link](https://drive.google.com/drive/folders/1ZErR7RMSVImkzYzz0hLl25f9agJwp0Zx)
- Retina: [download link](https://drive.google.com/file/d/1eVcT_IRF8n3sLNyZS-JT4iU-1C19QIOh/view?usp=sharing)
- COVIDx: [download link](https://drive.google.com/file/d/1mMJc4yXGKt6L3vkdcUFXZyUOvmFvG2Px/view?usp=sharing)

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
