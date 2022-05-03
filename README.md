# SSL-FL

## Set up environment
- ```conda env create -f environment_beit.yml```
- then ```pip install torch===1.7.1+cu110 torchvision===0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html```

## Datasets and simulated data splits
### Download data from google drive
```pip install gdown```

```gdown https://drive.google.com/uc?id=```

- Retina: 1BsOWjvBXktsHnKjNRol-PGF6HrYDyyrL 
- Derm / ISIC: 1EsnjGQI0exLgmPBvBQmqHFQsXaKKioRE
- COVID-FL: TO ADD
- Skin-FL: TO ADD

If you want to test on new datasets, please modify unilm/datasets.py and FedAvg_utils/data_utils.py

### Download gdrive for file uploading (optional)
Step1: ```wget https://github.com/prasmussen/gdrive/releases/download/2.1.1/gdrive_2.1.1_linux_386.tar.gz```

Step2: ```tar -xvf gdrive_2.1.1_linux_386.tar.gz```

Step3: ```./gdrive about```

Step4: ```./gdrive upload /home/documents/file_name.zip```

Step5: ```./gdrive list```

## Self-supervised Federated Pre-training
- Fed-BEiT: ```unilm/beit/run_beit_pretraining_FedAvg_distributed.py```
- Fed-MAE: ```unilm/mae/run_mae_pretraining_FedAvg.py```
### Fine-tuning with pre-trained checkpoints
The following table provides the pre-trained checkpoints used in the paper:
### BEiT Retina
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">Central</th>
<th valign="bottom">Split-1</th>
<th valign="bottom">Split-2</th>
<th valign="bottom">Split-3</th>
<!-- TABLE BODY -->
<tr><td align="left">pre-trained checkpoint</td>
<td align="center"><a href="">download</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1cMRtumZUm9Ftt8AssuKSUoxACkCEmaAg/view?usp=sharing">download</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1x_xdQDHFjEpwCq4AyMflHW8QITP3tvN5/view?usp=sharing">download</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1TPgoyqYK2ZBn4GmOdXX5AlDe8CrgWpx-/view?usp=sharing">download</a></td>
</tr>
</tbody></table>

### MAE Retina
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">Central</th>
<th valign="bottom">Split-1</th>
<th valign="bottom">Split-2</th>
<th valign="bottom">Split-3</th>
<!-- TABLE BODY -->
<tr><td align="left">pre-trained checkpoint</td>
<td align="center"><a href="https://drive.google.com/file/d/1Sih-9HPISfaR48DplmbvYmtv_xh2V8RJ/view?usp=sharing">download</a></td>
<td align="center"><a href="https://drive.google.com/file/d/18cG2rrweNKc8LS5LBTcUAt9A4om3YWGz/view?usp=sharing">download</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1Rdfm_o5CFWucLKckiOYbBr9UfEfcPaOu/view?usp=sharing">download</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1StZmgbxP0VWNane3K0R8jb8sVm2Xm3H4/view?usp=sharing">download</a></td>
</tr>
</tbody></table>

### BEiT COVID-FL
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">Central</th>
<th valign="bottom">Real-world Split/th>
<!-- TABLE BODY -->
<tr><td align="left">pre-trained checkpoint</td>
<td align="center"><a href="https://drive.google.com/file/d/1WI9TnIudIUmIfC6t3OyjPSR0T0LVlg7G/view?usp=sharing">download</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1B7fcORHeUB2rKTUu0vlTXqrcOc-XVub-/view?usp=sharing">download</a></td>
</tr>
</tbody></table>

### MAE COVID-FL
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">Central</th>
<th valign="bottom">Real-world Split/th>
<!-- TABLE BODY -->
<tr><td align="left">pre-trained checkpoint</td>
<td align="center"><a href="https://drive.google.com/file/d/1Ma55OepDzjcGHRYHVg4GahCxH9OY16gm/view?usp=sharing">download</a></td>
<td align="center"><a href="https://drive.google.com/file/d/16FIte4hkp5I9MUztEcgmAA2F02_2Zr1S/view?usp=sharing">download</a></td>
</tr>
</tbody></table>

## IMNET Pretrained models
### BEiT ImageNet-22k
BEiT weights pretrained on ImageNet-22k. The models were pretrained with 224x224 resolution.
- `BEiT-base`: #layer=12; hidden=768; FFN factor=4x; #head=12; patch=16x16 (#parameters: 86M)

Download checkpoints that are **self-supervised pretrained** on ImageNet-22k:
- BEiT-base: [beit_base_patch16_224_pt22k.pth](https://unilm.blob.core.windows.net/beit/beit_base_patch16_224_pt22k.pth)
- Dall-e Tokenizers: [download link](https://drive.google.com/file/d/1DkXJTQC7ELCoBUwq8j4XNoxe7dkPUEdr/view?usp=sharing)

### MAE ImageNet-22k
MAE weights pretrained on ImageNet-22k. The models were pretrained with 224x224 resolution.
Download checkpoints that are **self-supervised pretrained** on ImageNet-22k:
- MAE-base: [mae_pretrain_vit_base.pth](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth)


## Self-supervised Federated Fine-tuning
- Fed-BEiT: ```unilm/beit/run_class_finetuning_FedAvg_distributed.py```
- Fed-MAE: ```unilm/mae/run_class_finetuning_FedAvg.py```

Scripts for BEiT and MAE and in unilm/beit/script and unilm/mae/script.
