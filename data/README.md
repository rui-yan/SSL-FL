### Data Preparation
In this paper, we conduct experiments on Retina, Derm and COVID-FL datasets. 

```
data
|-- Retina
    |-- central
    |-- 5_clients/
        |-- split_1/
            |-- client_1.csv
            |-- client_2.csv
            |-- client_3.csv
            |-- client_4.csv
            |-- client_5.csv
        |-- split_2
        |-- split_3
    |-- train
    |-- test
    |-- train.csv
    |-- test.csv
    |-- labels.csv
|-- COVID-FL
    |-- central
    |-- 12_clients
        |-- split_real
            |-- bimcv.csv  
            |-- cohen.csv  
            |-- eurorad.csv  
            |-- gz.csv  
            |-- ml-workgroup.csv  
            |-- ricord_c.csv  
            |-- rsna-0.csv  
            |-- rsna-1.csv  
            |-- rsna-2.csv  
            |-- rsna-3.csv  
            |-- rsna-4.csv  
            |-- sirm.csv
    |-- train
    |-- test
    |-- train.csv
    |-- test.csv
    |-- labels.csv
|-- tokenizer_weight
|-- ckpts
```

Each data folder contains 'n_clients' subfolder, which includes data split information in a .csv file. The .csv file contains the filenames of the images belonging to each client in the data split. 

Here, [data_split.py](https://github.com/rui-yan/SSL-FL/blob/main/data/data_split.py) is used to simulate the IID and non-IID data partitions, and you can visualize the generated data partitions in [view_data_split.ipynb](https://github.com/rui-yan/SSL-FL/blob/main/data/view_data_split.ipynb).

If you would like to train using your own custom datasets, please ensure that your data is organized according to the directory structure mentioned above. Additionally, you can modify the data augmentation strategies in [SSL-FL/code/util/datasets.py](https://github.com/rui-yan/SSL-FL/blob/main/code/util/datasets.py) and the data loader in [SSL-FL/code/util/data_utils.py](https://github.com/rui-yan/SSL-FL/blob/main/code/util/data_utils.py).

Below are the download links for the Retina, COVID-FL, and Derm datasets.
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">Retina</th>
<th valign="bottom">Derm</th>
<th valign="bottom">COVID-FL</th>
<!-- TABLE BODY -->
<tr><td align="left">Download Link</td>
<td align="center"><a href="https://drive.google.com/file/d/1bW--_qRZnWbkb0XXvGBCSferdqXZ6pe7/view?usp=share_link">link</a></td>
<td align="center"><a href="https://drive.google.com/file/d/10M-yNrsQ6OdbRrn_Dj72ebHs9LxQtsZp/view?usp=share_link">link</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1cuvoYvt-EVs5qtA5Xgos0yUJmfPhRbwg/view?usp=share_link">link</a></td>
</tr>
</tbody></table>

### Use gdown to download data to your path (optional)
Step1: ```pip install gdown```

Step2: ```gdown https://drive.google.com/uc?id=<the_file_id>``` where <the_file_id> can be obtained from the download links above.


### Fed-BEiT and Fed-MAE pre-trained model checkpoints on target datasets: 
To reproduce the results of Fed-BEiT and Fed-MAE as reported in the paper, you have two options.

First, you can perform pre-training using Fed-BEiT and Fed-MAE on the datasets, and then fine-tune the pre-trained models. Alternatively, you can fine-tune the pre-trained models with checkpoints provided below.

#### Federated pre-training with Retina
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Method</th>
<th valign="bottom">Pre-training Data</th>
<th valign="bottom">Central</th>
<th valign="bottom">Split-1</th>
<th valign="bottom">Split-2</th>
<th valign="bottom">Split-3</th>
<!-- TABLE BODY -->
<tr>
<td align="left">Fed-BEiT</td>
<td align="left">Retina</td>
<td align="center"><a href="https://drive.google.com/file/d/1wxmxgbAws9ahrh8BAv7XW5RUZEy9BMbo/view?usp=sharing">download</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1cMRtumZUm9Ftt8AssuKSUoxACkCEmaAg/view?usp=sharing">download</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1x_xdQDHFjEpwCq4AyMflHW8QITP3tvN5/view?usp=sharing">download</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1TPgoyqYK2ZBn4GmOdXX5AlDe8CrgWpx-/view?usp=sharing">download</a></td>
</tr>
<tr>
<td align="left">Fed-MAE</td>
<td align="left">Retina</td>
<td align="center"><a href="https://drive.google.com/file/d/1Sih-9HPISfaR48DplmbvYmtv_xh2V8RJ/view?usp=sharing">download</a></td>
<td align="center"><a href="https://drive.google.com/file/d/18cG2rrweNKc8LS5LBTcUAt9A4om3YWGz/view?usp=sharing">download</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1Rdfm_o5CFWucLKckiOYbBr9UfEfcPaOu/view?usp=sharing">download</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1StZmgbxP0VWNane3K0R8jb8sVm2Xm3H4/view?usp=sharing">download</a></td>
</tr>
</tbody></table>

#### Federated pre-training with COVID-FL
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Method</th>
<th valign="bottom">Pre-training Data</th>
<th valign="bottom">Central</th>
<th valign="bottom">Real-world Split</th>
<!-- TABLE BODY -->
<tr>
<td align="left">Fed-BEiT</td>
<td align="left">COVID-FL</td>
<td align="center"><a href="https://drive.google.com/file/d/1WI9TnIudIUmIfC6t3OyjPSR0T0LVlg7G/view?usp=sharing">download</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1B7fcORHeUB2rKTUu0vlTXqrcOc-XVub-/view?usp=sharing">download</a></td>
</tr>
<tr>
<td align="left">Fed-MAE</td>
<td align="left">COVID-FL</td>
<td align="center"><a href="https://drive.google.com/file/d/1Ma55OepDzjcGHRYHVg4GahCxH9OY16gm/view?usp=sharing">download</a></td>
<td align="center"><a href="https://drive.google.com/file/d/16FIte4hkp5I9MUztEcgmAA2F02_2Zr1S/view?usp=sharing">download</a></td>
</tr>
</tbody></table>

### pre-trained model checkpoints on ImageNet
To obtain the results with models pre-trained on ImageNet, you can download the model checkpoints with supervised training, BEiT and MAE pre-training.
These checkpoints can be found on their official github pages. We also provide the links below.

#### model checkpoints supervised trained on ImageNet
Download the ViT-B/16 weights trained on ImageNet-22k:
- ```wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz```

See more details in https://github.com/google-research/vision_transformer.

#### model checkpoints self-supervised trained on ImageNet
Download BEiT weights pre-trained on ImageNet-22k: 
- ```wget https://unilm.blob.core.windows.net/beit/beit_base_patch16_224_pt22k.pth```
Download MAE weights pretrained on ImageNet-22k:
- ```wget https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth```

See more details in https://github.com/microsoft/unilm/tree/master/beit and https://github.com/facebookresearch/mae.
