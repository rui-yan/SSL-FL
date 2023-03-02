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

Each data folder contains 'n_clients' subfolders, each of which includes data split information in a .csv file. The .csv file contains the filenames of the images belonging to each client in the data split.

If you would like to train using your own custom datasets, please ensure that your data is organized according to the directory structure mentioned above. Additionally, you can modify the data augmentation strategies in SSL-FL/code/util/datasets and the data loader in SSL-FL/code/util/data_utils.py.

Below are the download links for the Retina, COVID-FL, and Derm datasets.
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">Retina</th>
<th valign="bottom">Derm</th>
<th valign="bottom">COVID-FL</th>
<th valign="bottom">Skin-FL</th>
<!-- TABLE BODY -->
<tr><td align="left">Link</td>
<td align="center"><a href="https://drive.google.com/file/d/1V5RR_VzfGdHCuI_am6uCohEqvKtjbeDY/view?usp=share_link">link</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1fDEKMyF9rHOMf4pY_q7Ys33uka4Z_kN6/view?usp=share_link">link</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1445S6t1jU0nhmE6HBhqs7p58ZKlt8nNS/view?usp=share_link">link</a></td>
<td align="center">TODO</td>
</tr>
</tbody></table>

### Use gdown to download data to your path (optional)
Step1: ```pip install gdown```
Step2: ```gdown https://drive.google.com/uc?id=<the_file_id>```
