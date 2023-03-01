### Data Preparation
In this paper, we conduct experiments on Retina, Derm and COVID-FL datasets. 

project/
├── src/
│   ├── index.html
│   ├── styles/
│   │   ├── style.css
│   ├── scripts/
│   │   ├── main.js
├── images/
│   ├── logo.png



├── data/
│   ├── Retina/
│   │   ├── 5_clients/
│   │   │   ├── split_1/
│   │   │   │   ├── client_1.csv
│   │   │   │   ├── client_2.csv
│   │   │   │   ├── client_3.csv
│   │   │   │   ├── client_4.csv
│   │   │   │   ├── client_5.csv
│   │   │   ├── split_2/
│   │   │   ├── split_3/
│   │   ├── central/
│   │   ├── test/
│   │   ├── train/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   ├── labels.csv
│   ├── COVID-FL/
│   ├── Derm/
│   ├── tokenizer_weight/
│   ├── ckpts/

More details about the datasets can be found in our paper.

The download links to the datasets are also provided below.

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
